"""Measure what a speech-to-text model costs and how well it transcribes, on this host.

Software characteristics, per model and configuration - the data a deployment
sizes itself from, not a sizing decision. Measured through this service's own
code - the audio validator and `FasterWhisperEngine`, exactly as a request
runs them:

- model load time, the first (cold) transcription, then N warm repetitions;
- latency modelled as `fixed + per_audio_second x duration`, fitted on warm
  medians, with the prediction error for every clip and the cost at the cap;
- resident memory idle / model loaded / peak during a transcription / after,
  and CPU in use (cores) sampled during each transcription;
- WER and CER against a reference, plus the word-level differences to read.

One model at a time, one transcription at a time (FSA_STT_MAX_CONCURRENT=1),
each model in a fresh interpreter so its memory is its own. Models are fetched
once, then measured with HF_HUB_OFFLINE=1: downloading is never timed.

Samples: one folder per category, audio inside, and for any file an optional
`<name>.txt` with what was said (scripts/build_stt_corpus.py builds one):

    .venv/bin/python scripts/benchmark_stt.py corpus/ \\
        --models base small --cpu-threads 2 4 --model-root /var/lib/fsa/models \\
        --language id --repeat 3 --output benchmark.md --json benchmark.json

Latency belongs to the host it was measured on. Accuracy does not: the same
model and compute type produce the same transcript on any CPU.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".flac", ".webm", ".3gp"}


# -- the host ----------------------------------------------------------------------


def sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def host() -> dict:
    info: dict = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
    }
    release = Path("/etc/os-release")
    if release.exists():
        fields = dict(
            line.split("=", 1) for line in release.read_text().splitlines() if "=" in line
        )
        info["distribution"] = fields.get("PRETTY_NAME", "").strip('"')

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        text = cpuinfo.read_text()
        models = re.findall(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        info["cpu_model"] = models[0] if models else None
        cores = {
            (block.get("physical id"), block.get("core id"))
            for block in (
                dict(
                    (key.strip(), value.strip())
                    for key, _, value in (line.partition(":") for line in chunk.splitlines())
                )
                for chunk in text.strip().split("\n\n")
            )
        }
        info["physical_cores"] = len(cores) or None
        memory = re.search(r"^MemTotal:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
        info["ram_gb"] = round(int(memory.group(1)) / 1024**2, 1) if memory else None
    elif sys.platform == "darwin":
        info["cpu_model"] = sysctl("machdep.cpu.brand_string")
        info["physical_cores"] = int(sysctl("hw.physicalcpu") or 0) or None
        info["ram_gb"] = round(int(sysctl("hw.memsize") or 0) / 1024**3, 1)

    for package in ("faster_whisper", "ctranslate2", "av", "numpy", "onnxruntime"):
        try:
            info[package] = getattr(__import__(package), "__version__", "?")
        except ImportError:
            info[package] = None
    return info


# -- memory and CPU ------------------------------------------------------------------


def rss_mb() -> float:
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**2
    output = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    )
    return int(output.stdout.strip()) / 1024


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024  # bytes vs kB


class Sampler:
    """Samples resident memory and CPU in use while a transcription runs."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.peak_rss = 0.0
        self.cores: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        last_cpu, last_wall = time.process_time(), time.perf_counter()
        while not self._stop.wait(self.interval):
            self.peak_rss = max(self.peak_rss, rss_mb())
            cpu, wall = time.process_time(), time.perf_counter()
            self.cores.append((cpu - last_cpu) / max(wall - last_wall, 1e-6))
            last_cpu, last_wall = cpu, wall
        self.peak_rss = max(self.peak_rss, rss_mb())

    @property
    def peak_cores(self) -> float | None:
        # Over ~0.5 s windows: one 100 ms tick is too noisy to call a peak.
        if not self.cores:
            return None
        window = 5
        smoothed = [
            statistics.fmean(self.cores[i : i + window])
            for i in range(max(1, len(self.cores) - window + 1))
        ]
        return round(max(smoothed), 2)

    @property
    def mean_cores(self) -> float | None:
        return round(statistics.fmean(self.cores), 2) if self.cores else None


# -- quality -----------------------------------------------------------------------------


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def edit_distance(ref: list, hyp: list) -> int:
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1]


def error_rates(reference: str, hypothesis: str) -> tuple[float, float]:
    ref, hyp = normalise(reference), normalise(hypothesis)
    wer = edit_distance(ref.split(), hyp.split()) / max(1, len(ref.split()))
    cer = edit_distance(list(ref), list(hyp)) / max(1, len(ref))
    return round(wer, 3), round(cer, 3)


def word_diff(reference: str, hypothesis: str) -> list[str]:
    """What was said against what was written, only where they differ."""
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    return [
        f"{' '.join(ref[i1:i2]) or '∅'} → {' '.join(hyp[j1:j2]) or '∅'}"
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


# -- one model, in its own process ------------------------------------------------


def run_worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(ROOT))

    from app.core.config import Settings
    from app.services.speech.audio import inspect_audio
    from app.services.speech.providers.faster_whisper import FasterWhisperEngine
    from app.services.speech.stt import TranscribeOptions

    settings = Settings(
        _env_file=None,
        stt_model=args.model,
        stt_model_root=args.model_root,
        stt_compute_type=args.compute_type,
        stt_cpu_threads=args.threads,
        stt_beam_size=args.beam_size,
    )
    engine = FasterWhisperEngine(settings)
    report: dict = {
        "model": args.model,
        "compute_type": args.compute_type,
        "cpu_threads": args.threads,
        "rss_idle_mb": round(rss_mb(), 1),
    }

    started = time.perf_counter()
    engine.load()
    report["load_seconds"] = round(time.perf_counter() - started, 2)
    report["rss_loaded_mb"] = round(rss_mb(), 1)
    report["peak_rss_during_load_mb"] = round(peak_rss_mb(), 1)

    if args.prefetch:
        print(json.dumps(report))
        return

    options = TranscribeOptions(language=args.language)

    def transcribe(path: Path) -> tuple[float, str, str | None, Sampler]:
        with Sampler() as sampler:
            started = time.perf_counter()
            inspect_audio(path, max_seconds=args.max_audio_seconds)
            transcript = engine.transcribe(path, options)
            elapsed = time.perf_counter() - started
        return elapsed, transcript.text, transcript.language, sampler

    samples = json.loads(Path(args.samples_json).read_text())

    # The first transcription after load pays one-off costs; it is reported
    # apart and kept out of the warm figures.
    cold_path = Path(samples[0]["path"])
    cold, _, _, _ = transcribe(cold_path)
    report["cold_first"] = {"sample": cold_path.name, "seconds": round(cold, 2)}

    results = []
    for sample in samples:
        path = Path(sample["path"])
        runs = [transcribe(path) for _ in range(args.repeat)]
        latencies = [run[0] for run in runs]
        text, language = runs[-1][1], runs[-1][2]
        duration = inspect_audio(path, max_seconds=args.max_audio_seconds).duration_seconds
        reference = sample["reference"]
        wer, cer = error_rates(reference, text) if reference else (None, None)
        peaks = [run[3].peak_cores for run in runs if run[3].peak_cores is not None]
        results.append(
            {
                **sample,
                "duration_seconds": duration,
                "warm_seconds": [round(value, 2) for value in latencies],
                "latency_seconds": round(statistics.median(latencies), 2),
                "rtf": round(statistics.median(latencies) / duration, 3),
                "peak_rss_mb": round(max(run[3].peak_rss for run in runs), 1),
                "peak_cpu_cores": max(peaks, default=None),
                "mean_cpu_cores": runs[-1][3].mean_cores,
                "wer": wer,
                "cer": cer,
                "language": language,
                "text": text,
                "differences": word_diff(reference, text) if reference else [],
            }
        )

    report["samples"] = results
    report["peak_rss_transcribing_mb"] = max(result["peak_rss_mb"] for result in results)
    report["rss_after_mb"] = round(rss_mb(), 1)
    # The whole life of the process, load included - not only while transcribing.
    report["rss_process_peak_mb"] = round(peak_rss_mb(), 1)
    print(json.dumps(report))


# -- the latency model -------------------------------------------------------------------


def fit_cost(samples: list[dict]) -> tuple[float, float] | None:
    """latency ~= fixed + per_second x duration, least squares on warm medians.

    A short clip's real-time factor is mostly fixed cost, so a worst-case RTF
    projected onto a five-minute clip overstates it several times over.
    """
    points = [(s["duration_seconds"], s["latency_seconds"]) for s in samples]
    if len({round(duration) for duration, _ in points}) < 2:
        return None
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    per_second = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread
    return max(0.0, mean_y - per_second * mean_x), per_second


# -- orchestration ----------------------------------------------------------------------


def discover(samples: Path) -> list[dict]:
    found = []
    for path in sorted(samples.rglob("*")):
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        reference = path.with_suffix(".txt")
        found.append(
            {
                "path": str(path),
                "category": path.parent.name if path.parent != samples else "-",
                "reference": reference.read_text().strip() if reference.exists() else None,
            }
        )
    return found


def run_model(args: argparse.Namespace, model: str, threads: int, samples_json: Path) -> dict:
    command = [
        sys.executable, __file__, "--worker", "--model", model, "--threads", str(threads),
        "--model-root", args.model_root, "--compute-type", args.compute_type,
        "--beam-size", str(args.beam_size), "--max-audio-seconds", str(args.max_audio_seconds),
        "--repeat", str(args.repeat), "--samples-json", str(samples_json),
    ]  # fmt: skip
    if args.language:
        command += ["--language", args.language]

    # Fetch once (not timed), then measure offline, as production runs.
    subprocess.run([*command, "--prefetch"], check=True, capture_output=True, text=True)
    measured = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    return json.loads(measured.stdout.strip().splitlines()[-1])


def mean(values) -> float | None:  # noqa: ANN001
    values = [value for value in values if value is not None]
    return round(statistics.fmean(values), 3) if values else None


def render(reports: list[dict], args: argparse.Namespace, machine: dict) -> str:
    lines = ["# STT benchmark", "", "## Host", ""]
    lines += [f"- {key}: `{value}`" for key, value in machine.items()]
    lines += [
        f"- beam_size={args.beam_size}, language={args.language or 'detect'}, concurrency=1, "
        f"warm repetitions={args.repeat}, "
        f"FSA_STT_MAX_AUDIO_SECONDS={args.max_audio_seconds:g}",
        "",
        "## Summary",
        "",
        "| model | threads | load s | cold first s | RSS idle MB | RSS loaded MB "
        "| RSS peak transcribing MB | RSS process peak MB | RSS after MB | peak CPU cores "
        "| fixed s | s per audio s | at cap | mean WER | mean CER |",
        "|" + "---|" * 15,
    ]
    for report in reports:
        samples = report["samples"]
        cost = fit_cost(samples)
        fixed, per_second = cost if cost else (None, None)
        at_cap = "-"
        if cost:
            at_cap = f"{fixed + per_second * args.max_audio_seconds:.0f}s"
        peak_cores = max((s["peak_cpu_cores"] or 0) for s in samples)
        lines.append(
            f"| {report['model']} | {report['cpu_threads']} | {report['load_seconds']} "
            f"| {report['cold_first']['seconds']} | {report['rss_idle_mb']} "
            f"| {report['rss_loaded_mb']} | {report['peak_rss_transcribing_mb']} "
            f"| {report['rss_process_peak_mb']} | {report['rss_after_mb']} | {peak_cores} "
            f"| {fixed and round(fixed, 2)} | {per_second and round(per_second, 4)} | {at_cap} "
            f"| {mean(s['wer'] for s in samples)} | {mean(s['cer'] for s in samples)} |"
        )

    lines += [
        "",
        "`at cap` = predicted time for one clip of FSA_STT_MAX_AUDIO_SECONDS on this host,",
        "from the fitted `fixed + per audio second` cost.",
        "",
        "## Latency model: observed against predicted (warm medians)",
        "",
        "| model | threads | clip | audio s | warm runs s | observed s | predicted s | error % "
        "| RTF |",
        "|" + "---|" * 9,
    ]
    for report in reports:
        cost = fit_cost(report["samples"])
        for s in sorted(report["samples"], key=lambda item: item["duration_seconds"]):
            predicted = cost[0] + cost[1] * s["duration_seconds"] if cost else None
            error = (
                round(100 * (s["latency_seconds"] - predicted) / predicted, 1)
                if predicted
                else None
            )
            lines.append(
                f"| {report['model']} | {report['cpu_threads']} "
                f"| {s['category']}/{Path(s['path']).name} | {s['duration_seconds']} "
                f"| {', '.join(map(str, s['warm_seconds']))} | {s['latency_seconds']} "
                f"| {predicted and round(predicted, 2)} | {error} | {s['rtf']} |"
            )

    lines += ["", "## Quality by category", ""]
    lines += ["| model | threads | category | n | mean WER | mean CER |", "|" + "---|" * 6]
    for report in reports:
        categories: dict[str, list[dict]] = {}
        for sample in report["samples"]:
            categories.setdefault(sample["category"], []).append(sample)
        for category, items in categories.items():
            lines.append(
                f"| {report['model']} | {report['cpu_threads']} | {category} | {len(items)} "
                f"| {mean(s['wer'] for s in items)} | {mean(s['cer'] for s in items)} |"
            )

    if args.show_text:
        lines += ["", "## Transcripts", ""]
        seen: set[str] = set()
        for report in reports:
            if report["model"] in seen:  # the text does not depend on thread count
                continue
            seen.add(report["model"])
            lines += [f"### {report['model']}", ""]
            for s in report["samples"]:
                lines += [
                    f"**{s['category']}/{Path(s['path']).name}** "
                    f"({s['duration_seconds']} s, WER {s['wer']}, CER {s['cer']})",
                    "",
                    f"- expected: {s['reference']}",
                    f"- actual: {s['text']}",
                    f"- differences: {'; '.join(s['differences']) or 'none'}",
                    "",
                ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("samples", nargs="?", type=Path)
    parser.add_argument("--models", nargs="+", default=["small"])
    parser.add_argument("--cpu-threads", nargs="+", type=int, default=[0])
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--language", default=None)
    parser.add_argument("--repeat", type=int, default=3, help="Warm runs per clip.")
    parser.add_argument("--max-audio-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", type=Path, help="Also write the raw measurements here.")
    parser.add_argument("--show-text", action="store_true", help="Include transcripts.")
    # Internal: one model in a fresh interpreter.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prefetch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--samples-json", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    if args.samples is None or not args.samples.is_dir():
        parser.error("samples must be a directory of audio files")
    samples = discover(args.samples)
    if not samples:
        parser.error(f"no audio files under {args.samples}")

    samples_json = Path(args.model_root) / ".benchmark-samples.json"
    samples_json.parent.mkdir(parents=True, exist_ok=True)
    samples_json.write_text(json.dumps(samples))

    machine = host()
    reports = []
    for model in args.models:
        for threads in args.cpu_threads:
            print(f"benchmarking {model} threads={threads}...", file=sys.stderr)
            reports.append(run_model(args, model, threads, samples_json))

    rendered = render(reports, args, machine)
    if args.output:
        args.output.write_text(rendered)
    if args.json:
        args.json.write_text(
            json.dumps({"host": machine, "reports": reports}, indent=2, ensure_ascii=False)
        )
    print(rendered)


if __name__ == "__main__":
    main()
