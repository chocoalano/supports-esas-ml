"""Benchmark speech-to-text models on the host that will run them.

Measures what choosing FSA_STT_MODEL and MemoryMax depends on - memory, latency,
real-time factor and, where a reference transcript is given, word error rate -
through this service's own code: the audio validator and `FasterWhisperEngine`,
exactly as a request runs them. One model at a time, one transcription at a
time (FSA_STT_MAX_CONCURRENT=1), each model in a fresh interpreter so its
memory figures are its own.

Samples: one folder per category, audio files inside, and for any file an
optional `<name>.txt` next to it holding what was actually said:

    samples/
      jelas_pria/            rekaman-01.m4a   rekaman-01.txt
      percakapan_wanita/     rekaman-02.webm  rekaman-02.txt
      bising_ringan/         ...
      panjang/               ...

Usage, from the repository root:

    .venv/bin/python scripts/benchmark_stt.py samples/ \\
        --models base small medium --model-root /var/lib/fsa/models \\
        --cpu-threads 2 --language id --output benchmark.md

Numbers from any machine other than the production host are not production
numbers. Word error rate penalises "8" against "delapan"; read the transcripts
(--show-text) before trusting it for Indonesian.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".flac", ".webm", ".3gp"}


# -- measuring ------------------------------------------------------------------


def rss_mb() -> float:
    """Resident set size now, as `ps` reports it."""
    output = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    )
    return int(output.stdout.strip()) / 1024


def peak_rss_mb() -> float:
    """Highest resident set size this process has reached."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Kilobytes on Linux, bytes on macOS.
    return peak / 1024 / 1024 if sys.platform == "darwin" else peak / 1024


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return None

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            current[j] = min(
                previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)
            )
        previous = current

    return previous[-1] / len(ref)


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
        stt_cpu_threads=args.cpu_threads,
        stt_beam_size=args.beam_size,
    )
    engine = FasterWhisperEngine(settings)
    report: dict = {"model": args.model, "rss_idle_mb": round(rss_mb(), 1)}

    started = time.perf_counter()
    engine.load()
    report["load_seconds"] = round(time.perf_counter() - started, 2)
    report["rss_loaded_mb"] = round(rss_mb(), 1)

    if args.prefetch:
        print(json.dumps(report))
        return

    options = TranscribeOptions(language=args.language)
    results = []
    for sample in json.loads(Path(args.samples_json).read_text()):
        path = Path(sample["path"])
        started = time.perf_counter()
        audio = inspect_audio(path, max_seconds=args.max_audio_seconds)
        transcript = engine.transcribe(path, options)
        elapsed = time.perf_counter() - started
        results.append(
            {
                **sample,
                "duration_seconds": audio.duration_seconds,
                "latency_seconds": round(elapsed, 2),
                "rtf": round(elapsed / audio.duration_seconds, 3),
                "wer": (
                    None
                    if sample["reference"] is None
                    else round(word_error_rate(sample["reference"], transcript.text), 3)
                ),
                "language": transcript.language,
                "text": transcript.text,
                "rss_peak_mb_so_far": round(peak_rss_mb(), 1),
            }
        )

    report["samples"] = results
    report["rss_peak_mb"] = round(peak_rss_mb(), 1)
    report["rss_after_mb"] = round(rss_mb(), 1)
    print(json.dumps(report))


# -- orchestration ------------------------------------------------------------------


def run_model(args: argparse.Namespace, model: str, samples_json: Path) -> dict:
    command = [
        sys.executable,
        __file__,
        "--worker",
        "--model",
        model,
        "--model-root",
        args.model_root,
        "--compute-type",
        args.compute_type,
        "--cpu-threads",
        str(args.cpu_threads),
        "--beam-size",
        str(args.beam_size),
        "--max-audio-seconds",
        str(args.max_audio_seconds),
        "--samples-json",
        str(samples_json),
    ]
    if args.language:
        command += ["--language", args.language]

    # Once to download, so the measured run times loading - not the network -
    # and then offline, as production should run.
    subprocess.run([*command, "--prefetch"], check=True, capture_output=True, text=True)
    measured = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    return json.loads(measured.stdout.strip().splitlines()[-1])


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def fit_cost(samples: list[dict]) -> tuple[float, float] | None:
    """latency ~= fixed + per_second * duration, by least squares.

    A short clip's real-time factor is mostly fixed overhead, so projecting the
    worst RTF onto a five-minute clip overstates it several times over. Needs
    clips of at least two different lengths - include short and long ones.
    """
    points = [(s["duration_seconds"], s["latency_seconds"]) for s in samples]
    if len({round(duration) for duration, _ in points}) < 2:
        return None

    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    per_second = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread

    return max(0.0, mean_y - per_second * mean_x), per_second


def render(reports: list[dict], args: argparse.Namespace) -> str:
    lines = [
        "# STT benchmark",
        "",
        f"- host: `{os.uname().sysname} {os.uname().machine}`, python {sys.version.split()[0]}",
        f"- compute_type={args.compute_type}, cpu_threads={args.cpu_threads}, "
        f"beam_size={args.beam_size}, language={args.language or 'detect'}, concurrency=1",
        f"- gateway timeout budget: {args.gateway_timeout:g}s, "
        f"FSA_STT_MAX_AUDIO_SECONDS: {args.max_audio_seconds:g}s",
        "",
        "| model | load s | RSS idle MB | RSS loaded MB | RSS peak MB | RSS after MB "
        "| mean RTF | worst RTF | mean WER | fixed s + s per audio s | at cap |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for report in reports:
        samples = report["samples"]
        rtfs = [s["rtf"] for s in samples]
        wers = [s["wer"] for s in samples if s["wer"] is not None]
        cost = fit_cost(samples)
        if cost is None:
            model_cost, at_cap = "need 2+ lengths", "-"
        else:
            fixed, per_second = cost
            predicted = fixed + per_second * args.max_audio_seconds
            model_cost = f"{fixed:.2f} + {per_second:.3f}"
            at_cap = f"{predicted:.0f}s {'OK' if predicted < args.gateway_timeout else 'OVER'}"
        lines.append(
            f"| {report['model']} | {report['load_seconds']} | {report['rss_idle_mb']} "
            f"| {report['rss_loaded_mb']} | {report['rss_peak_mb']} | {report['rss_after_mb']} "
            f"| {mean(rtfs)} | {max(rtfs, default=None)} | {mean(wers)} "
            f"| {model_cost} | {at_cap} |"
        )

    lines += ["", "## Per category", ""]
    lines += [
        "| model | category | n | mean latency s | mean RTF | mean WER |",
        "|---|---|---|---|---|---|",
    ]
    for report in reports:
        categories: dict[str, list[dict]] = {}
        for sample in report["samples"]:
            categories.setdefault(sample["category"], []).append(sample)
        for category, items in categories.items():
            lines.append(
                f"| {report['model']} | {category} | {len(items)} "
                f"| {mean([s['latency_seconds'] for s in items])} "
                f"| {mean([s['rtf'] for s in items])} "
                f"| {mean([s['wer'] for s in items if s['wer'] is not None])} |"
            )

    lines += ["", "## Per sample", ""]
    header = "| model | sample | seconds | latency s | RTF | WER | language |"
    columns = 8 if args.show_text else 7
    lines += [header + (" text |" if args.show_text else ""), "|---" * columns + "|"]
    for report in reports:
        for s in report["samples"]:
            row = (
                f"| {report['model']} | {Path(s['path']).name} | {s['duration_seconds']} "
                f"| {s['latency_seconds']} | {s['rtf']} | {s['wer']} | {s['language']} |"
            )
            text = s["text"].replace("|", "\\|")
            lines.append(row + (f" {text} |" if args.show_text else ""))

    lines += [
        "",
        "`at cap` is the fitted cost of one clip of FSA_STT_MAX_AUDIO_SECONDS, measured by",
        "extrapolation from the clips above - include long ones, or it is a guess. OVER means",
        "the gateway times out first: lower the cap, fix CPU, or change model, in that order,",
        "before anyone considers raising the timeout. RSS peak is the MemoryMax input.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("samples", nargs="?", type=Path)
    parser.add_argument("--models", nargs="+", default=["small"])
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-audio-seconds", type=float, default=300.0)
    parser.add_argument("--gateway-timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", type=Path, help="Also write the raw measurements here.")
    parser.add_argument("--show-text", action="store_true", help="Include transcripts.")
    # Internal: one model in a fresh interpreter.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prefetch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
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

    reports = []
    for model in args.models:
        print(f"benchmarking {model} on {len(samples)} samples...", file=sys.stderr)
        reports.append(run_model(args, model, samples_json))

    rendered = render(reports, args)
    if args.output:
        args.output.write_text(rendered)
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    print(rendered)


if __name__ == "__main__":
    main()
