"""Build an Indonesian STT benchmark corpus from open recordings, for scripts/benchmark_stt.py.

Real people, with what they said as the reference. Two sources:

- FLEURS (Conneau et al., 2022), `google/fleurs`, `id_id`, CC-BY-4.0: modern
  read sentences with clean references. Every Indonesian speaker in it is male
  (FLEURS says so, and pitch agrees), so it supplies the male, mixed-English
  and long clips.
- LibriVox Indonesia, `indonesian-nlp/librivox-indonesia` (public-domain
  LibriVox audiobooks): the female voices. Its text is in pre-1947 spelling
  ("itoe", "djadi"), converted here to modern spelling by the standard rules,
  so word error rates on these clips are approximate - archaic words remain.

What it is not, and the benchmark report must say so:
- read speech, not conversation: nobody hesitates, interrupts or trails off;
- recorded quietly: "light noise" is white noise mixed in at 20 dB SNR;
- gender is estimated from median pitch (below 160 Hz "pria", above 180 Hz
  "wanita"), because FLEURS labels are not usable - a heuristic, reported as one;
- long clips are consecutive FLEURS utterances joined with half a second of
  silence: several speakers, not one monologue.

    .venv/bin/python scripts/build_stt_corpus.py corpus/ --cache .corpus-cache
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tarfile
import urllib.request
import wave
from itertools import zip_longest
from pathlib import Path

import numpy as np

FLEURS_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/id_id"
LIBRIVOX_URL = "https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia/resolve/main/data"
RATE = 16000
GAP = np.zeros(int(0.5 * RATE), dtype=np.int16)
LIBRIVOX = {"source": "librivox", "reference_spelling": "converted from pre-1947"}

#: Words that mark a sentence as mixing in English terms.
ENGLISH = re.compile(
    r"\b(the|of|and|for|online|internet|website|software|email|smartphone|game|festival"
    r"|world|cup|club|team|park|center|university|airport|google|facebook|twitter|did|not"
    r"|finish|super|bowl|award|grand|prix|open|championship)\b",
    re.IGNORECASE,
)

#: category -> how many clips; durations are target seconds for joined clips.
SHORT_PER_GENDER = 4
MIXED = 3
NOISY_PER_GENDER = 2
LONG_TARGETS = {"durasi_030": 30.0, "durasi_060": 60.0, "durasi_180": 180.0, "durasi_300": 300.0}
#: The 300 s clip must fit under FSA_STT_MAX_AUDIO_SECONDS=300, not merely near it.
LONG_CEILING = 299.0


def fetch(base: str, name: str, cache: Path) -> Path:
    target = cache / f"{base.split('/')[5]}_{name.replace('/', '_')}"
    if not target.exists():
        print(f"downloading {name}...", file=sys.stderr)
        partial = target.with_suffix(".partial")
        with urllib.request.urlopen(f"{base}/{name}") as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        partial.rename(target)
    return target


def read_rows(tsv: Path) -> list[dict]:
    rows = []
    with tsv.open(newline="") as handle:
        for line in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append(
                {
                    "sentence_id": line[0],
                    "file": line[1],
                    "text": line[2].strip(),
                    "seconds": int(line[5]) / RATE,
                }
            )
    return rows


def load_audio(archive: Path, names: set[str]) -> dict[str, np.ndarray]:
    audio: dict[str, np.ndarray] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = member.name.rsplit("/", 1)[-1]
            if name not in names:
                continue
            audio[name] = decode(tar.extractfile(member))
    return audio


def decode(handle) -> np.ndarray:  # noqa: ANN001 - a file object
    """16 kHz mono int16. FLEURS ships 32-bit float WAV, which `wave` cannot read."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    chunks = []
    with av.open(handle) as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
    for resampled in resampler.resample(None):
        chunks.append(resampled.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.int16)


def median_pitch(samples: np.ndarray) -> float | None:
    """Median F0 over voiced 40 ms frames, by autocorrelation, 70-350 Hz."""
    signal = samples.astype(np.float32) / 32768.0
    frame, hop = int(0.04 * RATE), int(0.02 * RATE)
    low_lag, high_lag = RATE // 350, RATE // 70
    threshold = 0.3 * np.sqrt(np.mean(signal**2))
    pitches = []

    for start in range(0, len(signal) - frame, hop):
        window = signal[start : start + frame]
        if np.sqrt(np.mean(window**2)) < threshold:
            continue
        window = window - window.mean()
        corr = np.correlate(window, window, mode="full")[frame - 1 :]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        # Any smooth signal correlates highly with itself at small lags, so the
        # search starts past that first lobe - after the first zero crossing -
        # or it always "finds" the shortest period allowed.
        crossings = np.flatnonzero(corr[:high_lag] < 0)
        if not len(crossings):
            continue
        start = max(low_lag, int(crossings[0]))
        lag = start + int(np.argmax(corr[start:high_lag]))
        if corr[lag] > 0.45:  # clearly periodic: voiced
            pitches.append(RATE / lag)

    return float(np.median(pitches)) if len(pitches) >= 10 else None


def gender_of(pitch: float | None) -> str | None:
    if pitch is None:
        return None
    if pitch < 160:
        return "pria"
    if pitch > 180:
        return "wanita"
    return None


def with_noise(samples: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    signal = samples.astype(np.float32)
    power = np.mean(signal**2)
    noise = np.random.default_rng(seed).normal(size=len(signal)).astype(np.float32)
    noise *= np.sqrt(power / (10 ** (snr_db / 10)) / np.mean(noise**2))
    return np.clip(signal + noise, -32768, 32767).astype(np.int16)


def write_clip(folder: Path, name: str, samples: np.ndarray, reference: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with wave.open(str(folder / f"{name}.wav"), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(samples.tobytes())
    (folder / f"{name}.txt").write_text(reference + "\n")


def build(output: Path, cache: Path, split: str) -> list[dict]:
    rows = read_rows(fetch(FLEURS_URL, f"{split}.tsv", cache))
    archive = fetch(FLEURS_URL, f"audio/{split}.tar.gz", cache)
    audio = load_audio(archive, {r["file"] for r in rows})
    female = librivox_female(cache)
    for row in rows:
        row["pitch_hz"] = median_pitch(audio[row["file"]])
        row["gender"] = gender_of(row["pitch_hz"])

    manifest: list[dict] = []
    used: set[str] = set()

    def add(category: str, name: str, samples: np.ndarray, sources: list[dict], **extra):
        extra.setdefault("source", "fleurs")
        reference = " ".join(source["text"] for source in sources)
        write_clip(output / category, name, samples, reference)
        used.update(source["file"] for source in sources)
        manifest.append(
            {
                "category": category,
                "file": f"{category}/{name}.wav",
                "seconds": round(len(samples) / RATE, 2),
                "sources": [source["file"] for source in sources],
                "pitch_hz": [round(s["pitch_hz"]) if s["pitch_hz"] else None for s in sources],
                **extra,
            }
        )

    def pick(predicate, count: int) -> list[dict]:
        chosen = [r for r in rows if r["file"] not in used and predicate(r)][:count]
        used.update(r["file"] for r in chosen)
        return chosen

    for index, row in enumerate(
        pick(lambda r: r["gender"] == "pria" and 8 <= r["seconds"] <= 10.5, SHORT_PER_GENDER)
    ):
        add("pendek_pria", f"{index + 1:02d}", audio[row["file"]], [row])

    for index, clip in enumerate(female[:SHORT_PER_GENDER]):
        add("pendek_wanita", f"{index + 1:02d}", clip["audio"], clip["sources"], **LIBRIVOX)

    mixed = pick(lambda r: bool(ENGLISH.search(r["text"])) and r["seconds"] <= 20, MIXED)
    for index, row in enumerate(mixed):
        add("campuran_inggris", f"{index + 1:02d}", audio[row["file"]], [row])

    for index, row in enumerate(
        pick(lambda r: r["gender"] == "pria" and 8 <= r["seconds"] <= 12, NOISY_PER_GENDER)
    ):
        noisy = with_noise(audio[row["file"]], snr_db=20.0, seed=index)
        add("bising_ringan", f"pria_{index + 1:02d}", noisy, [row], snr_db=20.0)

    offset = SHORT_PER_GENDER
    for index, clip in enumerate(female[offset : offset + NOISY_PER_GENDER]):
        noisy = with_noise(clip["audio"], snr_db=20.0, seed=100 + index)
        name = f"wanita_{index + 1:02d}"
        add("bising_ringan", name, noisy, clip["sources"], snr_db=20.0, **LIBRIVOX)

    # Long clips: distinct sentences, so a repeated reading does not reward a
    # model that skips it.
    seen_sentences: set[str] = set()
    for category, target in LONG_TARGETS.items():
        ceiling = min(target + 3.0, LONG_CEILING)
        sources, parts, length = [], [], 0.0
        for row in rows:
            if row["file"] in used or row["sentence_id"] in seen_sentences:
                continue
            if length + row["seconds"] + 0.5 > ceiling:
                continue
            sources.append(row)
            parts += [audio[row["file"]], GAP]
            length += row["seconds"] + 0.5
            seen_sentences.add(row["sentence_id"])
            if length >= target - 3.0:
                break
        add(category, "01", np.concatenate(parts[:-1]), sources)

    return manifest


#: Pre-1947 Indonesian spelling -> today's, in an order where no rule feeds another.
OLD_SPELLING = (
    ("dj", "\x00"),
    ("tj", "c"),
    ("nj", "ny"),
    ("sj", "sy"),
    ("ch", "kh"),
    ("j", "y"),
    ("\x00", "j"),
    ("oe", "u"),
    ("é", "e"),
)


def modern_spelling(text: str) -> str:
    text = text.lower()
    for old, new in OLD_SPELLING:
        text = text.replace(old, new)
    return text


def librivox_female(cache: Path, per_reader: int = 3) -> list[dict]:
    """8-10.5 s clips of women reading, each consecutive sentences of one chapter.

    Readers are classed by the median pitch of all their clips, not per clip:
    one short sentence is too little to judge a voice by.
    """
    import gzip
    import io

    metadata = fetch(LIBRIVOX_URL, "metadata_test.csv.gz", cache)
    with gzip.open(metadata) as handle:
        rows = {
            row["path"]: row
            for row in csv.DictReader(io.TextIOWrapper(handle, "utf-8"))
            if row["language"] == "ind"
        }
    clips: dict[str, list[dict]] = {}
    with tarfile.open(fetch(LIBRIVOX_URL, "audio_test.tgz", cache), "r:gz") as tar:
        for member in tar:
            key = member.name.split("/", 1)[-1]
            if member.isfile() and key in rows:
                samples = decode(tar.extractfile(member))
                clips.setdefault(rows[key]["reader"], []).append(
                    {
                        "file": key,
                        "text": modern_spelling(rows[key]["sentence"]),
                        "seconds": len(samples) / RATE,
                        "pitch_hz": median_pitch(samples),
                        "samples": samples,
                    }
                )

    by_reader: list[list[dict]] = []
    for items in sorted(clips.values(), key=len, reverse=True):
        pitches = [item["pitch_hz"] for item in items if item["pitch_hz"]]
        if not pitches or gender_of(float(np.median(pitches))) != "wanita":
            continue
        items.sort(key=lambda item: item["file"])
        joined: list[dict] = []
        run, length = [], 0.0
        for item in items:
            chapter = item["file"].rsplit("_", 1)[0]
            if run and run[-1]["file"].rsplit("_", 1)[0] != chapter:
                run, length = [], 0.0
            run.append(item)
            length += item["seconds"] + 0.3
            if length > 10.5:
                run, length = [item], item["seconds"]
            if 8.0 <= length <= 10.5:
                parts = []
                for piece in run:
                    parts += [piece["samples"], np.zeros(int(0.3 * RATE), dtype=np.int16)]
                joined.append({"audio": np.concatenate(parts[:-1]), "sources": list(run)})
                run, length = [], 0.0
                if len(joined) == per_reader:
                    break
        by_reader.append(joined)

    # Round robin across readers, so four clips are not one voice.
    return [clip for turn in zip_longest(*by_reader) for clip in turn if clip is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".corpus-cache"))
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    manifest = build(args.output, args.cache, args.split)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "sources": [
                    "FLEURS id_id (google/fleurs), CC-BY-4.0",
                    "LibriVox Indonesia (indonesian-nlp/librivox-indonesia), public domain audio",
                ],
                "split": args.split,
                "clips": manifest,
            },
            indent=2,
        )
    )
    for clip in manifest:
        print(f"{clip['file']:32s} {clip['seconds']:7.2f}s  pitch {clip['pitch_hz']}")


if __name__ == "__main__":
    main()
