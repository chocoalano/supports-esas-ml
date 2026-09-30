"""Put the STT model on disk before the Speech runtime starts, and prove it loads offline.

The Speech runtime runs with HF_HUB_OFFLINE=1: it never downloads anything, so
a model that is not on disk makes every transcription a 503. This script is the
one step in a deployment that is allowed to reach HuggingFace.

    sudo -u fsa /opt/fsa/.venv/bin/python scripts/provision_stt_model.py \\
        --env-file /etc/fsa/speech.env

It reads FSA_STT_MODEL, FSA_STT_MODEL_ROOT, FSA_STT_DEVICE and
FSA_STT_COMPUTE_TYPE from that file - the same file the service reads, so what
is provisioned is what will be loaded - then:

1. downloads the model into FSA_STT_MODEL_ROOT (a no-op when already there),
2. in a fresh interpreter with HF_HUB_OFFLINE=1, runs the service's own boot
   check, loads the model through the service's own engine, and transcribes a
   second of generated audio.

Exit status 0 means the Speech runtime will find and load this model offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def settings_from(env_file: str | None):
    from app.core.config import Settings

    # Role forced to speech: the file is validated as the Speech runtime will
    # validate it, whatever else it says.
    os.environ["FSA_RUNTIME_ROLE"] = "speech"
    return Settings(_env_file=env_file) if env_file else Settings()


def download(settings) -> Path:
    from faster_whisper.utils import download_model

    if Path(settings.stt_model).is_absolute():
        return Path(settings.stt_model)

    return Path(download_model(settings.stt_model, cache_dir=settings.stt_model_root))


def verify_offline(settings) -> dict:
    """Run in a child with HF_HUB_OFFLINE=1, so nothing it does can reach the network."""
    import time

    from app.services.speech.audio import inspect_audio
    from app.services.speech.providers.faster_whisper import (
        FasterWhisperEngine,
        provisioning_problem,
    )
    from app.services.speech.stt import TranscribeOptions

    if problem := provisioning_problem(settings):
        raise SystemExit(f"boot check failed: {problem}")

    engine = FasterWhisperEngine(settings)
    started = time.perf_counter()
    engine.load()
    loaded = time.perf_counter() - started

    with tempfile.TemporaryDirectory() as directory:
        clip = Path(directory) / "tone"
        write_tone(clip)
        inspect_audio(clip, max_seconds=settings.stt_max_audio_seconds)
        engine.transcribe(clip, TranscribeOptions(language=settings.stt_default_language))

    return {"load_seconds": round(loaded, 2)}


def write_tone(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    """A second of 440 Hz as 16-bit mono WAV - stdlib only, no test helpers deployed."""
    import math
    import struct
    import wave

    samples = (
        int(0.3 * 32767 * math.sin(2 * math.pi * 440 * index / rate))
        for index in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def describe(path: Path) -> dict:
    weights = path / "model.bin"
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return {
        "path": str(path),
        "files": sorted(item.name for item in path.iterdir()),
        "model_bin_bytes": weights.stat().st_size,
        "model_bin_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--env-file", help="The Speech runtime's env file.")
    parser.add_argument(
        "--verify-only", action="store_true", help="Do not download; only prove it loads."
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    settings = settings_from(args.env_file)

    if args.child:
        print(json.dumps(verify_offline(settings)))
        return

    if not settings.stt_model_root and not Path(settings.stt_model).is_absolute():
        parser.error(
            "FSA_STT_MODEL_ROOT is not set. Production keeps the model in a persistent "
            "directory it names - not in a home directory, not under /tmp."
        )

    if args.verify_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    path = download(settings)
    print(f"model '{settings.stt_model}' is at {path}", file=sys.stderr)

    command = [sys.executable, __file__, "--child"]
    if args.env_file:
        command += ["--env-file", args.env_file]
    child = subprocess.run(
        command,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
    )
    if child.returncode != 0:
        sys.stderr.write(child.stderr)
        raise SystemExit("offline verification failed: the Speech runtime would not load it")

    print(
        json.dumps(
            {
                "model": settings.stt_model,
                "model_root": settings.stt_model_root,
                "compute_type": settings.stt_compute_type,
                **describe(path),
                "offline_verification": json.loads(child.stdout.strip().splitlines()[-1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
