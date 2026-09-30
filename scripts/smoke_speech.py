"""Speech for real, configured as the Speech runtime runs in production.

FSA_RUNTIME_ROLE=speech, both engines on, HF_HUB_OFFLINE=1 - so the STT model
must already be provisioned (scripts/provision_stt_model.py). Through the HTTP
API it checks:

- TTS: the default voice and another configured alias each come back as MP3
  that PyAV decodes, with a positive duration;
- STT: a real Indonesian recording (and the TTS output) is transcribed;
- network: an audit hook records every socket connection this process makes.
  Synthesis must reach the provider; transcription must reach nothing;
- isolation: OpenCV and InsightFace are never imported.

    HF_HUB_OFFLINE=1 FSA_STT_MODEL_ROOT=/var/lib/fsa/models \\
        .venv/bin/python scripts/smoke_speech.py --audio clip.wav --reference clip.txt \\
        --output speech.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--audio", type=Path, required=True, help="Indonesian recording.")
    parser.add_argument("--reference", type=Path, help="What is said in it.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("run with HF_HUB_OFFLINE=1, as the Speech runtime does")
    os.environ["FSA_RUNTIME_ROLE"] = "speech"
    os.environ["FSA_STT_ENABLED"] = "true"
    os.environ["FSA_TTS_ENABLED"] = "true"
    os.environ.setdefault("FSA_API_KEYS", "smoke:smoke-key")
    os.environ.setdefault("FSA_STT_DEFAULT_LANGUAGE", "id")

    phase = {"name": "boot"}
    connections: list[dict] = []

    def audit(event: str, event_args: tuple) -> None:
        if event == "socket.connect":
            connections.append({"phase": phase["name"], "address": repr(event_args[1])[:80]})

    sys.addaudithook(audit)

    import av
    from fastapi.testclient import TestClient

    from app.main import app

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_stt import error_rates

    key = os.environ["FSA_API_KEYS"].split(",")[0].split(":", 1)[-1]
    report: dict = {"checks": {}}

    def mp3_seconds(data: bytes) -> float:
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.audio[0]
            return sum(f.samples / f.sample_rate for f in container.decode(stream))

    with TestClient(app, headers={"X-API-Key": key}) as client:
        phase["name"] = "tts"
        voices = client.get("/api/v1/speech/voices").json()["voices"]
        alternate = next(v["id"] for v in voices if v["id"] != "default")
        synthesized = {}
        for voice in ("default", alternate):
            response = client.post(
                "/api/v1/speech/synthesize",
                json={"text": "Selamat pagi, absensi Anda sudah tercatat.", "voice": voice},
            )
            ok = response.status_code == 200 and response.headers["content-type"] == "audio/mpeg"
            seconds = mp3_seconds(response.content) if ok else 0.0
            synthesized[voice] = response.content
            report["checks"][f"tts_{voice}"] = {
                "status": response.status_code,
                "voice_header": response.headers.get("x-speech-voice"),
                "bytes": len(response.content),
                "mp3_seconds": round(seconds, 2),
                "ok": ok and seconds > 0 and response.headers.get("x-speech-voice") == voice,
            }

        phase["name"] = "stt"
        tts_connections = len(connections)
        for name, data, reference in (
            ("stt_recording", args.audio.read_bytes(), args.reference),
            ("stt_tts_output", synthesized["default"], None),
        ):
            started = time.perf_counter()
            response = client.post(
                "/api/v1/speech/transcribe",
                files={"audio": ("clip", data, "application/octet-stream")},
            )
            body = response.json()
            check = {
                "status": response.status_code,
                "seconds": round(time.perf_counter() - started, 2),
                "text": body.get("text"),
                "language": body.get("language"),
                "ok": response.status_code == 200 and bool(body.get("text")),
            }
            if reference and response.status_code == 200:
                expected = reference.read_text().strip()
                check["expected"] = expected
                check["wer"], check["cer"] = error_rates(expected, body["text"])
            report["checks"][name] = check

    stt_connections = [c for c in connections if c["phase"] == "stt"]
    report["network"] = {
        "tts_connections": tts_connections,
        "stt_connections": stt_connections,
    }
    report["checks"]["tts_reached_provider"] = {"ok": tts_connections > 0}
    report["checks"]["stt_made_no_connection"] = {"ok": not stt_connections}
    loaded = [m for m in ("cv2", "insightface") if m in sys.modules]
    report["checks"]["no_face_stack_imported"] = {"ok": not loaded, "loaded": loaded}

    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    failed = [name for name, check in report["checks"].items() if not check["ok"]]
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
