"""Smoke-test a deployed pair of runtimes: fsa-face and fsa-speech behind nginx.

Run on the server after systemd and nginx are up (step 18). HTTP checks go
through the public URL, the way Laravel calls it, and straight to the two
upstreams to prove each runtime serves only its own routes. With --systemd
(as root) it also stops, kills and restarts fsa-speech while Face is being
called, reads the unit's limits, and watches fsa-speech's system calls during
a transcription to show it opens no network connection.

    sudo /opt/face-api/venv/bin/python scripts/smoke_deploy.py \\
        --url https://face-api.example.com --face-key "$FACE_KEY" --speech-key "$SPEECH_KEY" \\
        --audio clip.wav --systemd --output smoke.json

Exit status 0 only when every check passed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class Checks:
    def __init__(self) -> None:
        self.results: dict[str, dict] = {}

    def record(self, name: str, ok: bool, **detail) -> None:
        self.results[name] = {"ok": bool(ok), **detail}
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail if not ok else ''}", flush=True)

    def info(self, name: str, value: object) -> None:
        """Recorded, never judged."""
        self.results[name] = {"ok": True, "info": value}
        print(f"INFO  {name}  {value}", flush=True)

    @property
    def failed(self) -> list[str]:
        return [name for name, result in self.results.items() if not result["ok"]]


# -- audio fixtures ----------------------------------------------------------------------


def encode(pcm, rate: int, container: str, codec: str, *, live: bool = False) -> bytes:  # noqa: ANN001
    from fractions import Fraction

    import av

    buffer = io.BytesIO()
    options = {"live": "1"} if live else {}
    with av.open(buffer, "w", format=container, container_options=options) as out:
        stream = out.add_stream(codec, rate=48000 if codec == "libopus" else rate)
        stream.layout = "mono"
        resampler = av.AudioResampler(format=stream.codec_context.format.name, layout="mono",
                                      rate=stream.codec_context.sample_rate)  # fmt: skip
        step = 960 * 4
        for start in range(0, len(pcm), step):
            frame = av.AudioFrame.from_ndarray(
                pcm[start : start + step].reshape(1, -1), format="s16", layout="mono"
            )
            frame.sample_rate, frame.pts, frame.time_base = rate, start, Fraction(1, rate)
            for resampled in resampler.resample(frame):
                for packet in stream.encode(resampled):
                    out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buffer.getvalue()


def read_pcm(path: Path):  # noqa: ANN201
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            chunks += [r.to_ndarray().reshape(-1) for r in resampler.resample(frame)]
    return np.concatenate(chunks).astype(np.int16)


def tone(seconds: float, rate: int = 16000):  # noqa: ANN201
    import numpy as np

    t = np.arange(int(seconds * rate)) / rate
    return (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)


def mp3_seconds(data: bytes) -> float:
    import av

    with av.open(io.BytesIO(data)) as container:
        return sum(f.samples / f.sample_rate for f in container.decode(audio=0))


# -- the run -------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", required=True, help="Public base URL, through nginx.")
    parser.add_argument("--face-upstream", default="http://127.0.0.1:8001")
    parser.add_argument("--speech-upstream", default="http://127.0.0.1:8002")
    parser.add_argument("--face-key", required=True)
    parser.add_argument("--speech-key", required=True)
    parser.add_argument("--audio", type=Path, required=True, help="An Indonesian recording.")
    parser.add_argument("--insecure", action="store_true", help="Accept a self-signed cert.")
    parser.add_argument("--systemd", action="store_true", help="Also stop/kill/restart (root).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import httpx
    from smoke_face import still_fixtures

    checks = Checks()
    public = httpx.Client(base_url=args.url, verify=not args.insecure, timeout=180)
    face_up = httpx.Client(base_url=args.face_upstream, timeout=180)
    speech_up = httpx.Client(base_url=args.speech_upstream, timeout=180)
    face_headers = {"X-API-Key": args.face_key}
    speech_headers = {"X-API-Key": args.speech_key}
    refs, capture = still_fixtures()

    def verify_face() -> tuple[int, float, str | None]:
        started = time.perf_counter()
        response = public.post(
            "/api/v1/face/verify-image",
            headers=face_headers,
            files=[*refs, ("image", ("capture.png", capture, "image/png"))],
        )
        elapsed = time.perf_counter() - started
        decision = response.json().get("decision") if response.status_code == 200 else None
        return response.status_code, elapsed, decision

    def transcribe(data: bytes, content_type: str | None, name: str = "clip"):  # noqa: ANN202
        if content_type is None:  # a part that declares no type at all
            boundary = "fsa-smoke"
            body = (
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; '
                    f'filename="{name}"\r\n\r\n'
                ).encode()
                + data
                + f"\r\n--{boundary}--\r\n".encode()
            )
            return public.post(
                "/api/v1/speech/transcribe",
                content=body,
                headers={
                    **speech_headers,
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
        return public.post(
            "/api/v1/speech/transcribe",
            headers=speech_headers,
            files={"audio": (name, data, content_type)},
        )

    def code(response) -> str | None:  # noqa: ANN001
        try:
            return response.json()["error"]["code"]
        except Exception:
            return None

    # -- routing and isolation ----------------------------------------------------------
    status, _, decision = verify_face()
    checks.record("face_through_nginx", status == 200 and decision == "match", status=status)
    response = public.get("/api/v1/speech/capabilities", headers=speech_headers)
    checks.record("speech_through_nginx", response.status_code == 200, status=response.status_code)
    capabilities = response.json() if response.status_code == 200 else {}

    for path in ("/api/v1/speech/capabilities", "/api/v1/speech/voices"):
        status = face_up.get(path, headers=face_headers).status_code
        checks.record(f"face_runtime_has_no{path.replace('/', '_')}", status == 404, status=status)
    for path in ("/api/v1/face/verify", "/api/v1/face/verify-image", "/api/v1/liveness/challenge"):
        status = speech_up.post(path, headers=speech_headers).status_code
        checks.record(
            f"speech_runtime_has_no{path.replace('/', '_')}", status == 404, status=status
        )
    for path in ("/", "/demo", "/docs", "/openapi.json"):
        status = public.get(path, follow_redirects=False).status_code
        checks.record(f"nginx_blocks_{path.strip('/') or 'root'}", status == 404, status=status)
    health = public.get("/api/v1/health")
    checks.record(
        "health_contract_unchanged",
        health.status_code == 200
        and set(health.json()) == {"status", "version", "model", "model_loaded"},
        status=health.status_code,
    )

    # -- STT ---------------------------------------------------------------------------
    pcm = read_pcm(args.audio)
    wav = encode(pcm, 16000, "wav", "pcm_s16le")
    webm = encode(pcm, 16000, "webm", "libopus", live=True)
    cap = float(capabilities.get("stt", {}).get("max_audio_seconds", 300))

    for name, data, content_type in (
        ("stt_webm_indonesian", webm, "audio/webm"),
        ("stt_wav", wav, "audio/wav"),
        ("stt_missing_content_type", wav, None),
        ("stt_video_webm_declared_opus_audio", webm, "video/webm"),
    ):
        response = transcribe(data, content_type)
        text = response.json().get("text") if response.status_code == 200 else None
        checks.record(
            name, response.status_code == 200 and bool(text), status=response.status_code, text=text
        )

    response = transcribe(os.urandom(64 * 1024), "application/octet-stream")
    checks.record("stt_invalid_binary_422", response.status_code == 422, code=code(response))
    response = transcribe(os.urandom(26 * 1024 * 1024), "audio/wav")
    checks.record(
        "stt_oversize_413_from_the_application",
        response.status_code == 413 and code(response) == "payload_too_large",
        status=response.status_code,
        code=code(response),
    )
    response = transcribe(encode(tone(cap + 5), 16000, "wav", "pcm_s16le"), "audio/wav")
    checks.record(
        "stt_over_duration_422",
        response.status_code == 422 and code(response) == "audio_too_long",
        status=response.status_code,
        code=code(response),
    )

    # -- TTS ---------------------------------------------------------------------------
    voices = public.get("/api/v1/speech/voices", headers=speech_headers).json()["voices"]
    alternate = next(v["id"] for v in voices if v["id"] != "default")
    for voice in ("default", alternate):
        response = public.post(
            "/api/v1/speech/synthesize",
            headers=speech_headers,
            json={"text": "Selamat pagi, absensi Anda sudah tercatat.", "voice": voice},
        )
        ok = response.status_code == 200 and response.headers.get("content-type") == "audio/mpeg"
        seconds = mp3_seconds(response.content) if ok else 0
        checks.record(
            f"tts_{voice}_valid_mp3",
            ok and seconds > 0 and response.headers.get("x-speech-voice") == voice,
            status=response.status_code,
            seconds=round(seconds, 2),
        )
    for name, body, expected in (
        ("tts_invalid_alias", {"text": "halo", "voice": "id-ID-ArdiNeural"}, "voice_not_available"),
        ("tts_invalid_rate", {"text": "halo", "rate": "+150%"}, "invalid_request"),
        ("tts_invalid_volume", {"text": "halo", "volume": "loud"}, "invalid_request"),
    ):
        response = public.post("/api/v1/speech/synthesize", headers=speech_headers, json=body)
        checks.record(
            name, response.status_code == 422 and code(response) == expected, code=code(response)
        )

    # -- rate budgets --------------------------------------------------------------------
    statuses = [
        public.get("/api/v1/speech/capabilities", headers=speech_headers).status_code
        for _ in range(40)
    ]
    status, _, _ = verify_face()
    checks.record(
        "speech_burst_does_not_spend_face_budget",
        429 in statuses and status == 200,
        speech_429s=statuses.count(429),
        face_status=status,
        same_key=args.face_key == args.speech_key,
    )
    time.sleep(12)  # let the speech zone drain before anything else uses it

    # -- systemd ----------------------------------------------------------------------------
    if args.systemd:
        systemd_checks(checks, verify_face, transcribe, wav)

    args.output.write_text(json.dumps(checks.results, indent=2, ensure_ascii=False))
    if checks.failed:
        raise SystemExit(f"failed: {checks.failed}")
    print(f"all {len(checks.results)} checks passed")


def systemd_checks(checks: Checks, verify_face, transcribe, wav: bytes) -> None:  # noqa: ANN001
    def show(unit: str, prop: str) -> str:
        return subprocess.run(
            ["systemctl", "show", unit, "-p", prop, "--value"], capture_output=True, text=True
        ).stdout.strip()

    def wait_active(unit: str, seconds: float = 90) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if show(unit, "ActiveState") == "active":
                return True
            time.sleep(1)
        return False

    # Resource limits are a deployment's choice, not something to pass or
    # fail: reported so the result says what was running.
    for prop in ("MemoryMax", "CPUWeight", "OOMScoreAdjust"):
        checks.info(f"speech_{prop}", show("fsa-speech", prop))
    environment = show("fsa-speech", "Environment")
    checks.record("speech_offline_mode", "HF_HUB_OFFLINE=1" in environment)
    checks.record("speech_role_pinned_by_unit", "FSA_RUNTIME_ROLE=speech" in environment)

    # Face latency before, Speech stopped, Speech restarted, Face latency after.
    before = [verify_face()[1] for _ in range(5)]
    subprocess.run(["systemctl", "stop", "fsa-speech"], check=True)
    status, _, decision = verify_face()
    checks.record(
        "face_serves_with_speech_stopped", status == 200 and decision == "match", status=status
    )
    subprocess.run(["systemctl", "start", "fsa-speech"], check=True)
    checks.record("speech_restarts", wait_active("fsa-speech"))
    time.sleep(3)
    after = [verify_face()[1] for _ in range(5)]
    ratio = statistics.median(after) / statistics.median(before)
    checks.record(
        "face_latency_unchanged_after_speech_restart",
        ratio < 1.25,
        before_ms=round(1000 * statistics.median(before)),
        after_ms=round(1000 * statistics.median(after)),
    )

    # A crash is restarted by systemd (Restart=on-failure).
    subprocess.run(["systemctl", "kill", "-s", "KILL", "fsa-speech"], check=True)
    time.sleep(2)
    checks.record("speech_restarted_after_kill", wait_active("fsa-speech"))
    time.sleep(3)

    # Offline, observed: every connect() fsa-speech makes while transcribing.
    pid = show("fsa-speech", "MainPID")
    log = Path(tempfile.mkdtemp()) / "strace.log"
    tracer = subprocess.Popen(
        ["strace", "-f", "-e", "trace=connect", "-o", str(log), "-p", pid],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    response = transcribe(wav, "audio/wav")
    time.sleep(1)
    tracer.terminate()
    tracer.wait()
    calls = log.read_text().splitlines() if log.exists() else []
    outward = [
        line for line in calls
        if re.search(r"AF_INET6?", line) and not re.search(r"127\.0\.0\.1|::1|0\.0\.0\.0", line)
    ]  # fmt: skip
    checks.record(
        "stt_offline_no_outbound_connection",
        response.status_code == 200 and not outward,
        status=response.status_code,
        traced_connects=len(calls),
        outbound=outward[:5],
    )


if __name__ == "__main__":
    main()
