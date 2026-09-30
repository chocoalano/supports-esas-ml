"""Speech must not be able to cost Face anything.

In production that is guaranteed by running them as separate processes. These
tests hold the parts that code owns: which routers a runtime role mounts, that
nothing heavy is imported or constructed before it is used, that the two
workloads never share a semaphore - and that PyAV and OpenCV, which each ship
their own FFmpeg, still work side by side in the one place they do meet:
development and this suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.api.deps as deps
from app.api.deps import (
    get_challenge_service,
    get_face_engine,
    get_frame_sampler,
    get_inference_limiter,
    get_stt_limiter,
    get_tts_limiter,
)
from app.core.config import Settings, get_settings
from app.core.errors import LanguageNotSupportedError
from app.main import create_app
from app.services.speech.providers.faster_whisper import FasterWhisperEngine
from app.services.speech.stt import TranscribeOptions
from tests import audio_samples
from tests.conftest import API_KEY, MP3_FRAME, build_files
from tests.speech_stubs import block_imports, install_edge_tts, install_faster_whisper

ROOT = Path(__file__).resolve().parent.parent

FACE_ROUTES = [
    ("POST", "/api/v1/face/verify"),
    ("POST", "/api/v1/face/verify-image"),
    ("POST", "/api/v1/liveness/challenge"),
    ("POST", "/api/v1/liveness/verify"),
]
SPEECH_ROUTES = [
    ("POST", "/api/v1/speech/transcribe"),
    ("POST", "/api/v1/speech/synthesize"),
    ("GET", "/api/v1/speech/voices"),
    ("GET", "/api/v1/speech/capabilities"),
]


class Counter:
    """Stands in for a class or factory, and counts how often it is called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("constructed where it must not be")


def role_client(
    monkeypatch: pytest.MonkeyPatch, role: str, settings: Settings, **overrides
) -> TestClient:
    """An app booted for `role`, the way uvicorn boots it: from the environment."""
    monkeypatch.setenv("FSA_RUNTIME_ROLE", role)
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    for dependency, value in overrides.items():
        app.dependency_overrides[getattr(deps, dependency)] = lambda value=value: value
    return TestClient(app, headers={"X-API-Key": API_KEY})


def run_python(code: str, tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    """A fresh interpreter, so `sys.modules` says what one process really imported.

    Run from a directory with no `.env`, so the developer's is not read.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=tmp_path,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": str(ROOT),
            "FSA_API_KEYS": f"test:{API_KEY}",
            **env,
        },
        capture_output=True,
        text=True,
        timeout=180,
    )


# -- runtime role: what gets mounted ------------------------------------------


def test_face_role_serves_face_and_not_speech(
    monkeypatch, settings, engine, sampler, challenges
):
    build_stt, build_tts = Counter(), Counter()
    monkeypatch.setattr(deps, "build_stt_engine", build_stt)
    monkeypatch.setattr(deps, "build_tts_engine", build_tts)

    with role_client(
        monkeypatch,
        "face",
        settings.model_copy(update={"stt_enabled": True, "tts_enabled": True}),
        get_face_engine=engine,
        get_frame_sampler=sampler,
        get_challenge_service=challenges,
    ) as client:
        assert client.post("/api/v1/face/verify", files=build_files()).status_code == 200
        assert client.post("/api/v1/liveness/challenge", json={}).status_code == 200
        assert client.get("/demo").status_code == 200

        for method, path in SPEECH_ROUTES:
            assert client.request(method, path).status_code == 404, path

    # Not merely unreachable: never built, even with both flags on.
    assert build_stt.calls == 0
    assert build_tts.calls == 0


def test_speech_role_serves_speech_and_not_face(
    monkeypatch, speech_settings, stt_engine, tts_engine
):
    face_engine, frame_sampler = Counter(), Counter()
    monkeypatch.setattr(deps, "InsightFaceEngine", face_engine)
    monkeypatch.setattr(deps, "OpenCVFrameSampler", frame_sampler)
    get_face_engine.cache_clear()
    get_frame_sampler.cache_clear()
    # On by default in production. A Speech runtime must ignore it.
    monkeypatch.setenv("FSA_WARM_UP_ON_STARTUP", "true")

    try:
        with role_client(
            monkeypatch,
            "speech",
            speech_settings,
            get_stt_engine=stt_engine,
            get_tts_engine=tts_engine,
        ) as client:
            assert client.get("/api/v1/speech/capabilities").status_code == 200
            assert client.get("/api/v1/speech/voices").status_code == 200
            assert (
                client.post("/api/v1/speech/synthesize", json={"text": "halo"}).status_code
                == 200
            )
            assert (
                client.post(
                    "/api/v1/speech/transcribe",
                    files={"audio": ("clip.wav", audio_samples.wav(), "audio/wav")},
                ).status_code
                == 200
            )

            for method, path in FACE_ROUTES:
                assert client.request(method, path, files=build_files()).status_code == 404
            # The Face playground calls /face/*; it goes where they go.
            assert client.get("/demo").status_code == 404
            assert client.get("/", follow_redirects=False).status_code == 404
    finally:
        get_face_engine.cache_clear()
        get_frame_sampler.cache_clear()

    assert face_engine.calls == 0
    assert frame_sampler.calls == 0
    # Nor the challenge signer, whose boot warning belongs to Face.
    assert get_challenge_service.cache_info().currsize == 0


def test_health_keeps_its_contract_on_the_speech_role(monkeypatch, speech_settings):
    with role_client(monkeypatch, "speech", speech_settings) as client:
        response = client.get("/api/v1/health", headers={"X-API-Key": ""})

    assert response.status_code == 200
    assert set(response.json()) == {"status", "version", "model", "model_loaded"}


def test_all_role_serves_everything(
    monkeypatch, speech_settings, engine, sampler, challenges, stt_engine, tts_engine
):
    with role_client(
        monkeypatch,
        "all",
        speech_settings,
        get_face_engine=engine,
        get_frame_sampler=sampler,
        get_challenge_service=challenges,
        get_stt_engine=stt_engine,
        get_tts_engine=tts_engine,
    ) as client:
        assert client.post("/api/v1/face/verify", files=build_files()).status_code == 200
        assert client.post("/api/v1/liveness/challenge", json={}).status_code == 200
        assert client.get("/api/v1/speech/capabilities").status_code == 200
        assert (
            client.post("/api/v1/speech/synthesize", json={"text": "halo"}).status_code == 200
        )


def test_the_default_role_is_face(monkeypatch):
    monkeypatch.delenv("FSA_RUNTIME_ROLE")

    assert Settings(_env_file=None).runtime_role.value == "face"


@pytest.mark.parametrize("value", ["bogus", "FACE", "face,speech", ""])
def test_an_unknown_role_fails_the_boot(monkeypatch, value):
    monkeypatch.setenv("FSA_RUNTIME_ROLE", value)
    get_settings.cache_clear()

    with pytest.raises(ValidationError, match="runtime_role"):
        create_app()


def test_an_unknown_role_stops_the_process_from_importing(tmp_path):
    """What uvicorn sees: `app.main` itself cannot be imported."""
    result = run_python("import app.main", tmp_path, FSA_RUNTIME_ROLE="speach")

    assert result.returncode != 0
    assert "runtime_role" in result.stderr


# -- runtime role: what gets imported -------------------------------------------


def test_a_face_runtime_never_imports_the_speech_stack(tmp_path):
    result = run_python(
        """
        import json, sys
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app, headers={"X-API-Key": "suite-key"}) as client:
            statuses = [client.post(path).status_code for path in (
                "/api/v1/speech/transcribe", "/api/v1/speech/synthesize")]
            statuses.append(client.get("/api/v1/speech/capabilities").status_code)
            statuses.append(client.get("/api/v1/health").status_code)
        loaded = [m for m in ("av", "faster_whisper", "edge_tts") if m in sys.modules]
        print(json.dumps({"statuses": statuses, "loaded": loaded}))
        """,
        tmp_path,
        FSA_RUNTIME_ROLE="face",
        FSA_WARM_UP_ON_STARTUP="false",
        FSA_STT_ENABLED="true",
        FSA_TTS_ENABLED="true",
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["statuses"] == [404, 404, 404, 200]
    assert report["loaded"] == []


def test_a_speech_runtime_serving_audio_never_imports_opencv(tmp_path):
    """The reason for the role, end to end, in a real process.

    A transcription runs for real - the audio validator decodes with PyAV - and
    misrouted Face requests arrive too. OpenCV and InsightFace must still not be
    in the process afterwards, with Face warm-up switched on.
    """
    wav = tmp_path / "clip.wav"
    wav.write_bytes(audio_samples.wav())

    result = run_python(
        f"""
        import json, sys, types

        # faster-whisper stand-in: the real one would download a model.
        stub = types.ModuleType("faster_whisper")
        class WhisperModel:
            supported_languages = ["id"]
            def __init__(self, *args, **kwargs): pass
            def transcribe(self, audio, **kwargs):
                return iter([types.SimpleNamespace(start=0, end=1, text="halo")]), \\
                    types.SimpleNamespace(language="id", language_probability=0.9)
        stub.WhisperModel = WhisperModel
        stub.utils = types.ModuleType("faster_whisper.utils")
        stub.utils.download_model = lambda name, **kwargs: "/nonexistent/model"
        sys.modules["faster_whisper"] = stub
        sys.modules["faster_whisper.utils"] = stub.utils

        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app, headers={{"X-API-Key": "suite-key"}}) as client:
            with open({str(wav)!r}, "rb") as handle:
                transcribed = client.post(
                    "/api/v1/speech/transcribe", files={{"audio": ("a.wav", handle)}}
                ).status_code
            misrouted = [client.post(path).status_code for path in (
                "/api/v1/face/verify", "/api/v1/face/verify-image",
                "/api/v1/liveness/challenge", "/api/v1/liveness/verify")]
            health = client.get("/api/v1/health").status_code
        print(json.dumps({{
            "transcribed": transcribed, "misrouted": misrouted, "health": health,
            "loaded": [m for m in ("cv2", "insightface", "onnxruntime") if m in sys.modules],
            "av": "av" in sys.modules,
        }}))
        """,
        tmp_path,
        FSA_RUNTIME_ROLE="speech",
        FSA_STT_ENABLED="true",
        FSA_TTS_ENABLED="true",
        FSA_WARM_UP_ON_STARTUP="true",
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["transcribed"] == 200
    assert report["misrouted"] == [404, 404, 404, 404]
    assert report["health"] == 200
    assert report["av"] is True  # the check below is only meaningful if audio ran
    assert report["loaded"] == []


# -- semaphores ---------------------------------------------------------------


def test_every_workload_has_its_own_semaphore():
    face, stt, tts = get_inference_limiter(), get_stt_limiter(), get_tts_limiter()

    assert stt is not face
    assert tts is not face
    assert stt is not tts


def test_each_semaphore_is_sized_by_its_own_setting(monkeypatch):
    monkeypatch.setenv("FSA_MAX_CONCURRENT_INFERENCES", "3")
    monkeypatch.setenv("FSA_STT_MAX_CONCURRENT", "1")
    monkeypatch.setenv("FSA_TTS_MAX_CONCURRENT", "5")
    get_settings.cache_clear()
    get_inference_limiter.cache_clear()

    try:
        assert get_inference_limiter().value == 3
        assert get_stt_limiter().value == 1
        assert get_tts_limiter().value == 5
    finally:
        get_inference_limiter.cache_clear()


def test_speech_requests_never_touch_face_resources(
    monkeypatch, speech_settings, stt_engine, tts_engine
):
    """In the one process where both exist, Speech still never reaches Face's."""
    face_engine = Counter()
    monkeypatch.setattr(deps, "InsightFaceEngine", face_engine)
    get_face_engine.cache_clear()
    get_inference_limiter.cache_clear()

    try:
        with role_client(
            monkeypatch,
            "all",
            speech_settings,
            get_stt_engine=stt_engine,
            get_tts_engine=tts_engine,
        ) as client:
            client.post(
                "/api/v1/speech/transcribe",
                files={"audio": ("clip.wav", audio_samples.wav(), "audio/wav")},
            )
            client.post("/api/v1/speech/synthesize", json={"text": "halo"})
            client.get("/api/v1/speech/voices")
            client.get("/api/v1/speech/capabilities")

        # Never even created, let alone acquired.
        assert get_inference_limiter.cache_info().currsize == 0
        assert face_engine.calls == 0
    finally:
        get_face_engine.cache_clear()
        get_inference_limiter.cache_clear()


# -- lazy loading ---------------------------------------------------------------


def test_the_app_imports_and_describes_itself_without_the_speech_libraries(tmp_path):
    result = run_python(
        """
        import sys
        sys.modules["faster_whisper"] = None   # ImportError on `import faster_whisper`
        sys.modules["edge_tts"] = None
        from app.main import app
        assert "/api/v1/speech/transcribe" in app.openapi()["paths"]
        print("ok")
        """,
        tmp_path,
        FSA_RUNTIME_ROLE="all",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_a_missing_library_is_a_503_not_a_crash(monkeypatch, speech_settings):
    block_imports(monkeypatch, "faster_whisper", "edge_tts")

    with role_client(monkeypatch, "all", speech_settings) as client:
        transcribed = client.post(
            "/api/v1/speech/transcribe",
            files={"audio": ("clip.wav", audio_samples.wav(), "audio/wav")},
        )
        synthesised = client.post("/api/v1/speech/synthesize", json={"text": "halo"})

    assert transcribed.status_code == 503
    assert transcribed.json()["error"]["code"] == "speech_engine_unavailable"
    assert transcribed.json()["error"]["details"] == {"reason": "engine_not_installed"}
    assert synthesised.status_code == 503
    assert synthesised.json()["error"]["code"] == "speech_engine_unavailable"
    assert synthesised.json()["error"]["details"] == {"reason": "engine_not_installed"}
    # Which library is missing is the operator's business; it is in the log.
    for response in (transcribed, synthesised):
        assert "whisper" not in response.text.lower()
        assert "edge" not in response.text.lower()


def test_nothing_loads_the_model_but_a_transcription(monkeypatch, speech_settings):
    """Not the schema, not /health, not capabilities; the first transcription, once."""
    whisper = install_faster_whisper(monkeypatch)

    with role_client(monkeypatch, "all", speech_settings) as client:
        client.app.openapi()
        client.get("/api/v1/health")
        client.get("/api/v1/speech/capabilities")
        client.get("/api/v1/speech/voices")
        assert whisper.constructed == []

        for _ in range(3):
            response = client.post(
                "/api/v1/speech/transcribe",
                files={"audio": ("clip.wav", audio_samples.wav(), "audio/wav")},
            )
            assert response.status_code == 200, response.text

    assert len(whisper.constructed) == 1
    assert len(whisper.transcribed) == 3


def test_racing_loads_build_the_model_once(monkeypatch, settings):
    whisper = install_faster_whisper(monkeypatch, load_seconds=0.05)
    engine = FasterWhisperEngine(settings)

    threads = [threading.Thread(target=engine.load) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(whisper.constructed) == 1
    assert engine.is_loaded()


def test_catalogue_endpoints_touch_no_engine_and_no_network(monkeypatch, speech_settings):
    """Both answer from settings. A provider being down cannot take them down."""
    block_imports(monkeypatch, "faster_whisper", "edge_tts")
    build_stt, build_tts = Counter(), Counter()
    monkeypatch.setattr(deps, "build_stt_engine", build_stt)
    monkeypatch.setattr(deps, "build_tts_engine", build_tts)

    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr("socket.socket.connect", no_network)
    monkeypatch.setattr("socket.create_connection", no_network)

    with role_client(monkeypatch, "all", speech_settings) as client:
        assert client.get("/api/v1/speech/capabilities").status_code == 200
        assert client.get("/api/v1/speech/voices").status_code == 200

    assert build_stt.calls == 0
    assert build_tts.calls == 0


@pytest.mark.parametrize(
    ("disabled", "path", "code"),
    [
        ("stt_enabled", "/api/v1/speech/transcribe", "stt_disabled"),
        ("tts_enabled", "/api/v1/speech/synthesize", "tts_disabled"),
    ],
)
def test_a_disabled_feature_never_builds_its_engine(
    monkeypatch, speech_settings, disabled, path, code
):
    build_stt, build_tts = Counter(), Counter()
    monkeypatch.setattr(deps, "build_stt_engine", build_stt)
    monkeypatch.setattr(deps, "build_tts_engine", build_tts)
    off = speech_settings.model_copy(update={disabled: False})

    with role_client(monkeypatch, "all", off) as client:
        if path.endswith("transcribe"):
            response = client.post(
                path, files={"audio": ("clip.wav", audio_samples.wav(), "audio/wav")}
            )
        else:
            response = client.post(path, json={"text": "halo"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == code
    assert build_stt.calls == 0
    assert build_tts.calls == 0


# -- the real providers, against stand-in libraries ------------------------------


def test_faster_whisper_is_configured_from_settings(monkeypatch, settings, tmp_path):
    whisper = install_faster_whisper(monkeypatch)
    configured = settings.model_copy(
        update={
            "stt_model": "base",
            "stt_compute_type": "int8",
            "stt_cpu_threads": 2,
            "stt_model_root": str(tmp_path),
            "stt_beam_size": 1,
            "stt_vad_filter": True,
        }
    )
    clip = tmp_path / "clip"
    clip.write_bytes(audio_samples.wav())

    transcript = FasterWhisperEngine(configured).transcribe(clip, TranscribeOptions(language=None))

    # Resolved by name into the model root, then loaded from the resolved path.
    assert whisper.resolved == [("base", False, str(tmp_path))]
    ((model, kwargs),) = whisper.constructed
    assert model == whisper.model_dir
    assert kwargs == {"device": "cpu", "compute_type": "int8", "cpu_threads": 2, "num_workers": 1}
    assert whisper.transcribed[0]["beam_size"] == 1
    assert whisper.transcribed[0]["vad_filter"] is True
    assert transcript.text == "halo dunia"
    assert transcript.language == "id"
    assert transcript.language_probability == pytest.approx(0.9123)


def test_faster_whisper_refuses_a_language_the_model_lacks(monkeypatch, settings, tmp_path):
    install_faster_whisper(monkeypatch)
    clip = tmp_path / "clip"
    clip.write_bytes(audio_samples.wav())

    with pytest.raises(LanguageNotSupportedError):
        FasterWhisperEngine(settings).transcribe(clip, TranscribeOptions(language="xx"))


def test_edge_gets_integers_formatted_by_us_and_returns_only_audio(monkeypatch, speech_settings):
    edge = install_edge_tts(
        monkeypatch,
        chunks=[
            {"type": "WordBoundary", "offset": 0, "text": "halo"},
            {"type": "audio", "data": MP3_FRAME},
            {"type": "SentenceBoundary", "offset": 1},
            {"type": "audio", "data": MP3_FRAME},
        ],
    )

    with role_client(monkeypatch, "all", speech_settings) as client:
        response = client.post(
            "/api/v1/speech/synthesize",
            json={"text": "  halo  ", "voice": "id_female", "rate": "-10%", "volume": "+0%"},
        )

    assert response.status_code == 200, response.text
    assert response.content == MP3_FRAME * 2
    (call,) = edge.communicated
    assert call == {
        "text": "halo",
        "voice": "id-ID-GadisNeural",
        "rate": "-10%",
        "volume": "+0%",
        # Half the service's 30s budget: a stalled provider is caught by the
        # library, which closes its own websocket, before the backstop cancels.
        "connect_timeout": 15,
        "receive_timeout": 15,
    }


# -- two FFmpeg builds, one process ---------------------------------------------


@pytest.mark.parametrize("first", ["av", "cv2"])
def test_opencv_and_pyav_still_work_side_by_side(tmp_path, first):
    """PyAV and OpenCV each bundle their own FFmpeg. Production keeps them apart;
    development and this suite load both. If a future pair of versions stops
    coexisting, this is where it should show first - on Linux as well as macOS,
    where the warning printed about duplicate classes proves nothing either way.
    """
    wav = tmp_path / "clip.wav"
    wav.write_bytes(audio_samples.wav(seconds=1.5))
    second = "cv2" if first == "av" else "av"

    result = run_python(
        f"""
        import json, os, tempfile
        import {first}
        import {second}
        import cv2, numpy as np
        from pathlib import Path
        from app.core.config import Settings
        from app.services.video import OpenCVFrameSampler
        from app.services.speech.audio import inspect_audio

        path = os.path.join(tempfile.mkdtemp(), "clip.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
        for index in range(60):
            writer.write(np.full((240, 320, 3), index * 4 % 256, dtype=np.uint8))
        writer.release()

        sampled = OpenCVFrameSampler(Settings(_env_file=None)).sample(Path(path))
        with av.open(path) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        audio = inspect_audio(Path({str(wav)!r}), max_seconds=10)
        print(json.dumps({{
            "total_frames": sampled.total_frames,
            "sampled": len(sampled.frames),
            "fps": sampled.fps,
            "av_frames": decoded,
            "audio_seconds": audio.duration_seconds,
        }}))
        """,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["total_frames"] == 60
    assert report["sampled"] > 0
    assert report["fps"] == pytest.approx(30.0)
    assert report["av_frames"] == 60
    assert report["audio_seconds"] == pytest.approx(1.5, abs=0.05)
