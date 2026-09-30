"""The STT model is provisioned before the Speech runtime starts - never fetched by a request.

Measured: a first transcription that has to download `small` takes ~80 seconds.
Production runs with HF_HUB_OFFLINE=1, so a model that is not on disk must make
a transcription fail at once and say why, and the boot log must say it first.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.errors import SpeechEngineUnavailableError
from app.main import create_app
from app.services.speech.providers.faster_whisper import (
    FasterWhisperEngine,
    provisioning_problem,
)
from app.services.speech.stt import TranscribeOptions
from tests import audio_samples
from tests.conftest import API_KEY
from tests.speech_stubs import block_imports, install_faster_whisper


def speech_runtime(monkeypatch, **env: str) -> TestClient:
    """A Speech runtime booted from the environment, as systemd would start it."""
    monkeypatch.setenv("FSA_RUNTIME_ROLE", "speech")
    monkeypatch.setenv("FSA_STT_ENABLED", "true")
    monkeypatch.setenv("FSA_API_KEYS", f"test:{API_KEY}")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return TestClient(create_app(), headers={"X-API-Key": API_KEY})


def transcribe(client: TestClient):
    return client.post(
        "/api/v1/speech/transcribe",
        files={"audio": ("clip.wav", audio_samples.wav(), "audio/wav")},
    )


# -- a request against a model that is not there -----------------------------------------


def test_a_missing_model_is_a_fast_clear_503(monkeypatch):
    whisper = install_faster_whisper(monkeypatch, missing=True)

    with speech_runtime(monkeypatch, FSA_STT_MODEL_ROOT="/var/lib/fsa/models") as client:
        response = transcribe(client)

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "speech_engine_unavailable"
    assert error["details"] == {"reason": "model_not_provisioned"}
    # The operator's details go to the log, not to the caller.
    assert "/var/lib/fsa" not in response.text
    assert whisper.constructed == []


def test_a_model_name_is_never_read_as_a_directory_in_the_working_directory(
    monkeypatch, settings, tmp_path
):
    """`WhisperModel("small")` loads ./small when it exists. Ours never asks it to."""
    whisper = install_faster_whisper(monkeypatch)
    (tmp_path / "small").mkdir()
    monkeypatch.chdir(tmp_path)
    clip = tmp_path / "clip"
    clip.write_bytes(audio_samples.wav())

    FasterWhisperEngine(settings).transcribe(clip, TranscribeOptions(language=None))

    assert whisper.resolved == [("small", False, None)]
    ((model, _),) = whisper.constructed
    assert model == whisper.model_dir


def test_an_absolute_path_is_a_local_model(monkeypatch, settings, tmp_path):
    whisper = install_faster_whisper(monkeypatch)
    clip = tmp_path / "clip"
    clip.write_bytes(audio_samples.wav())
    local = settings.model_copy(update={"stt_model": str(tmp_path)})

    FasterWhisperEngine(local).transcribe(clip, TranscribeOptions(language=None))

    assert whisper.resolved == []
    assert whisper.constructed[0][0] == str(tmp_path)


def test_a_failed_load_is_retried_by_the_next_request(monkeypatch, settings):
    """Provisioning the model fixes the running service; no restart needed."""
    install_faster_whisper(monkeypatch, missing=True)
    engine = FasterWhisperEngine(settings)

    with pytest.raises(SpeechEngineUnavailableError):
        engine.load()

    whisper = install_faster_whisper(monkeypatch)
    engine.load()

    assert engine.is_loaded()
    assert len(whisper.constructed) == 1


# -- the boot check -------------------------------------------------------------------------


def test_the_boot_check_looks_on_disk_only(monkeypatch, settings):
    whisper = install_faster_whisper(monkeypatch)

    assert provisioning_problem(settings) is None
    assert whisper.resolved == [("small", True, None)]
    assert whisper.constructed == []


def test_the_boot_check_names_what_is_missing(monkeypatch, settings):
    install_faster_whisper(monkeypatch, missing=True)
    configured = settings.model_copy(update={"stt_model_root": "/var/lib/fsa/models"})

    problem = provisioning_problem(configured)

    assert "'small'" in problem
    assert "/var/lib/fsa/models" in problem


def test_the_boot_check_notices_a_model_without_weights(monkeypatch, settings):
    whisper = install_faster_whisper(monkeypatch)
    (Path(whisper.model_dir) / "model.bin").unlink()

    assert "model.bin" in provisioning_problem(settings)


def test_the_boot_check_notices_a_missing_library(monkeypatch, settings):
    block_imports(monkeypatch, "faster_whisper", "faster_whisper.utils")

    assert "not installed" in provisioning_problem(settings)


def test_a_speech_runtime_says_at_boot_that_its_model_is_missing(monkeypatch, caplog):
    whisper = install_faster_whisper(monkeypatch, missing=True)

    with caplog.at_level(logging.ERROR, logger="app.main"), speech_runtime(monkeypatch):
        pass

    assert any("model cannot be loaded" in record.message for record in caplog.records)
    assert whisper.constructed == []


def test_a_speech_runtime_with_its_model_boots_quietly(monkeypatch, caplog):
    whisper = install_faster_whisper(monkeypatch)

    with caplog.at_level(logging.ERROR, logger="app.main"), speech_runtime(monkeypatch):
        pass

    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    # Checked, not loaded: the model still loads on the first transcription.
    assert whisper.constructed == []


def test_the_boot_check_is_skipped_when_stt_is_off(monkeypatch):
    whisper = install_faster_whisper(monkeypatch, missing=True)

    with speech_runtime(monkeypatch, FSA_STT_ENABLED="false"):
        pass

    assert whisper.resolved == []
