"""`/speech/transcribe`: the response contract, languages, and a strict form."""

from __future__ import annotations

import pytest

from app.core.errors import SpeechEngineUnavailableError
from app.services.speech.stt import Transcript
from tests import audio_samples
from tests.conftest import make_speech_client

TRANSCRIBE = "/api/v1/speech/transcribe"


def transcribe(client, data: bytes | None = None, **form):
    return client.post(
        TRANSCRIBE,
        files={"audio": ("clip.wav", data or audio_samples.wav(), "audio/wav")},
        data={key: str(value) for key, value in form.items()},
    )


@pytest.fixture
def configured(speech_settings, stt_engine, tts_engine):
    def build(**overrides):
        settings = speech_settings.model_copy(update=overrides)
        return make_speech_client(settings, stt_engine, tts_engine)

    return build


# -- the response ------------------------------------------------------------------


def test_a_transcription_reports_text_language_and_length(speech_client):
    response = transcribe(speech_client, audio_samples.wav(seconds=2.0))

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "text",
        "language",
        "language_probability",
        "duration_seconds",
        "segments",
        "processing_ms",
    }
    assert body["text"] == "halo dunia"
    assert body["language"] == "id"
    assert body["language_probability"] == pytest.approx(0.97)
    # Measured by our decoder, not reported by the engine.
    assert body["duration_seconds"] == pytest.approx(2.0, abs=0.05)
    assert body["processing_ms"] >= 0


def test_segments_are_left_out_unless_asked_for(speech_client):
    assert transcribe(speech_client).json()["segments"] == []


def test_segments_are_returned_when_asked_for(speech_client):
    body = transcribe(speech_client, include_segments="true").json()

    assert body["segments"] == [
        {"start": 0.0, "end": 1.2, "text": "halo"},
        {"start": 1.2, "end": 2.0, "text": "dunia"},
    ]


def test_the_model_is_loaded_on_the_first_transcription_only(speech_client, stt_engine):
    assert not stt_engine.is_loaded()

    for _ in range(3):
        transcribe(speech_client)

    assert stt_engine.loads == 1


# -- language -------------------------------------------------------------------------


def test_a_requested_language_reaches_the_engine(speech_client, stt_engine):
    transcribe(speech_client, language="ID")

    assert stt_engine.calls[0].language == "id"


def test_the_default_language_applies_when_none_is_asked_for(configured, stt_engine):
    with configured(stt_default_language="id") as client:
        transcribe(client)

    assert stt_engine.calls[0].language == "id"


def test_with_no_language_and_no_default_the_engine_detects(speech_client, stt_engine):
    transcribe(speech_client)

    assert stt_engine.calls[0].language is None


@pytest.mark.parametrize("language", ["indonesian", "i1", "id-ID", "<id>"])
def test_something_that_is_not_a_language_code_is_refused(speech_client, stt_engine, language):
    response = transcribe(speech_client, language=language)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["details"]["field"] == "language"
    assert stt_engine.calls == []


def test_a_language_outside_the_allow_list_is_refused_before_any_work(configured, stt_engine):
    with configured(stt_allowed_languages=["id", "en"]) as client:
        response = transcribe(client, language="fr")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "language_not_supported"
    assert response.json()["error"]["details"]["allowed"] == ["id", "en"]
    assert stt_engine.calls == []


def test_a_detected_language_outside_the_allow_list_is_refused(configured, stt_engine):
    """Nothing was asked for, so the allow-list is applied to what came back."""
    stt_engine.transcript = Transcript(text="selamat", language="ms", language_probability=0.6)

    with configured(stt_allowed_languages=["id", "en"]) as client:
        response = transcribe(client)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "language_not_supported"
    assert response.json()["error"]["details"]["language"] == "ms"


# -- a strict form ------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["include_segment", "lang", "format", "model"])
def test_an_unknown_field_is_refused_not_ignored(speech_client, stt_engine, field):
    response = transcribe(speech_client, **{field: "x"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"
    assert stt_engine.calls == []


def test_the_audio_part_is_required(speech_client):
    response = speech_client.post(TRANSCRIBE, data={"language": "id"})

    assert response.status_code == 422


# -- engine state ---------------------------------------------------------------------------


def test_an_engine_that_cannot_load_is_a_503(speech_client, stt_engine):
    stt_engine.error = SpeechEngineUnavailableError("no model")

    response = transcribe(speech_client)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "speech_engine_unavailable"


def test_a_disabled_stt_says_so(configured, stt_engine):
    with configured(stt_enabled=False) as client:
        response = transcribe(client)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "stt_disabled"
    assert stt_engine.calls == []
