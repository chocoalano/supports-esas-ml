"""`/speech/capabilities`: settings, reflected - no model, no network.

That it touches no engine and no socket is proved in test_speech_isolation.py;
this file is about what it says.
"""

from __future__ import annotations

from app.services.speech.audio import SUPPORTED_CONTAINERS
from tests.conftest import make_speech_client

CAPABILITIES = "/api/v1/speech/capabilities"


def test_it_reports_both_features(speech_client):
    body = speech_client.get(CAPABILITIES).json()

    assert body["stt"]["enabled"] is True
    assert body["stt"]["provider"] == "faster_whisper"
    assert body["stt"]["model"] == "small"
    assert body["stt"]["supported_containers"] == list(SUPPORTED_CONTAINERS)
    assert body["stt"]["max_audio_seconds"] == 300.0
    assert body["tts"]["enabled"] is True
    assert body["tts"]["provider"] == "edge"
    assert body["tts"]["media_type"] == "audio/mpeg"
    assert body["tts"]["voices"] == ["default", "id_male", "id_female"]
    assert body["tts"]["rate"] == {"min": -100, "max": 100}
    assert body["tts"]["volume"] == {"min": -100, "max": 100}


def test_it_mirrors_settings_rather_than_constants(speech_settings, stt_engine, tts_engine):
    configured = speech_settings.model_copy(
        update={
            "stt_enabled": False,
            "stt_model": "base",
            "stt_max_audio_seconds": 90.0,
            "stt_max_audio_bytes": 1234,
            "stt_default_language": "id",
            "stt_allowed_languages": ["id", "en"],
            "tts_max_text_chars": 500,
            "tts_timeout_seconds": 12.5,
        }
    )

    with make_speech_client(configured, stt_engine, tts_engine) as client:
        body = client.get(CAPABILITIES).json()

    assert body["stt"] == {
        "enabled": False,
        "provider": "faster_whisper",
        "model": "base",
        "supported_containers": list(SUPPORTED_CONTAINERS),
        "allowed_content_types": configured.stt_allowed_audio_types,
        "max_audio_bytes": 1234,
        "max_audio_seconds": 90.0,
        "default_language": "id",
        "allowed_languages": ["id", "en"],
    }
    assert body["tts"]["max_text_chars"] == 500
    assert body["tts"]["timeout_seconds"] == 12.5


def test_it_answers_with_both_features_off(speech_settings, stt_engine, tts_engine):
    """So an operator can ask a runtime what it thinks it is doing."""
    off = speech_settings.model_copy(update={"stt_enabled": False, "tts_enabled": False})

    with make_speech_client(off, stt_engine, tts_engine) as client:
        response = client.get(CAPABILITIES)

    assert response.status_code == 200
    assert response.json()["stt"]["enabled"] is False
    assert response.json()["tts"]["enabled"] is False


def test_the_provider_field_is_documented_as_not_for_branching(speech_client):
    schema = speech_client.app.openapi()["components"]["schemas"]

    for name in ("SttCapabilities", "TtsCapabilities"):
        assert "must not branch" in schema[name]["properties"]["provider"]["description"]
