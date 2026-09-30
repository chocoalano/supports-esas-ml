"""`/speech/voices`: the approved aliases, from configuration, and nothing else."""

from __future__ import annotations

from app.core.config import VoiceAlias
from tests.conftest import make_speech_client

VOICES = "/api/v1/speech/voices"


def test_the_configured_aliases_are_listed(speech_client):
    response = speech_client.get(VOICES)

    assert response.status_code == 200
    assert response.json() == {
        "default_voice": "default",
        "voices": [
            {
                "id": "default",
                "label": "Bahasa Indonesia - Default",
                "language": "id-ID",
                "gender": "male",
            },
            {
                "id": "id_male",
                "label": "Bahasa Indonesia - Pria",
                "language": "id-ID",
                "gender": "male",
            },
            {
                "id": "id_female",
                "label": "Bahasa Indonesia - Wanita",
                "language": "id-ID",
                "gender": "female",
            },
        ],
    }


def test_provider_identifiers_are_never_listed(speech_client):
    body = speech_client.get(VOICES).text

    assert "Neural" not in body
    assert "provider" not in body


def test_the_list_is_whatever_was_configured(speech_settings, stt_engine, tts_engine):
    """Changing providers is a change to this list, invisible to callers."""
    configured = speech_settings.model_copy(
        update={
            "tts_default_voice": "narrator",
            "tts_voice_aliases": [
                VoiceAlias(
                    id="narrator",
                    provider_voice="some-other-provider-voice-42",
                    label="Narator",
                    language="id-ID",
                    gender="neutral",
                )
            ],
        }
    )

    with make_speech_client(configured, stt_engine, tts_engine) as client:
        body = client.get(VOICES).json()

    assert body["default_voice"] == "narrator"
    assert [voice["id"] for voice in body["voices"]] == ["narrator"]
    assert "some-other-provider" not in str(body)


def test_voices_are_refused_when_tts_is_off(speech_settings, stt_engine, tts_engine):
    off = speech_settings.model_copy(update={"tts_enabled": False})

    with make_speech_client(off, stt_engine, tts_engine) as client:
        response = client.get(VOICES)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "tts_disabled"
