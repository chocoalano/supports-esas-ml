"""One event per speech request, whatever happened to it - and nothing anybody said.

The events are what an operator sizes a deployment from, so they carry the
timings and the configuration that produced them. They never carry the audio,
the transcript, or the text to be spoken.
"""

from __future__ import annotations

import logging

import anyio
import pytest

from app.api.deps import get_speech_telemetry
from app.services.speech.telemetry import (
    CANCELLED,
    INTERNAL_ERROR,
    LogTelemetry,
    SttEvent,
    TtsEvent,
    logfmt,
)
from app.services.speech.tts import TextToSpeechService
from tests import audio_samples
from tests.conftest import FakeTtsEngine, make_speech_client

TRANSCRIBE = "/api/v1/speech/transcribe"
SYNTHESIZE = "/api/v1/speech/synthesize"


class Recorder(LogTelemetry):
    """The log sink, keeping what it wrote."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, SttEvent | TtsEvent]] = []

    def end(self, kind, event) -> None:  # noqa: ANN001
        super().end(kind, event)
        self.events.append((kind, event))


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def client(speech_settings, stt_engine, tts_engine, recorder):
    test_client = make_speech_client(speech_settings, stt_engine, tts_engine)
    test_client.app.dependency_overrides[get_speech_telemetry] = lambda: recorder
    with test_client:
        yield test_client


def only(recorder: Recorder, kind: str):  # noqa: ANN201
    assert [k for k, _ in recorder.events] == [kind]
    return recorder.events[0][1]


def upload(data: bytes, content_type: str = "audio/wav") -> dict:
    return {"audio": ("clip.wav", data, content_type)}


# -- speech-to-text -----------------------------------------------------------------------


def test_a_transcription_is_one_event_with_what_sizing_needs(client, recorder, speech_settings):
    response = client.post(TRANSCRIBE, files=upload(audio_samples.wav(seconds=2.0)))

    assert response.status_code == 200
    event = only(recorder, "stt")
    assert (event.outcome, event.status, event.caller) == ("ok", 200, "test")
    assert (event.provider, event.model, event.device, event.compute_type) == (
        speech_settings.stt_provider,
        speech_settings.stt_model,
        speech_settings.stt_device,
        speech_settings.stt_compute_type,
    )
    assert event.audio_seconds == pytest.approx(2.0, abs=0.05)
    assert event.container == "wav"
    assert event.upload_bytes == len(audio_samples.wav(seconds=2.0))
    assert (event.language, event.segments) == ("id", 2)
    assert event.in_flight == 1
    for timing in (event.wait_ms, event.validate_ms, event.transcribe_ms, event.total_ms):
        assert isinstance(timing, int) and timing >= 0


@pytest.mark.parametrize(
    ("files", "data", "outcome", "status"),
    [
        (upload(b"not audio at all" * 64), {}, "invalid_upload", 422),
        (upload(audio_samples.wav(), "image/png"), {}, "unsupported_media_type", 415),
        (upload(audio_samples.mp4_video_only()), {}, "no_audio_stream", 422),
        (upload(audio_samples.wav()), {"language": "english"}, "invalid_request", 422),
    ],
    ids=["not-audio", "declared-type", "no-audio-stream", "bad-language"],
)
def test_a_refused_transcription_is_one_event_with_its_code(
    client, recorder, files, data, outcome, status
):
    response = client.post(TRANSCRIBE, files=files, data=data)

    assert response.status_code == status
    event = only(recorder, "stt")
    assert (event.outcome, event.status) == (outcome, status)
    assert event.transcribe_ms is None  # the engine was never asked


def test_an_engine_that_breaks_is_an_internal_error_event(client, recorder, stt_engine):
    stt_engine.error = RuntimeError("the engine fell over")

    with pytest.raises(RuntimeError):
        client.post(TRANSCRIBE, files=upload(audio_samples.wav()))

    event = only(recorder, "stt")
    assert (event.outcome, event.status) == (INTERNAL_ERROR, 500)


def test_a_disabled_feature_never_reaches_the_service(speech_settings, stt_engine, tts_engine):
    """Refused by the feature gate, before any service exists to measure it."""
    recorder = Recorder()
    off = speech_settings.model_copy(update={"stt_enabled": False})
    test_client = make_speech_client(off, stt_engine, tts_engine)
    test_client.app.dependency_overrides[get_speech_telemetry] = lambda: recorder

    with test_client:
        assert test_client.post(TRANSCRIBE, files=upload(audio_samples.wav())).status_code == 503

    assert recorder.events == []


# -- text-to-speech ------------------------------------------------------------------------


def test_a_synthesis_is_one_event_named_by_its_alias(client, recorder):
    response = client.post(
        SYNTHESIZE, json={"text": "  Halo, selamat pagi.  ", "voice": "id_female"}
    )

    assert response.status_code == 200
    event = only(recorder, "tts")
    assert (event.outcome, event.status, event.voice) == ("ok", 200, "id_female")
    assert event.text_chars == len("Halo, selamat pagi.")
    assert event.output_bytes == len(response.content)
    assert event.provider == "edge"
    assert isinstance(event.synthesize_ms, int) and isinstance(event.wait_ms, int)


def test_a_refused_voice_leaves_no_voice_in_the_event(client, recorder):
    client.post(SYNTHESIZE, json={"text": "halo", "voice": "id-ID-ArdiNeural"})

    event = only(recorder, "tts")
    assert (event.outcome, event.voice) == ("voice_not_available", None)


def test_a_provider_timeout_is_a_504_event(speech_settings, stt_engine):
    recorder = Recorder()
    slow = speech_settings.model_copy(update={"tts_timeout_seconds": 0.05})
    test_client = make_speech_client(slow, stt_engine, FakeTtsEngine(delay=5.0))
    test_client.app.dependency_overrides[get_speech_telemetry] = lambda: recorder

    with test_client:
        test_client.post(SYNTHESIZE, json={"text": "halo"})

    event = only(recorder, "tts")
    assert (event.outcome, event.status) == ("speech_provider_timeout", 504)
    assert event.synthesize_ms is not None and event.synthesize_ms >= 50


def test_a_caller_that_hangs_up_is_recorded_as_cancelled(speech_settings):
    """The request never finished, and still it is counted, once, and let go of."""
    recorder = Recorder()
    service = TextToSpeechService(
        engine=FakeTtsEngine(delay=5.0),
        settings=speech_settings,
        limiter=anyio.Semaphore(1),
        telemetry=recorder,
    )

    async def main() -> None:
        with anyio.move_on_after(0.05):  # the connection drops
            await service.synthesize(text="halo")

    anyio.run(main)

    event = only(recorder, "tts")
    assert (event.outcome, event.status) == (CANCELLED, 499)
    assert recorder.in_flight("tts") == 0


# -- in flight, and the log line ---------------------------------------------------------------


def test_in_flight_counts_return_to_zero(client, recorder):
    client.post(TRANSCRIBE, files=upload(audio_samples.wav()))
    client.post(TRANSCRIBE, files=upload(b"junk" * 100))
    client.post(SYNTHESIZE, json={"text": "halo"})

    assert recorder.in_flight("stt") == 0
    assert recorder.in_flight("tts") == 0


def test_the_log_never_holds_what_was_said(client, caplog):
    secret_text = "Nomor rekening saya 1234567890"
    with caplog.at_level(logging.INFO, logger="app.speech"):
        client.post(TRANSCRIBE, files=upload(audio_samples.wav()))
        client.post(SYNTHESIZE, json={"text": secret_text, "voice": "id_male"})

    lines = [record.getMessage() for record in caplog.records if record.name == "app.speech"]
    assert len(lines) == 2
    assert lines[0].startswith("speech.stt outcome=ok status=200")
    assert lines[1].startswith("speech.tts outcome=ok status=200")
    text = "\n".join(lines)
    assert "halo dunia" not in text  # the fake engine's transcript
    assert "rekening" not in text and "1234567890" not in text
    assert "Neural" not in text  # provider voice ids stay inside
    assert "voice=id_male" in text


def test_server_side_failures_are_logged_louder_than_refusals(client, caplog, stt_engine):
    with caplog.at_level(logging.INFO, logger="app.speech"):
        client.post(TRANSCRIBE, files=upload(b"junk" * 100))
        stt_engine.error = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            client.post(TRANSCRIBE, files=upload(audio_samples.wav()))

    levels = [record.levelno for record in caplog.records if record.name == "app.speech"]
    assert levels == [logging.INFO, logging.ERROR]


def test_logfmt_quotes_what_would_split_a_field():
    assert logfmt({"a": "plain", "b": "two words", "c": None, "d": 1.5, "e": 'say "hi"'}) == (
        'a=plain b="two words" c=- d=1.5 e="say \\"hi\\""'
    )
