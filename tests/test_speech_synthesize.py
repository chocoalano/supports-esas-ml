"""`/speech/synthesize`: a strict request, and a response that is MP3 or an error.

The provider is remote, so its failures are ours to contain: whatever it does -
fail early, fail half-way, dawdle, send too much, send nothing, send something
that is not MP3 - the caller gets a JSON error, never a 200 with a broken body.
"""

from __future__ import annotations

import tempfile

import anyio
import pytest

from app.core.errors import SpeechEngineUnavailableError
from app.services.speech.tts import TextToSpeechService
from tests.conftest import MP3_FRAME, FakeTtsEngine, make_speech_client

SYNTHESIZE = "/api/v1/speech/synthesize"
PROVIDER_VOICES = ("id-ID-ArdiNeural", "id-ID-GadisNeural")


@pytest.fixture
def configured(speech_settings, stt_engine):
    def build(tts: FakeTtsEngine, **overrides):
        settings = speech_settings.model_copy(update=overrides)
        return make_speech_client(settings, stt_engine, tts)

    return build


def error_of(response) -> dict:
    return response.json()["error"]


# -- success ---------------------------------------------------------------------------


def test_the_response_is_the_mp3_and_nothing_else(speech_client, tts_engine):
    response = speech_client.post(SYNTHESIZE, json={"text": "Halo, selamat datang."})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"
    assert int(response.headers["content-length"]) == len(response.content)
    assert response.headers["content-disposition"] == 'inline; filename="speech.mp3"'
    assert response.content == MP3_FRAME * 2


def test_the_default_voice_is_used_and_named_by_its_alias(speech_client, tts_engine):
    response = speech_client.post(SYNTHESIZE, json={"text": "halo"})

    assert response.headers["x-speech-voice"] == "default"
    (request,) = tts_engine.requests
    assert request.voice == "id-ID-ArdiNeural"
    assert (request.rate_percent, request.volume_percent) == (0, 0)


def test_an_alias_is_resolved_to_the_provider_voice(speech_client, tts_engine):
    response = speech_client.post(SYNTHESIZE, json={"text": "halo", "voice": "id_female"})

    assert response.headers["x-speech-voice"] == "id_female"
    assert tts_engine.requests[0].voice == "id-ID-GadisNeural"


@pytest.mark.parametrize("voice", [None, "", "   "])
def test_a_missing_or_blank_voice_means_the_default(speech_client, tts_engine, voice):
    response = speech_client.post(SYNTHESIZE, json={"text": "halo", "voice": voice})

    assert response.status_code == 200
    assert response.headers["x-speech-voice"] == "default"


def test_the_text_is_trimmed_before_it_is_spoken(speech_client, tts_engine):
    speech_client.post(SYNTHESIZE, json={"text": "\n  halo  \t"})

    assert tts_engine.requests[0].text == "halo"


def test_nothing_is_written_to_disk(speech_client, tmp_path, monkeypatch):
    directory = tmp_path / "tmp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))

    assert speech_client.post(SYNTHESIZE, json={"text": "halo"}).status_code == 200
    assert list(directory.iterdir()) == []


# -- voices: aliases only --------------------------------------------------------------


@pytest.mark.parametrize("voice", [*PROVIDER_VOICES, "nobody", "DEFAULT"])
def test_anything_but_a_configured_alias_is_refused(speech_client, tts_engine, voice):
    """Including the provider's own identifiers: they are not part of the contract."""
    response = speech_client.post(SYNTHESIZE, json={"text": "halo", "voice": voice})

    assert response.status_code == 422
    assert error_of(response)["code"] == "voice_not_available"
    assert error_of(response)["details"]["available"] == ["default", "id_male", "id_female"]
    assert tts_engine.requests == []


@pytest.mark.parametrize("voice", [None, "id_female", "nobody"])
def test_no_provider_voice_ever_appears_in_a_response(speech_client, voice):
    """Not in a header, not in an error listing what is available."""
    response = speech_client.post(SYNTHESIZE, json={"text": "halo", "voice": voice})

    for provider_voice in PROVIDER_VOICES:
        assert provider_voice not in str(response.headers)
        if response.status_code != 200:
            assert provider_voice not in response.text


# -- a strict request -----------------------------------------------------------------


@pytest.mark.parametrize(
    "extra", [{"format": "wav"}, {"voices": "id_female"}, {"speed": "+10%"}, {"ssml": "<x/>"}]
)
def test_an_unknown_field_is_refused_not_ignored(speech_client, tts_engine, extra):
    response = speech_client.post(SYNTHESIZE, json={"text": "halo", **extra})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"
    assert tts_engine.requests == []


def test_text_is_required(speech_client):
    assert speech_client.post(SYNTHESIZE, json={"voice": "default"}).status_code == 422


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_is_refused(speech_client, tts_engine, text):
    response = speech_client.post(SYNTHESIZE, json={"text": text})

    assert response.status_code == 422
    assert error_of(response)["code"] == "invalid_request"
    assert error_of(response)["details"]["field"] == "text"


def test_text_over_the_limit_is_refused(configured, tts_engine):
    engine = FakeTtsEngine()

    with configured(engine, tts_max_text_chars=10) as client:
        response = client.post(SYNTHESIZE, json={"text": "x" * 11})
        within = client.post(SYNTHESIZE, json={"text": "x" * 10})

    assert response.status_code == 413
    assert error_of(response)["code"] == "payload_too_large"
    assert within.status_code == 200


# -- rate and volume: syntax AND range --------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("+0%", 0), ("-0%", 0), ("+100%", 100), ("-100%", -100), ("+5%", 5), ("-25%", -25)],
)
@pytest.mark.parametrize("field", ["rate", "volume"])
def test_values_inside_the_contract_reach_the_provider_as_integers(
    speech_client, tts_engine, field, value, expected
):
    response = speech_client.post(SYNTHESIZE, json={"text": "halo", field: value})

    assert response.status_code == 200, response.text
    assert getattr(tts_engine.requests[0], f"{field}_percent") == expected


@pytest.mark.parametrize(
    "value",
    [
        "+101%",  # syntax fine, range not
        "-101%",
        "+999%",
        "-999%",
        "+1000%",
        "10%",  # no sign
        "0%",
        "+10",  # no percent
        "+1e2%",
        "+10%%",
        "drop table",
        '+10%" pitch="+50Hz',  # an attribute smuggled into SSML
        "<prosody rate='x'>",
        "+10% slower",
    ],
)
@pytest.mark.parametrize("field", ["rate", "volume"])
def test_values_outside_the_contract_are_refused(speech_client, tts_engine, field, value):
    response = speech_client.post(SYNTHESIZE, json={"text": "halo", field: value})

    assert response.status_code == 422, value
    assert error_of(response)["code"] == "invalid_request"
    assert error_of(response)["details"] == {"field": field, "min": -100, "max": 100}
    assert tts_engine.requests == []


# -- the provider misbehaving ------------------------------------------------------------


def test_output_over_the_ceiling_is_an_error_not_a_truncated_mp3(configured):
    engine = FakeTtsEngine([MP3_FRAME] * 10)

    with configured(engine, tts_max_output_bytes=len(MP3_FRAME) * 3) as client:
        response = client.post(SYNTHESIZE, json={"text": "halo"})

    assert response.status_code == 502
    assert response.headers["content-type"] == "application/json"
    assert error_of(response)["code"] == "tts_output_too_large"
    # The provider's stream was closed, not left for the garbage collector.
    assert engine.closed == 1


def test_a_provider_failing_before_any_audio_is_a_502(configured):
    engine = FakeTtsEngine(error=ConnectionError("upstream said no: secret-detail"), fail_at=0)

    with configured(engine) as client:
        response = client.post(SYNTHESIZE, json={"text": "halo"})

    assert response.status_code == 502
    assert error_of(response)["code"] == "speech_provider_failed"
    # The provider's own words stay in our log, not in the caller's hands.
    assert "secret-detail" not in response.text
    assert "ConnectionError" not in response.text


def test_a_provider_failing_half_way_is_a_502_not_a_partial_200(configured):
    engine = FakeTtsEngine([MP3_FRAME] * 5, error=ConnectionError("dropped"), fail_at=3)

    with configured(engine) as client:
        response = client.post(SYNTHESIZE, json={"text": "halo"})

    assert response.status_code == 502
    assert error_of(response)["code"] == "speech_provider_failed"


def test_a_provider_that_takes_too_long_is_a_504(configured):
    engine = FakeTtsEngine(delay=5.0)

    with configured(engine, tts_timeout_seconds=0.05) as client:
        response = client.post(SYNTHESIZE, json={"text": "halo"})

    assert response.status_code == 504
    assert error_of(response)["code"] == "speech_provider_timeout"
    assert engine.closed == 1


def test_a_provider_that_times_out_by_itself_is_the_same_504(configured):
    engine = FakeTtsEngine(error=TimeoutError("socket read timed out"), fail_at=1)

    with configured(engine) as client:
        response = client.post(SYNTHESIZE, json={"text": "halo"})

    assert response.status_code == 504
    assert error_of(response)["code"] == "speech_provider_timeout"


def test_a_provider_that_sends_nothing_is_a_502_not_an_empty_200(configured):
    with configured(FakeTtsEngine([])) as client:
        response = client.post(SYNTHESIZE, json={"text": "halo"})

    assert response.status_code == 502
    assert error_of(response)["code"] == "speech_provider_failed"


def test_a_provider_that_sends_something_other_than_mp3_is_a_502(configured):
    with configured(FakeTtsEngine([b"RIFF....WAVEfmt "])) as client:
        response = client.post(SYNTHESIZE, json={"text": "halo"})

    assert response.status_code == 502
    assert error_of(response)["code"] == "speech_provider_failed"


def test_a_provider_library_that_is_missing_is_a_503(configured):
    engine = FakeTtsEngine(error=SpeechEngineUnavailableError("edge-tts is not installed"))

    with configured(engine) as client:
        response = client.post(SYNTHESIZE, json={"text": "halo"})

    assert response.status_code == 503
    assert error_of(response)["code"] == "speech_engine_unavailable"


def test_a_disabled_tts_says_so(configured):
    engine = FakeTtsEngine()

    with configured(engine, tts_enabled=False) as client:
        response = client.post(SYNTHESIZE, json={"text": "halo"})

    assert response.status_code == 503
    assert error_of(response)["code"] == "tts_disabled"
    assert engine.requests == []


@pytest.mark.parametrize(
    ("engine", "limits", "refusal"),
    [
        (FakeTtsEngine([MP3_FRAME] * 10), {"tts_max_output_bytes": len(MP3_FRAME)}, 502),
        (FakeTtsEngine(delay=5.0), {"tts_timeout_seconds": 0.05}, 504),
    ],
    ids=["too-large", "timeout"],
)
def test_the_provider_stream_is_closed_before_the_error_is_returned(
    speech_settings, engine, limits, refusal
):
    """Closed by us, on the way out - not later by the event loop's finaliser.

    Checked with no `await` between the refusal and the assertion: a stream left
    to the garbage collector is only closed on a later turn of the loop, which
    for a real provider is a socket held open past the response.
    """
    service = TextToSpeechService(
        engine=engine,
        settings=speech_settings.model_copy(update=limits),
        limiter=anyio.Semaphore(1),
    )
    closed_on_the_way_out: list[int] = []

    async def main() -> None:
        try:
            await service.synthesize(text="halo")
        except Exception as exc:
            closed_on_the_way_out.append(engine.closed)
            assert exc.status_code == refusal

    anyio.run(main)

    assert closed_on_the_way_out == [1]


# -- concurrency -----------------------------------------------------------------------------


def test_syntheses_never_exceed_their_semaphore(speech_settings):
    engine = FakeTtsEngine(delay=0.02)
    service = TextToSpeechService(
        engine=engine, settings=speech_settings, limiter=anyio.Semaphore(2)
    )

    async def main() -> None:
        async with anyio.create_task_group() as group:
            for _ in range(6):
                group.start_soon(lambda: service.synthesize(text="halo"))

    anyio.run(main)

    assert len(engine.requests) == 6
    assert engine.peak == 2
