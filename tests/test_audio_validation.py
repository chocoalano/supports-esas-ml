"""What counts as audio, and how long is too long.

Every layer is tested for the case it exists to catch, including the ones the
layer before it lets through: a declared type that is absent, a signature that
is right on a file that is wrong, a container that opens and holds no sound, a
length that is not declared, or is declared wrongly.

The files are real, encoded by FFmpeg while the suite runs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from app.core.errors import AudioDecodeError, AudioTooLongError
from app.services.speech import audio as audio_module
from app.services.speech.audio import SUPPORTED_CONTAINERS, container_family, inspect_audio
from tests import audio_samples
from tests.conftest import make_speech_client

TRANSCRIBE = "/api/v1/speech/transcribe"


def post_audio(
    client,
    data: bytes,
    filename: str = "clip.bin",
    content_type: str | None = "application/octet-stream",
):
    """Upload `data`; `content_type=None` sends a part that declares no type at all.

    The default is what httpx and Guzzle send when they cannot guess one.
    """
    if content_type is not None:
        return client.post(TRANSCRIBE, files={"audio": (filename, data, content_type)})

    # httpx always writes a part Content-Type, guessing when it is not given
    # one, so a part without the header has to be written by hand.
    boundary = "fsa-test-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n\r\n'
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return client.post(
        TRANSCRIBE,
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


def error_code(response) -> str:
    return response.json()["error"]["code"]


@pytest.fixture
def capped(speech_settings, stt_engine, tts_engine):
    """A client whose limits are small enough to test with short files."""

    def build(**limits):
        return make_speech_client(speech_settings.model_copy(update=limits), stt_engine, tts_engine)

    return build


# -- every container we claim to support ---------------------------------------


@pytest.mark.parametrize(
    ("maker", "family"),
    [
        (audio_samples.wav, "wav"),
        (audio_samples.mp3, "mp3"),
        (audio_samples.m4a, "mp4"),
        (audio_samples.ogg_opus, "ogg"),
        (audio_samples.flac, "flac"),
        (audio_samples.webm_opus, "webm"),
    ],
)
def test_each_supported_container_is_accepted(speech_client, stt_engine, maker, family):
    response = post_audio(speech_client, maker())

    assert response.status_code == 200, response.text
    assert response.json()["duration_seconds"] == pytest.approx(2.0, abs=0.05)
    assert family in SUPPORTED_CONTAINERS
    assert len(stt_engine.calls) == 1


# -- layer: declared Content-Type (not an authority) -----------------------------


def test_no_declared_type_and_real_audio_is_accepted(speech_client):
    """#1 - the header is optional, so its absence decides nothing."""
    response = post_audio(speech_client, audio_samples.wav(), content_type=None)

    assert response.status_code == 200, response.text


def test_no_declared_type_and_not_audio_is_still_refused(speech_client, stt_engine):
    """#2 - skipping the header check skips nothing that matters."""
    response = post_audio(speech_client, os.urandom(4096), content_type=None)

    assert response.status_code == 422
    assert error_code(response) == "invalid_upload"
    assert stt_engine.calls == []


def test_a_declared_type_that_is_not_audio_is_refused(speech_client, stt_engine):
    """#3"""
    response = post_audio(speech_client, audio_samples.wav(), content_type="image/png")

    assert response.status_code == 415
    assert error_code(response) == "unsupported_media_type"
    assert stt_engine.calls == []


def test_a_declared_type_with_parameters_is_read_as_its_media_type(speech_client):
    """MediaRecorder declares `audio/webm;codecs=opus`."""
    response = post_audio(
        speech_client, audio_samples.recorder_webm(), content_type="audio/webm;codecs=opus"
    )

    assert response.status_code == 200, response.text


# -- layer: the file name (never consulted) --------------------------------------


def test_a_png_named_like_a_wav_is_refused(speech_client, stt_engine):
    """#4"""
    response = post_audio(
        speech_client, audio_samples.PNG_HEADER + os.urandom(512), filename="voice.wav"
    )

    assert response.status_code == 422
    assert error_code(response) == "invalid_upload"
    assert stt_engine.calls == []


def test_a_real_wav_named_like_text_is_accepted(speech_client):
    """#14 - the name decides nothing, in either direction."""
    response = post_audio(speech_client, audio_samples.wav(), filename="notes.txt")

    assert response.status_code == 200, response.text


def test_the_uploaded_name_never_reaches_the_decoder(speech_client, stt_engine, monkeypatch):
    """FFmpeg's probe weighs a file's extension, so the one on disk is not the caller's."""
    seen: list[Path] = []
    real = audio_module.inspect_audio

    def spy(path: Path, **kwargs):
        seen.append(path)
        return real(path, **kwargs)

    monkeypatch.setattr("app.services.speech.stt.inspect_audio", spy)
    post_audio(speech_client, audio_samples.wav(), filename="payload.m3u8")

    assert seen and seen[0].suffix == ""


# -- layer: signature (early rejection, not an authority) -------------------------


def test_an_id3_header_in_front_of_text_is_refused_by_the_decoder(speech_client, stt_engine):
    """#5 - the signature layer lets it through; the decoder does not."""
    payload = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"not audio at all " * 100

    assert container_family(payload) == "mp3"
    response = post_audio(speech_client, payload, filename="song.mp3")

    assert response.status_code == 422
    assert error_code(response) == "audio_decode_failed"
    assert stt_engine.calls == []


def test_a_malformed_webm_is_refused_by_the_decoder(speech_client, stt_engine):
    """#7 - the right magic bytes on garbage."""
    payload = b"\x1a\x45\xdf\xa3" + os.urandom(4096)

    response = post_audio(speech_client, payload, filename="clip.webm")

    assert response.status_code == 422
    assert error_code(response) == "audio_decode_failed"
    assert stt_engine.calls == []


# -- layer: the decoder, and what it opened --------------------------------------


def test_a_container_without_sound_is_refused(speech_client, stt_engine):
    """#6 - a valid MP4 with a picture and no audio."""
    response = post_audio(speech_client, audio_samples.mp4_video_only(), filename="clip.mp4")

    assert response.status_code == 422
    assert error_code(response) == "no_audio_stream"
    assert stt_engine.calls == []


def test_opening_is_not_proof_of_audio(tmp_path, monkeypatch):
    """FFmpeg opens a PNG happily - it has an image demuxer. Were the signature
    layer ever fooled, the demuxer allow-list still refuses it."""
    path = tmp_path / "image"
    import cv2
    import numpy as np

    ok, png = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    path.write_bytes(png.tobytes())
    monkeypatch.setattr(audio_module, "container_family", lambda head: "wav")

    with pytest.raises(AudioDecodeError, match="not an audio container"):
        inspect_audio(path, max_seconds=60)


def test_a_demuxer_outside_the_allow_list_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "clip"
    path.write_bytes(audio_samples.wav())
    monkeypatch.setattr(audio_module, "AUDIO_DEMUXERS", frozenset({"mp3"}))

    with pytest.raises(AudioDecodeError, match="not an audio container"):
        inspect_audio(path, max_seconds=60)


# -- layer: length -----------------------------------------------------------------


def test_a_recording_with_no_declared_length_is_measured_by_decoding(speech_client):
    """#8 - MediaRecorder's WebM declares no duration anywhere."""
    response = post_audio(speech_client, audio_samples.recorder_webm(seconds=3.0))

    assert response.status_code == 200, response.text
    assert response.json()["duration_seconds"] == pytest.approx(3.0, abs=0.05)


def test_a_declared_length_over_the_cap_is_refused_without_decoding(
    capped, stt_engine, monkeypatch
):
    """#9"""

    def must_not_decode(*args, **kwargs):
        raise AssertionError("decoded a file its metadata had already disqualified")

    monkeypatch.setattr(audio_module, "_decoded_seconds", must_not_decode)

    with capped(stt_max_audio_seconds=2.0) as client:
        response = post_audio(client, audio_samples.wav(seconds=6.0))

    assert response.status_code == 422
    assert error_code(response) == "audio_too_long"
    assert response.json()["error"]["details"]["measured"] == "metadata"
    assert stt_engine.calls == []


def test_an_undeclared_length_over_the_cap_is_caught_by_decoding(capped, stt_engine):
    """#10 - nothing declared, so only decoding can tell."""
    with capped(stt_max_audio_seconds=2.0) as client:
        response = post_audio(client, audio_samples.recorder_webm(seconds=6.0))

    assert response.status_code == 422
    assert error_code(response) == "audio_too_long"
    assert response.json()["error"]["details"]["measured"] == "decoded"
    assert stt_engine.calls == []


def test_a_declared_length_that_lies_is_caught_by_decoding(capped, stt_engine, monkeypatch):
    """#10 - declared short, really long. Metadata may only ever say no."""
    monkeypatch.setattr(audio_module, "_declared_seconds", lambda *args: 1.0)

    with capped(stt_max_audio_seconds=2.0) as client:
        response = post_audio(client, audio_samples.wav(seconds=6.0))

    assert response.status_code == 422
    assert response.json()["error"]["details"]["measured"] == "decoded"


def test_decoding_stops_as_soon_as_the_cap_is_passed(tmp_path):
    """#11 - asserted on how much was decoded, not on wall-clock time.

    Twenty seconds of audio against a two-second cap: had the decoder run to
    the end before refusing, the measured length would be twenty.
    """
    path = tmp_path / "long"
    path.write_bytes(audio_samples.recorder_webm(seconds=20.0))

    with pytest.raises(AudioTooLongError) as refusal:
        inspect_audio(path, max_seconds=2.0)

    decoded = refusal.value.details["duration_seconds"]
    assert 2.0 < decoded < 2.1


def test_a_clip_inside_the_cap_is_measured_exactly(tmp_path):
    path = tmp_path / "clip"
    path.write_bytes(audio_samples.recorder_webm(seconds=4.0))

    info = inspect_audio(path, max_seconds=5.0)

    assert info.container == "webm"
    assert info.codec == "opus"
    assert info.duration_seconds == pytest.approx(4.0, abs=0.05)


# -- layer: bytes (absolute) -------------------------------------------------------


def test_an_empty_upload_is_refused(speech_client, stt_engine):
    """#12"""
    response = post_audio(speech_client, b"", content_type="audio/wav")

    assert response.status_code == 422
    assert error_code(response) == "invalid_upload"
    assert stt_engine.calls == []


def test_an_upload_over_the_byte_cap_is_refused(capped, stt_engine):
    """#13"""
    with capped(stt_max_audio_bytes=10_000) as client:
        response = post_audio(client, audio_samples.wav(seconds=2.0))

    assert response.status_code == 413
    assert error_code(response) == "payload_too_large"
    assert stt_engine.calls == []


# -- temp files --------------------------------------------------------------------


@pytest.fixture
def temp_dir(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "tmp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


def test_the_upload_is_on_disk_for_the_engine_and_gone_after(
    speech_client, stt_engine, temp_dir
):
    """#53"""
    response = post_audio(speech_client, audio_samples.wav())

    assert response.status_code == 200
    assert stt_engine.path_existed == [True]
    assert list(temp_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("payload", "limits"),
    [
        (lambda: os.urandom(4096), {}),  # signature
        (lambda: b"\x1a\x45\xdf\xa3" + os.urandom(4096), {}),  # decoder
        (audio_samples.mp4_video_only, {}),  # no audio stream
        (lambda: audio_samples.wav(seconds=6.0), {"stt_max_audio_seconds": 2.0}),  # length
        (lambda: audio_samples.wav(seconds=2.0), {"stt_max_audio_bytes": 10_000}),  # bytes
    ],
    ids=["signature", "decoder", "no-audio", "too-long", "too-big"],
)
def test_the_upload_is_removed_whichever_layer_refuses_it(capped, temp_dir, payload, limits):
    """#54"""
    with capped(**limits) as client:
        response = post_audio(client, payload())

    assert response.status_code in (413, 422)
    assert list(temp_dir.iterdir()) == []


def test_the_upload_is_removed_when_the_engine_fails(speech_client, stt_engine, temp_dir):
    """#54 - the last layer too."""
    stt_engine.error = RuntimeError("engine blew up")

    with pytest.raises(RuntimeError):
        post_audio(speech_client, audio_samples.wav())

    assert list(temp_dir.iterdir()) == []
