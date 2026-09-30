"""Audio uploads: receiving them, and deciding whether they are audio at all.

The layers, and which of them is an authority:

    bytes on disk ........ `receive_audio`   authority: absolute, content-blind
    declared type ........ route             authority: none - skipped when absent
    signature ............ `container_family` authority: none - early rejection only
    decoder opens it ..... `inspect_audio`   authority: "a readable container"
    audio stream exists .. `inspect_audio`   authority: "there is sound in it"
    decoded length ....... `inspect_audio`   authority: the only honest duration

"The decoder opened it" is not proof of audio: FFmpeg has demuxers for images
and video too, and a video-only MP4 is a perfectly good container. Each layer
exists because the one before it lets something through.

`av` (PyAV) is imported lazily, like every heavy dependency here, so a Face
runtime never loads it. It ships its own FFmpeg build, and so does OpenCV.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.errors import (
    AudioDecodeError,
    AudioTooLongError,
    InvalidUploadError,
    NoAudioStreamError,
    PayloadTooLargeError,
    SpeechEngineUnavailableError,
)

CHUNK_SIZE = 1024 * 1024
HEAD_BYTES = 64


def _adts(head: bytes) -> bool:
    # 12-bit sync, then ID, then a 2-bit layer that is always 00 for AAC.
    return len(head) >= 2 and head[0] == 0xFF and head[1] & 0xF6 == 0xF0


def _mpeg_audio(head: bytes) -> bool:
    # 11-bit frame sync. Checked after ADTS, which shares the first 11 bits.
    return len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0


#: Container signatures, in the order they are tried. `supported_containers` in
#: `/speech/capabilities` is derived from this table rather than from a setting,
#: so it cannot claim a format the code does not recognise.
SIGNATURES: tuple[tuple[str, Callable[[bytes], bool]], ...] = (
    ("wav", lambda head: head[:4] == b"RIFF" and head[8:12] == b"WAVE"),
    ("flac", lambda head: head[:4] == b"fLaC"),
    ("ogg", lambda head: head[:4] == b"OggS"),
    ("webm", lambda head: head[:4] == b"\x1a\x45\xdf\xa3"),
    ("mp4", lambda head: head[4:8] == b"ftyp"),
    # An ID3v2 tag says "tagged audio follows", usually MP3; the decoder decides.
    ("mp3", lambda head: head[:3] == b"ID3"),
    ("aac", _adts),
    ("mp3", _mpeg_audio),
)

SUPPORTED_CONTAINERS: tuple[str, ...] = tuple(dict.fromkeys(name for name, _ in SIGNATURES))

#: FFmpeg demuxers a file may be read with, as `container.format.name` reports
#: them. Anything else is refused even when it opens.
#:
#: This is the line that matters for what faster-whisper does next. It opens the
#: same path with the same `av` build, and FFmpeg probing is a function of the
#: bytes and the file name - both identical here, the name carrying no suffix -
#: so the demuxer allowed here is the demuxer it gets. None of these follow a
#: reference out of the file: no playlist, no concat list, no image sequence.
AUDIO_DEMUXERS = frozenset(
    {
        "wav",
        "mp3",
        "aac",
        "ogg",
        "flac",
        "matroska,webm",
        "mov,mp4,m4a,3gp,3g2,mj2",
    }
)


@dataclass(frozen=True)
class AudioInfo:
    """What the decoder found. `duration_seconds` is decoded, never declared."""

    container: str
    codec: str
    duration_seconds: float
    sample_rate: int
    channels: int


def container_family(head: bytes) -> str | None:
    """The container a file claims to be by its first bytes, or None.

    Early rejection, not a verdict: `ID3` followed by text passes here and is
    caught by the decoder. What it buys is not paying for a decoder on a file
    that is plainly a PDF.
    """
    for name, matches in SIGNATURES:
        if matches(head):
            return name

    return None


@asynccontextmanager
async def receive_audio(upload: UploadFile, *, max_bytes: int) -> AsyncIterator[Path]:
    """Stream an upload to a temp file that is removed on exit, success or not.

    Not `media.temp_file`, which keeps the caller's file extension. FFmpeg's
    probe weighs the file name, so a suffix chosen by the caller could steer
    which demuxer reads the bytes - here and again inside faster-whisper. The
    file is written without one, and the name the caller sent decides nothing.
    """
    # Not a `with` block: the file must outlive the write loop and be handed to
    # the decoder by path, so cleanup is done in the `finally` below instead.
    handle = tempfile.NamedTemporaryFile(prefix="fsa-audio-", suffix="", delete=False)  # noqa: SIM115
    path = Path(handle.name)
    total = 0
    try:
        while True:
            chunk = await upload.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise PayloadTooLargeError(
                    f"Audio exceeds the maximum size of {max_bytes} bytes.",
                    details={"filename": upload.filename, "max_bytes": max_bytes},
                )
            handle.write(chunk)
        handle.close()
        await upload.close()

        if total == 0:
            raise InvalidUploadError("Audio is empty.", details={"filename": upload.filename})
        yield path
    finally:
        if not handle.closed:
            handle.close()
        path.unlink(missing_ok=True)


def inspect_audio(path: Path, *, max_seconds: float) -> AudioInfo:
    """Validate an audio file and measure it, or raise the error that says why not.

    Blocking and CPU-bound: call it from a worker thread.

    Raises:
        InvalidUploadError: the first bytes match no audio container.
        AudioDecodeError: the decoder cannot read it, reads it as something
            other than audio, or finds no decodable sound in it.
        NoAudioStreamError: a readable container with no audio stream.
        AudioTooLongError: longer than `max_seconds`, by metadata or by decoding.
        SpeechEngineUnavailableError: PyAV is not installed.
    """
    with path.open("rb") as handle:
        head = handle.read(HEAD_BYTES)

    family = container_family(head)
    if family is None:
        raise InvalidUploadError(
            "Audio is not a recognisable audio file.",
            details={"supported_containers": list(SUPPORTED_CONTAINERS)},
        )

    av = _import_av()

    try:
        container = av.open(
            str(path),
            # A local file may only ever read local files. Belt and braces: no
            # allowed demuxer below opens anything, but this holds even for the
            # ones that are refused after opening.
            container_options={"protocol_whitelist": "file"},
            # As faster-whisper opens it: a tag in a legacy encoding is not a
            # reason to refuse the sound.
            metadata_errors="ignore",
        )
    except av.error.FFmpegError as exc:
        raise AudioDecodeError(
            "The audio could not be read.", details={"container": family}
        ) from exc

    with container:
        if container.format.name not in AUDIO_DEMUXERS:
            raise AudioDecodeError(
                "The file is not an audio container.",
                details={"supported_containers": list(SUPPORTED_CONTAINERS)},
            )

        if not container.streams.audio:
            raise NoAudioStreamError("The file contains no audio stream.")

        stream = container.streams.audio[0]

        declared = _declared_seconds(av, container, stream)
        if declared is not None and declared > max_seconds:
            # The cheap refusal: nothing has been decoded yet. Metadata is only
            # ever trusted in this direction - to say no early, never to say
            # yes, because it is sometimes absent and sometimes wrong.
            raise _too_long(declared, max_seconds, measured="metadata")

        decoded = _decoded_seconds(av, container, stream, max_seconds)
        if decoded <= 0:
            raise AudioDecodeError("The file contains no decodable audio.")

        return AudioInfo(
            container=family,
            codec=stream.codec_context.name,
            duration_seconds=round(decoded, 3),
            sample_rate=int(stream.codec_context.sample_rate or 0),
            channels=int(stream.codec_context.channels or 0),
        )


def _declared_seconds(av, container, stream) -> float | None:  # noqa: ANN001 - lazy `av` types
    """Length according to the file itself, when it says. Browsers' WebM often does not."""
    if stream.duration is not None and stream.time_base:
        seconds = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        seconds = container.duration / av.time_base
    else:
        return None

    return seconds if seconds > 0 else None


def _decoded_seconds(av, container, stream, max_seconds: float) -> float:  # noqa: ANN001
    """Decode the audio stream, stopping the moment it runs past `max_seconds`.

    The authority on length. It never decodes a long clip to the end in order
    to refuse it: the overshoot is at most one packet.

    Undecodable packets are skipped rather than fatal, as faster-whisper skips
    them. Refusing here what the engine would have transcribed helps nobody.
    """
    total = 0.0

    try:
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue

            for frame in frames:
                if frame.sample_rate:
                    total += frame.samples / frame.sample_rate

            if total > max_seconds:
                raise _too_long(total, max_seconds, measured="decoded")
    except av.error.FFmpegError:
        # The container itself broke mid-file. What decoded before it is what
        # faster-whisper will get too; if that is nothing, the caller says so.
        pass

    return total


def _too_long(seconds: float, max_seconds: float, *, measured: str) -> AudioTooLongError:
    return AudioTooLongError(
        f"Audio is longer than the maximum of {max_seconds:g} seconds.",
        details={
            "max_seconds": max_seconds,
            # "At least" when measured by decoding: decoding stopped here.
            "duration_seconds": round(seconds, 3),
            "measured": measured,
        },
    )


def _import_av():  # noqa: ANN202 - the module itself
    try:
        import av
    except ImportError as exc:
        raise SpeechEngineUnavailableError(
            "The audio decoder is not installed. Run `pip install -r requirements.txt`.",
            details={"import_error": str(exc)},
        ) from exc

    return av
