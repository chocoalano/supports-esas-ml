"""Real audio files, encoded with PyAV while the suite runs.

Generated rather than checked in: the validator under test is the FFmpeg
decoder's judgement, so the files have to be what an encoder really produces -
including the WebM a browser's MediaRecorder writes, which declares no duration
at all (the muxer's `live` mode reproduces exactly that).
"""

from __future__ import annotations

import io
import math
from fractions import Fraction
from functools import lru_cache

import av
import numpy as np


def _tone(seconds: float, rate: int) -> np.ndarray:
    samples = int(seconds * rate)
    t = np.arange(samples) / rate
    return (0.3 * np.sin(2 * math.pi * 440.0 * t) * 32767).astype(np.int16)


@lru_cache
def encode(
    container: str, codec: str, *, seconds: float, rate: int = 48000, live: bool = False
) -> bytes:
    """A mono 440 Hz tone of `seconds`, in `container` with `codec`."""
    buffer = io.BytesIO()
    options = {"live": "1"} if live else {}

    with av.open(buffer, "w", format=container, container_options=options) as output:
        stream = output.add_stream(codec, rate=rate)
        stream.layout = "mono"
        pcm = _tone(seconds, rate)
        step = stream.codec_context.frame_size or 1024

        for start in range(0, len(pcm), step):
            frame = av.AudioFrame.from_ndarray(
                pcm[start : start + step].reshape(1, -1), format="s16", layout="mono"
            )
            frame.sample_rate = rate
            frame.pts = start
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                output.mux(packet)

        for packet in stream.encode(None):
            output.mux(packet)

    return buffer.getvalue()


def wav(seconds: float = 2.0) -> bytes:
    return encode("wav", "pcm_s16le", seconds=seconds, rate=16000)


def mp3(seconds: float = 2.0) -> bytes:
    return encode("mp3", "libmp3lame", seconds=seconds, rate=44100)


def m4a(seconds: float = 2.0) -> bytes:
    return encode("mp4", "aac", seconds=seconds, rate=44100)


def ogg_opus(seconds: float = 2.0) -> bytes:
    return encode("ogg", "libopus", seconds=seconds)


def flac(seconds: float = 2.0) -> bytes:
    return encode("flac", "flac", seconds=seconds, rate=16000)


def webm_opus(seconds: float = 2.0) -> bytes:
    return encode("webm", "libopus", seconds=seconds)


def recorder_webm(seconds: float = 2.0) -> bytes:
    """WebM/Opus with no duration anywhere, as a browser's MediaRecorder writes it."""
    return encode("webm", "libopus", seconds=seconds, live=True)


@lru_cache
def mp4_video_only(frames: int = 10) -> bytes:
    """A perfectly valid MP4 with a picture and no sound."""
    buffer = io.BytesIO()

    with av.open(buffer, "w", format="mp4") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(frames):
            image = np.full((48, 64, 3), index * 20 % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)

    return buffer.getvalue()


PNG_HEADER = b"\x89PNG\r\n\x1a\n"
