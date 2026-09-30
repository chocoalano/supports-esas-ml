"""The one place speech work is measured.

Every transcription and every synthesis produces exactly one event - on
success, on a refusal, on a failure, and when the caller hangs up - carrying
what an operator needs to see where time and capacity go: how long it waited
for a slot, how long the engine took, how much audio, which model on which
device. Never the audio, the transcript, or the text to be spoken: those are
what somebody said.

The services emit events; a sink decides what to do with them. The sink here
writes one structured log line. A metrics backend is another sink behind the
same Protocol, so adding one touches neither routes nor providers:

    stt_requests_total{outcome}      <- count of SttEvent by outcome
    stt_duration_seconds             <- SttEvent.total_ms
    stt_audio_seconds                <- SttEvent.audio_seconds
    stt_active_requests              <- begin() / end() pairs
    tts_requests_total{outcome}      <- count of TtsEvent by outcome
    tts_duration_seconds             <- TtsEvent.total_ms
    speech_errors_total{code}        <- events whose outcome is not "ok"
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

logger = logging.getLogger("app.speech")

Kind = Literal["stt", "tts"]

#: The outcome of a request the client abandoned before it was answered.
CANCELLED = "cancelled"
#: The outcome of an exception that is not one of ours - a bug, not a refusal.
INTERNAL_ERROR = "internal_error"


@dataclass
class SttEvent:
    caller: str
    provider: str
    model: str
    device: str
    compute_type: str
    #: STT requests in flight in this process when this one began, itself included.
    in_flight: int = 0
    #: "ok", an error code, CANCELLED or INTERNAL_ERROR.
    outcome: str = CANCELLED
    #: HTTP status the caller received; 499 when it went away first.
    status: int = 499
    upload_bytes: int | None = None
    container: str | None = None
    audio_seconds: float | None = None
    language: str | None = None
    segments: int | None = None
    #: Waiting for an STT slot (FSA_STT_MAX_CONCURRENT).
    wait_ms: int | None = None
    #: Decoding the upload to validate and measure it.
    validate_ms: int | None = None
    #: The engine itself, model load included when this request paid for it.
    transcribe_ms: int | None = None
    total_ms: int = 0


@dataclass
class TtsEvent:
    caller: str
    provider: str
    in_flight: int = 0
    outcome: str = CANCELLED
    status: int = 499
    #: The alias used - never the provider's voice id. None if it was refused.
    voice: str | None = None
    text_chars: int = 0
    output_bytes: int | None = None
    #: Waiting for a TTS slot (FSA_TTS_MAX_CONCURRENT).
    wait_ms: int | None = None
    #: The provider round trip.
    synthesize_ms: int | None = None
    total_ms: int = 0


class SpeechTelemetry(Protocol):
    def begin(self, kind: Kind) -> int:
        """One more request of this kind in flight; returns how many now are."""

    def end(self, kind: Kind, event: SttEvent | TtsEvent) -> None:
        """The request is over, however it ended. Called exactly once per begin()."""


class LogTelemetry:
    """Writes each event as one `key=value` line, and counts requests in flight.

    One line per request, every field every time, so a log pipeline can parse
    it without knowing which fields a given outcome fills in.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight = {"stt": 0, "tts": 0}

    def begin(self, kind: Kind) -> int:
        with self._lock:
            self._in_flight[kind] += 1
            return self._in_flight[kind]

    def end(self, kind: Kind, event: SttEvent | TtsEvent) -> None:
        with self._lock:
            self._in_flight[kind] -= 1

        level = logging.INFO
        if event.status >= 500:
            level = logging.ERROR if event.outcome == INTERNAL_ERROR else logging.WARNING

        fields = asdict(event)
        # What happened first, so `grep outcome=` reads down one column.
        ordered = {"outcome": fields.pop("outcome"), "status": fields.pop("status"), **fields}
        logger.log(level, "speech.%s %s", kind, logfmt(ordered))

    def in_flight(self, kind: Kind) -> int:
        with self._lock:
            return self._in_flight[kind]


def logfmt(fields: dict) -> str:
    def value(item: object) -> str:
        if item is None:
            return "-"
        if isinstance(item, float):
            return f"{item:.3f}".rstrip("0").rstrip(".")
        text = str(item)
        if not text or any(c in text for c in ' "='):
            return '"' + text.replace('"', '\\"') + '"'
        return text

    return " ".join(f"{key}={value(item)}" for key, item in fields.items())
