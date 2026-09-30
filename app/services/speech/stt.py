"""Speech-to-text: the engine interface, and the service that bounds it.

The engine takes a path, not decoded samples. That costs a second decode - the
validator decodes to measure, the engine decodes to listen - and it is paid on
purpose: a path is what every provider can take, a local model and a remote API
alike, so adding one never means changing this interface for all of them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

from anyio import Semaphore, to_thread
from fastapi import UploadFile

from app.core.config import LANGUAGE_CODE, Settings
from app.core.errors import AppError, InvalidSpeechRequestError, LanguageNotSupportedError
from app.services.media import ensure_content_type
from app.services.speech.audio import AudioInfo, inspect_audio, receive_audio
from app.services.speech.telemetry import INTERNAL_ERROR, LogTelemetry, SpeechTelemetry, SttEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscribeOptions:
    #: Lowercase ISO 639 code, or None to let the engine detect it.
    language: str | None


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    language_probability: float | None
    segments: tuple[TranscriptSegment, ...] = ()


class SpeechToTextEngine(Protocol):
    """Interface the API depends on - trivially fakeable in tests."""

    name: str

    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def transcribe(self, path: Path, options: TranscribeOptions) -> Transcript:
        """Blocking. Loads the model on first use. Runs in a worker thread."""


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: Transcript
    audio: AudioInfo
    processing_ms: int


class SpeechToTextService:
    def __init__(
        self,
        *,
        engine: SpeechToTextEngine,
        settings: Settings,
        limiter: Semaphore,
        telemetry: SpeechTelemetry | None = None,
    ) -> None:
        self._engine = engine
        self._settings = settings
        self._limiter = limiter
        self._telemetry = telemetry or LogTelemetry()

    async def transcribe_upload(
        self, upload: UploadFile, *, language: str | None, caller: str
    ) -> TranscriptionResult:
        """The whole request: language, declared type, upload, validation, engine.

        One place, so every way a request can end - answered, refused, failed,
        abandoned - is measured once, here, and not in each route that calls it.
        """
        settings = self._settings
        event = SttEvent(
            caller=caller,
            provider=settings.stt_provider,
            model=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            in_flight=self._telemetry.begin("stt"),
        )
        started = time.perf_counter()
        try:
            options = self.options_for(language)
            event.language = options.language
            ensure_content_type(upload, settings.stt_allowed_audio_types, label="Audio")
            async with receive_audio(upload, max_bytes=settings.stt_max_audio_bytes) as path:
                event.upload_bytes = path.stat().st_size
                result = await self.transcribe(path, options, event=event)
            event.outcome, event.status = "ok", 200
            return result
        except AppError as exc:
            event.outcome, event.status = exc.code, exc.status_code
            raise
        except Exception:
            event.outcome, event.status = INTERNAL_ERROR, 500
            raise
        finally:
            event.total_ms = _ms_since(started)
            self._telemetry.end("stt", event)

    def options_for(self, language: str | None) -> TranscribeOptions:
        """Resolve and check the language, before any upload is written or decoded.

        Raises:
            InvalidSpeechRequestError: not a language code at all.
            LanguageNotSupportedError: a code outside `FSA_STT_ALLOWED_LANGUAGES`.
        """
        requested = language.strip().lower() if language and language.strip() else None

        if requested is not None and not LANGUAGE_CODE.fullmatch(requested):
            raise InvalidSpeechRequestError(
                "language must be an ISO 639 code such as 'id' or 'en'.",
                details={"field": "language"},
            )

        effective = requested or self._settings.stt_default_language
        self._ensure_allowed(effective)

        return TranscribeOptions(language=effective)

    async def transcribe(
        self, path: Path, options: TranscribeOptions, *, event: SttEvent | None = None
    ) -> TranscriptionResult:
        """Validate and transcribe a file already on disk. `event` collects timings."""
        event = event or SttEvent(caller="-", provider="-", model="-", device="-", compute_type="-")
        started = time.perf_counter()

        # Validation decodes too, so it waits for the same slot as inference:
        # CPU-bound audio work in this process is bounded by one number,
        # FSA_STT_MAX_CONCURRENT, whichever kind of work it is. An invalid
        # upload queues behind a transcription - the price of CPU use that
        # stays bounded under a burst of uploads.
        async with self._limiter:
            event.wait_ms = _ms_since(started)
            step = time.perf_counter()
            audio = await to_thread.run_sync(
                partial(inspect_audio, path, max_seconds=self._settings.stt_max_audio_seconds)
            )
            event.validate_ms = _ms_since(step)
            event.container, event.audio_seconds = audio.container, audio.duration_seconds
            step = time.perf_counter()
            transcript = await to_thread.run_sync(self._engine.transcribe, path, options)
            event.transcribe_ms = _ms_since(step)

        event.language, event.segments = transcript.language, len(transcript.segments)
        if options.language is None:
            # Nothing was asked for, so nothing was checked before inference.
            # The allow-list still means what it says about what comes back.
            self._ensure_allowed(transcript.language)

        return TranscriptionResult(
            transcript=transcript,
            audio=audio,
            processing_ms=int((time.perf_counter() - started) * 1000),
        )

    def _ensure_allowed(self, language: str | None) -> None:
        allowed = self._settings.stt_allowed_languages

        if language is not None and allowed and language not in allowed:
            raise LanguageNotSupportedError(
                f"Language '{language}' is not supported by this service.",
                details={"language": language, "allowed": allowed},
            )


def _ms_since(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
