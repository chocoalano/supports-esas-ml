"""`SpeechToTextEngine` backed by faster-whisper (CTranslate2).

`faster_whisper` is imported on first use - the first transcription, or warm-up
when it is turned on - so the application, its OpenAPI schema and the test
suite run without it installed, and a Face runtime never loads it.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.core.config import Settings
from app.core.errors import (
    AudioDecodeError,
    LanguageNotSupportedError,
    SpeechEngineUnavailableError,
)
from app.services.speech.stt import TranscribeOptions, Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


class FasterWhisperEngine:
    name = "faster_whisper"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        # Double-checked, as `InsightFaceEngine.load`: requests pass the
        # semaphore one after another, but warm-up and the first request can
        # still race, and loading twice is hundreds of megabytes twice.
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise SpeechEngineUnavailableError(
                    "faster-whisper is not installed. Run `pip install -r requirements.txt`.",
                    details={"import_error": str(exc)},
                ) from exc

            settings = self._settings
            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)...",
                settings.stt_model,
                settings.stt_device,
                settings.stt_compute_type,
            )
            try:
                model = WhisperModel(
                    settings.stt_model,
                    device=settings.stt_device,
                    compute_type=settings.stt_compute_type,
                    cpu_threads=settings.stt_cpu_threads,
                    download_root=settings.stt_model_root,
                )
            except Exception as exc:  # pragma: no cover - depends on local models
                raise SpeechEngineUnavailableError(
                    "Failed to initialise the speech-to-text engine.",
                    details={"reason": str(exc)},
                ) from exc

            self._model = model
            logger.info("Speech-to-text engine ready.")

    def transcribe(self, path: Path, options: TranscribeOptions) -> Transcript:
        self.load()
        model = self._model
        assert model is not None

        if options.language is not None and options.language not in model.supported_languages:
            raise LanguageNotSupportedError(
                f"Language '{options.language}' is not supported by the speech engine.",
                details={"language": options.language},
            )

        import av  # already loaded: faster-whisper decodes with it

        try:
            # A generator: the inference happens while it is consumed, so the
            # list() is inside the `try` too.
            segments, info = model.transcribe(
                str(path),
                language=options.language,
                beam_size=self._settings.stt_beam_size,
                vad_filter=self._settings.stt_vad_filter,
            )
            collected = [
                TranscriptSegment(
                    start=round(segment.start, 3),
                    end=round(segment.end, 3),
                    text=segment.text.strip(),
                )
                for segment in segments
            ]
        except av.error.FFmpegError as exc:
            # The validator decoded this file a moment ago, so this is rare -
            # but it is still the caller's audio, not our fault.
            raise AudioDecodeError("The audio could not be decoded for transcription.") from exc

        return Transcript(
            text=" ".join(segment.text for segment in collected if segment.text),
            language=info.language,
            # Reported as 1.0 when the language was given, which is a statement
            # about the request rather than a measurement: say nothing instead.
            language_probability=(
                None if options.language else round(float(info.language_probability), 4)
            ),
            segments=tuple(collected),
        )
