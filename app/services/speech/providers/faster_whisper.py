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
                from faster_whisper.utils import download_model
            except ImportError as exc:
                logger.error(
                    "faster-whisper is not installed (%s). Run `pip install -r requirements.txt`.",
                    exc,
                )
                raise SpeechEngineUnavailableError(
                    "The speech-to-text engine is not installed on this server.",
                    details={"reason": "engine_not_installed"},
                ) from exc

            settings = self._settings
            try:
                model_path = _model_path(settings, download_model, local_files_only=False)
            except Exception as exc:
                # With HF_HUB_OFFLINE=1 - as production runs - this is a
                # filesystem lookup that fails in milliseconds, not a network
                # timeout. The operator needs the detail; the caller does not.
                logger.error(
                    "Speech-to-text model '%s' is not available in %s: %r. Provision it "
                    "with scripts/provision_stt_model.py before starting the Speech runtime.",
                    settings.stt_model,
                    settings.stt_model_root or "the HuggingFace cache",
                    exc,
                )
                raise SpeechEngineUnavailableError(
                    "The speech-to-text model is not provisioned on this server.",
                    details={"reason": "model_not_provisioned"},
                ) from exc

            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)...",
                settings.stt_model,
                settings.stt_device,
                settings.stt_compute_type,
            )
            try:
                model = WhisperModel(
                    model_path,
                    device=settings.stt_device,
                    compute_type=settings.stt_compute_type,
                    cpu_threads=settings.stt_cpu_threads,
                    # One CTranslate2 worker per allowed concurrent transcription.
                    # With the library's default of 1, concurrent calls queue
                    # inside the model and FSA_STT_MAX_CONCURRENT > 1 would add
                    # waiting, not throughput. Threads in use are therefore
                    # FSA_STT_MAX_CONCURRENT x FSA_STT_CPU_THREADS.
                    num_workers=max(1, settings.stt_max_concurrent),
                )
            except Exception as exc:
                # Device, compute type, a corrupt model: the operator's to fix,
                # and the library's words are for the log, not the caller.
                logger.error(
                    "Speech-to-text engine failed to load model '%s' (device=%s, "
                    "compute_type=%s): %r",
                    settings.stt_model,
                    settings.stt_device,
                    settings.stt_compute_type,
                    exc,
                )
                raise SpeechEngineUnavailableError(
                    "The speech-to-text engine could not be started.",
                    details={"reason": "engine_failed_to_load"},
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


def _model_path(settings: Settings, download_model, *, local_files_only: bool) -> str:  # noqa: ANN001
    """The directory holding the model's files, found - or fetched - by name.

    Resolved here rather than handed to `WhisperModel` as a name, because
    `WhisperModel` first tries the name as a path relative to the working
    directory: `FSA_STT_MODEL=small` with a directory called `small` wherever
    the service was started loads that directory instead. Only an absolute
    path is taken as a local model; anything else is a model name.

    Honours HF_HUB_OFFLINE: offline, a model that is not already cached raises
    at once instead of reaching for the network.
    """
    if Path(settings.stt_model).is_absolute():
        return settings.stt_model

    return download_model(
        settings.stt_model,
        local_files_only=local_files_only,
        cache_dir=settings.stt_model_root,
    )


def provisioning_problem(settings: Settings) -> str | None:
    """Why the configured model could not be loaded right now, as configured, or None.

    For boot: whether the device and compute type exist here, and whether the
    model is on disk - never a download and never a model load. It costs
    milliseconds and no memory beyond importing the library, and it turns
    `FSA_STT_DEVICE=cuda` on a host without a GPU into a line in the boot log
    instead of a 503 on the first transcription.
    """
    try:
        import ctranslate2
        from faster_whisper.utils import download_model
    except ImportError as exc:
        return f"faster-whisper is not installed ({exc})"

    if problem := _compute_problem(settings, ctranslate2):
        return problem

    try:
        path = _model_path(settings, download_model, local_files_only=True)
    except Exception as exc:
        return (
            f"model '{settings.stt_model}' is not in "
            f"{settings.stt_model_root or 'the HuggingFace cache'} ({type(exc).__name__})"
        )

    if not (Path(path) / "model.bin").is_file():
        return f"model directory {path} has no model.bin"

    return None


def _compute_problem(settings: Settings, ctranslate2) -> str | None:  # noqa: ANN001 - lazy module
    device = settings.stt_device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
        return "FSA_STT_DEVICE=cuda, but no CUDA device is visible to this process"

    try:
        supported = ctranslate2.get_supported_compute_types(device)
    except Exception as exc:
        return f"FSA_STT_DEVICE={settings.stt_device} cannot be used here ({exc})"

    compute = settings.stt_compute_type
    if compute not in ("default", "auto") and compute not in supported:
        return (
            f"FSA_STT_COMPUTE_TYPE={compute} is not supported on {device} "
            f"(supported: {', '.join(sorted(supported))})"
        )
    return None
