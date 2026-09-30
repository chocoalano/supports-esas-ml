"""Which concrete engine backs each Protocol.

Provider modules are imported inside these functions, not at the top: a
process only ever loads the provider it was configured to use, and a Face
runtime - which never calls either - loads none. Constructing an engine is
cheap and imports nothing heavy; the library itself is imported on first use.
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import SpeechEngineUnavailableError
from app.services.speech.stt import SpeechToTextEngine
from app.services.speech.tts import TextToSpeechEngine


def build_stt_engine(settings: Settings) -> SpeechToTextEngine:
    if settings.stt_provider == "faster_whisper":
        from app.services.speech.providers.faster_whisper import FasterWhisperEngine

        return FasterWhisperEngine(settings)

    raise SpeechEngineUnavailableError(f"Unknown STT provider '{settings.stt_provider}'.")


def build_tts_engine(settings: Settings) -> TextToSpeechEngine:
    if settings.tts_provider == "edge":
        from app.services.speech.providers.edge import EdgeTtsEngine

        return EdgeTtsEngine(settings)

    raise SpeechEngineUnavailableError(f"Unknown TTS provider '{settings.tts_provider}'.")


def stt_provisioning_problem(settings: Settings) -> str | None:
    """Why the configured STT model is not ready to load, checked on disk only."""
    if settings.stt_provider == "faster_whisper":
        from app.services.speech.providers.faster_whisper import provisioning_problem

        return provisioning_problem(settings)

    return None
