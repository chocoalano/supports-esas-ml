"""Request/response models for the speech endpoints.

Nothing here names a provider except `provider` in capabilities, which is there
for operators. Voices are aliases this service owns; a provider's own voice
identifier never appears in a request, a response, or an error.

Requests are strict: an unknown field is a 422, not silently ignored. A caller
who sends `"format": "wav"` or `include_segment=true` is told, rather than
handed an MP3 or a transcript without segments and left to wonder why.
"""

from __future__ import annotations

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field

# --- /speech/transcribe --------------------------------------------------------


class TranscribeForm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: UploadFile = Field(
        description="The recording. wav, mp3, aac, m4a/mp4, ogg/opus, flac or webm. "
        "What it is is decided by decoding it - not by its file name, and not by the "
        "Content-Type sent with it."
    )
    language: str | None = Field(
        default=None,
        description="ISO 639 code such as `id`. Omit it to use the configured default, or "
        "to let the engine detect the language when there is none.",
    )
    include_segments: bool = Field(
        default=False, description="Return timed segments alongside the full text."
    )


class TranscriptSegmentReport(BaseModel):
    start: float = Field(description="Seconds from the start of the audio.")
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None = Field(
        default=None, description="The language transcribed - requested, or detected."
    )
    language_probability: float | None = Field(
        default=None,
        description="Confidence of the detection, 0..1. Null when the language was given.",
    )
    duration_seconds: float = Field(description="Length of the audio, measured by decoding it.")
    segments: list[TranscriptSegmentReport] = Field(
        default_factory=list, description="Empty unless `include_segments` was set."
    )
    processing_ms: int


# --- /speech/synthesize --------------------------------------------------------


class SynthesizeRequest(BaseModel):
    """The whole contract. The response is always MP3: there is no `format`."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="What to say. Leading and trailing whitespace is ignored.")
    voice: str | None = Field(
        default=None,
        description="A voice `id` from /speech/voices. Omitted or blank: the default voice.",
    )
    rate: str | None = Field(
        default=None,
        description="Speaking rate: a sign, 1-3 digits and `%`, from -100% to +100%.",
        examples=["+0%", "-10%"],
    )
    volume: str | None = Field(
        default=None,
        description="Volume: a sign, 1-3 digits and `%`, from -100% to +100%.",
        examples=["+0%", "+20%"],
    )


# --- /speech/voices ------------------------------------------------------------


class VoiceReport(BaseModel):
    id: str = Field(description="Send this as `voice`. Stable across provider changes.")
    label: str
    language: str = Field(description="BCP 47 tag, e.g. `id-ID`.")
    gender: str = Field(description="`male`, `female` or `neutral`.")


class VoicesResponse(BaseModel):
    default_voice: str = Field(description="The `id` used when a request names none.")
    voices: list[VoiceReport]


# --- /speech/capabilities --------------------------------------------------------


class PercentRange(BaseModel):
    min: int
    max: int


class SttCapabilities(BaseModel):
    enabled: bool
    provider: str = Field(
        description="For operators. Clients must not branch on it: it changes when the "
        "engine does, and the contract does not."
    )
    model: str = Field(description="For operators, like `provider`.")
    supported_containers: list[str]
    allowed_content_types: list[str]
    max_audio_bytes: int
    max_audio_seconds: float
    default_language: str | None
    allowed_languages: list[str] = Field(description="Empty means any language.")


class TtsCapabilities(BaseModel):
    enabled: bool
    provider: str = Field(
        description="For operators. Clients must not branch on it: it changes when the "
        "engine does, and the contract does not."
    )
    media_type: str = Field(description="Always `audio/mpeg`.")
    default_voice: str
    voices: list[str] = Field(description="Voice ids; /speech/voices describes them.")
    max_text_chars: int
    rate: PercentRange
    volume: PercentRange
    timeout_seconds: float


class SpeechCapabilitiesResponse(BaseModel):
    stt: SttCapabilities
    tts: TtsCapabilities
