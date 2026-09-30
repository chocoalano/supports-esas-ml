"""Speech endpoints: transcribe audio, synthesise speech, and describe both.

Nothing in this module knows which provider does the work, or what a provider
calls a voice. Every route is behind `GuardDep`, the two GETs included - they
are cheap, and that is exactly why they are easy to forget.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, Response

from app.api.deps import GuardDep, SettingsDep, SttServiceDep, TtsEnabledDep, TtsServiceDep
from app.core.config import PROSODY_RANGE
from app.schemas.speech import (
    PercentRange,
    SpeechCapabilitiesResponse,
    SttCapabilities,
    SynthesizeRequest,
    TranscribeForm,
    TranscriptionResponse,
    TranscriptSegmentReport,
    TtsCapabilities,
    VoiceReport,
    VoicesResponse,
)
from app.schemas.verification import ErrorResponse
from app.services.speech.audio import SUPPORTED_CONTAINERS
from app.services.speech.tts import MEDIA_TYPE

router = APIRouter(prefix="/speech", tags=["speech"])

#: The alias the audio was spoken in. Never the provider's identifier for it.
VOICE_HEADER = "X-Speech-Voice"

_GUARDED = {
    401: {"model": ErrorResponse, "description": "Missing or unrecognised API key"},
    429: {"model": ErrorResponse, "description": "Too many requests for this key"},
}


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe a recording to text",
    responses={
        **_GUARDED,
        413: {"model": ErrorResponse, "description": "Upload too large"},
        415: {"model": ErrorResponse, "description": "Declared Content-Type not accepted"},
        422: {
            "model": ErrorResponse,
            "description": "Not audio, no audio stream, too long, or language not supported",
        },
        503: {"model": ErrorResponse, "description": "Speech-to-text disabled or unavailable"},
    },
)
async def transcribe(
    caller: GuardDep,
    service: SttServiceDep,
    form: Annotated[TranscribeForm, Form()],
) -> TranscriptionResponse:
    """Synchronous and bounded: the audio's length is capped, not the caller's patience.

    Measured and logged by the service, one event per request whatever its outcome.
    """
    result = await service.transcribe_upload(form.audio, language=form.language, caller=caller)
    transcript = result.transcript

    return TranscriptionResponse(
        text=transcript.text,
        language=transcript.language,
        language_probability=transcript.language_probability,
        duration_seconds=result.audio.duration_seconds,
        segments=(
            [
                TranscriptSegmentReport(start=segment.start, end=segment.end, text=segment.text)
                for segment in transcript.segments
            ]
            if form.include_segments
            else []
        ),
        processing_ms=result.processing_ms,
    )


@router.post(
    "/synthesize",
    response_class=Response,
    summary="Speak a text, as MP3",
    responses={
        200: {"content": {MEDIA_TYPE: {}}, "description": "The speech, as MP3."},
        **_GUARDED,
        413: {"model": ErrorResponse, "description": "Text too long"},
        422: {"model": ErrorResponse, "description": "Invalid text, rate, volume or voice"},
        502: {"model": ErrorResponse, "description": "The speech provider failed"},
        503: {"model": ErrorResponse, "description": "Text-to-speech disabled or unavailable"},
        504: {"model": ErrorResponse, "description": "The speech provider timed out"},
    },
)
async def synthesize(
    caller: GuardDep,
    service: TtsServiceDep,
    payload: SynthesizeRequest,
) -> Response:
    """The whole MP3 is in hand before the 200 is sent: a failure is always a JSON error."""
    result = await service.synthesize(
        text=payload.text,
        voice=payload.voice,
        rate=payload.rate,
        volume=payload.volume,
        caller=caller,
    )

    return Response(
        content=result.audio,
        media_type=result.media_type,
        headers={
            # Inline: this is a service-to-service response, not a download.
            "Content-Disposition": 'inline; filename="speech.mp3"',
            VOICE_HEADER: result.voice,
        },
    )


@router.get(
    "/voices",
    response_model=VoicesResponse,
    summary="The voices a caller may ask for",
    responses={
        **_GUARDED,
        503: {"model": ErrorResponse, "description": "Text-to-speech disabled"},
    },
)
async def voices(caller: GuardDep, settings: SettingsDep, _: TtsEnabledDep) -> VoicesResponse:
    """Configuration, not discovery: no provider is asked, so this cannot fail on one.

    Only the approved aliases are listed, and only what a caller needs of them.
    """
    return VoicesResponse(
        default_voice=settings.tts_default_voice,
        # Field by field: `provider_voice` is on the alias and must not reach
        # the caller, so nothing here copies an alias wholesale.
        voices=[
            VoiceReport(
                id=alias.id, label=alias.label, language=alias.language, gender=alias.gender
            )
            for alias in settings.tts_voice_aliases
        ],
    )


@router.get(
    "/capabilities",
    response_model=SpeechCapabilitiesResponse,
    summary="What this service accepts and returns",
    responses=_GUARDED,
)
async def capabilities(caller: GuardDep, settings: SettingsDep) -> SpeechCapabilitiesResponse:
    """Answered from settings alone: no model is loaded and no provider is contacted."""
    low, high = PROSODY_RANGE

    return SpeechCapabilitiesResponse(
        stt=SttCapabilities(
            enabled=settings.stt_enabled,
            provider=settings.stt_provider,
            model=settings.stt_model,
            supported_containers=list(SUPPORTED_CONTAINERS),
            allowed_content_types=settings.stt_allowed_audio_types,
            max_audio_bytes=settings.stt_max_audio_bytes,
            max_audio_seconds=settings.stt_max_audio_seconds,
            default_language=settings.stt_default_language,
            allowed_languages=settings.stt_allowed_languages,
        ),
        tts=TtsCapabilities(
            enabled=settings.tts_enabled,
            provider=settings.tts_provider,
            media_type=MEDIA_TYPE,
            default_voice=settings.tts_default_voice,
            voices=[alias.id for alias in settings.tts_voice_aliases],
            max_text_chars=settings.tts_max_text_chars,
            rate=PercentRange(min=low, max=high),
            volume=PercentRange(min=low, max=high),
            timeout_seconds=settings.tts_timeout_seconds,
        ),
    )
