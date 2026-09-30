"""Text-to-speech: the engine interface, and the service that bounds it.

A provider is a stream of MP3 bytes and nothing more. Everything a caller can
get wrong, and everything a remote provider can do to us, is handled here once
rather than in each provider:

- which voice: the caller names an alias; only this module knows what the
  provider calls it
- `rate` / `volume`: parsed to integers before they go anywhere
- how long: a wall-clock timeout around the provider
- how big: a byte ceiling enforced while the bytes arrive
- what comes back: non-empty, and MP3, or it is an error

Nothing is sent to the caller until all of it is in hand, so a provider that
fails half-way produces a JSON error, never a truncated MP3 under a 200.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Protocol

from anyio import Semaphore, fail_after

from app.core.config import PROSODY_RANGE, Settings, VoiceAlias, parse_prosody_percent
from app.core.errors import (
    AppError,
    InvalidSpeechRequestError,
    PayloadTooLargeError,
    SpeechProviderError,
    SpeechProviderTimeoutError,
    TtsOutputTooLargeError,
    VoiceNotAvailableError,
)

logger = logging.getLogger(__name__)

MEDIA_TYPE = "audio/mpeg"


@dataclass(frozen=True)
class SynthesisRequest:
    """What a provider is handed. Every field is already validated."""

    text: str
    #: The provider's own identifier for the voice - resolved from an alias,
    #: never a string a caller sent.
    voice: str
    rate_percent: int
    volume_percent: int


class TextToSpeechEngine(Protocol):
    """Interface the API depends on - trivially fakeable in tests."""

    name: str

    def stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        """Yield MP3 bytes as the provider produces them. Network-bound: no thread."""


@dataclass(frozen=True)
class Synthesis:
    audio: bytes
    #: The alias that was used - the only name for a voice a caller ever sees.
    voice: str
    media_type: str = MEDIA_TYPE


class TextToSpeechService:
    def __init__(
        self, *, engine: TextToSpeechEngine, settings: Settings, limiter: Semaphore
    ) -> None:
        self._engine = engine
        self._settings = settings
        self._limiter = limiter

    async def synthesize(
        self,
        *,
        text: str,
        voice: str | None = None,
        rate: str | None = None,
        volume: str | None = None,
    ) -> Synthesis:
        alias, request = self.prepare(text=text, voice=voice, rate=rate, volume=volume)

        async with self._limiter:
            try:
                with fail_after(self._settings.tts_timeout_seconds):
                    audio = await self._collect(request)
            except TimeoutError as exc:
                raise SpeechProviderTimeoutError(
                    "The speech provider did not answer in time.",
                    details={"timeout_seconds": self._settings.tts_timeout_seconds},
                ) from exc

        return Synthesis(audio=audio, voice=alias.id)

    def prepare(
        self, *, text: str, voice: str | None, rate: str | None, volume: str | None
    ) -> tuple[VoiceAlias, SynthesisRequest]:
        """Validate a request completely, before a semaphore slot or a socket is spent."""
        spoken = text.strip()
        if not spoken:
            raise InvalidSpeechRequestError("text must not be empty.", details={"field": "text"})

        limit = self._settings.tts_max_text_chars
        if len(spoken) > limit:
            raise PayloadTooLargeError(
                f"text exceeds the maximum of {limit} characters.",
                details={"field": "text", "max_chars": limit, "chars": len(spoken)},
            )

        alias = self._resolve_voice(voice)

        return alias, SynthesisRequest(
            text=spoken,
            voice=alias.provider_voice,
            rate_percent=self._prosody("rate", rate, self._settings.tts_default_rate),
            volume_percent=self._prosody("volume", volume, self._settings.tts_default_volume),
        )

    def _resolve_voice(self, voice: str | None) -> VoiceAlias:
        name = voice.strip() if voice and voice.strip() else self._settings.tts_default_voice
        aliases = self._settings.voice_alias_map

        if (alias := aliases.get(name)) is None:
            # Provider identifiers are refused like any other unknown name. The
            # contract is the alias; accepting a provider's id "just this once"
            # is how a caller ends up depending on it.
            raise VoiceNotAvailableError(
                "That voice is not available.",
                details={"voice": name[:64], "available": list(aliases)},
            )

        return alias

    @staticmethod
    def _prosody(field: str, value: str | None, default: str) -> int:
        chosen = value.strip() if value and value.strip() else default
        parsed = parse_prosody_percent(chosen)

        if parsed is None:
            low, high = PROSODY_RANGE
            raise InvalidSpeechRequestError(
                f"{field} must look like '+10%' or '-25%', between {low:+d}% and {high:+d}%.",
                details={"field": field, "min": low, "max": high},
            )

        return parsed

    async def _collect(self, request: SynthesisRequest) -> bytes:
        limit = self._settings.tts_max_output_bytes
        parts: list[bytes] = []
        total = 0

        try:
            # `aclosing` so a refusal part-way closes the provider's stream -
            # and its socket - instead of leaving it to the garbage collector.
            async with aclosing(self._engine.stream(request)) as chunks:
                async for chunk in chunks:
                    total += len(chunk)
                    if total > limit:
                        raise TtsOutputTooLargeError(
                            "The speech provider produced more audio than this service returns.",
                            details={"max_bytes": limit},
                        )
                    parts.append(chunk)
        except AppError:
            raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            # The provider's own timeout (separate classes before Python 3.11).
            # To the caller it is the same event as ours, so it is the same 504.
            raise SpeechProviderTimeoutError("The speech provider did not answer in time.") from exc
        except Exception as exc:
            # The provider's exception stays in the log. Its class name would be
            # an invitation to branch on the provider, which callers must not.
            logger.warning("TTS provider %s failed: %r", self._engine.name, exc)
            raise SpeechProviderError(
                "The speech provider failed to synthesise the audio."
            ) from exc

        audio = b"".join(parts)

        if not audio:
            raise SpeechProviderError("The speech provider returned no audio.")

        if not _looks_like_mp3(audio[:4]):
            raise SpeechProviderError("The speech provider returned something other than MP3.")

        return audio


def _looks_like_mp3(head: bytes) -> bool:
    """ID3 tag or an MPEG audio frame sync. A 200 labelled audio/mpeg is only ever that."""
    return head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)
