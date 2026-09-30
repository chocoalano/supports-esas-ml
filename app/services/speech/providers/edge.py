"""`TextToSpeechEngine` backed by Microsoft Edge's online voices (`edge-tts`).

A thin stream and nothing else. Timeout, byte ceiling, empty and non-MP3
output are all enforced by `TextToSpeechService`, so they hold for whichever
provider comes next.

What reaches the provider's SSML is safe by construction: the text is escaped
by edge-tts itself, the voice comes from configuration, and rate and volume
arrive here as integers and are formatted here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing

from app.core.config import Settings
from app.core.errors import SpeechEngineUnavailableError
from app.services.speech.tts import SynthesisRequest


class EdgeTtsEngine:
    name = "edge"

    def __init__(self, settings: Settings) -> None:
        # edge-tts's own socket timeouts, at half the service's budget. A
        # provider that stalls is then noticed by the library itself, which
        # unwinds and closes its websocket properly. The service's wall-clock
        # timeout remains the outer bound, but it works by cancelling, and a
        # cancelled stream cannot finish closing its socket - so it is the
        # backstop for a provider that trickles, not the usual path.
        self._socket_timeout = max(1, int(settings.tts_timeout_seconds / 2))

    async def stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        try:
            import edge_tts
        except ImportError as exc:
            raise SpeechEngineUnavailableError(
                "edge-tts is not installed. Run `pip install -r requirements.txt`.",
                details={"import_error": str(exc)},
            ) from exc

        communicate = edge_tts.Communicate(
            request.text,
            request.voice,
            rate=f"{request.rate_percent:+d}%",
            volume=f"{request.volume_percent:+d}%",
            connect_timeout=self._socket_timeout,
            receive_timeout=self._socket_timeout,
        )

        # Closed explicitly. When the service stops reading - output ceiling
        # reached - it closes this generator, and without `aclosing` the
        # edge-tts generator inside it would be dropped mid-stream, leaving its
        # websocket and TLS socket for the garbage collector to find.
        async with aclosing(communicate.stream()) as chunks:
            async for chunk in chunks:
                # Word and sentence boundaries arrive on the same stream; only
                # the audio is ours to return.
                if chunk["type"] == "audio" and chunk.get("data"):
                    yield chunk["data"]
