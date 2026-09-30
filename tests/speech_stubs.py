"""Stand-ins for the provider libraries, installed into `sys.modules`.

They let the suite run the *real* provider classes - lazy import, locked load,
chunk filtering, rate formatting - without half a gigabyte of weights or a
connection to Microsoft. Each records what was asked of it.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from dataclasses import dataclass, field

import pytest


@dataclass
class WhisperRecord:
    constructed: list[tuple[str, dict]] = field(default_factory=list)
    transcribed: list[dict] = field(default_factory=list)


def install_faster_whisper(
    monkeypatch: pytest.MonkeyPatch, *, load_seconds: float = 0.0, text: str = " halo dunia "
) -> WhisperRecord:
    record = WhisperRecord()
    lock = threading.Lock()

    class WhisperModel:
        supported_languages = ["en", "id", "ms"]

        def __init__(self, model_size_or_path: str, **kwargs) -> None:
            if load_seconds:
                time.sleep(load_seconds)  # widen the window a racing load would use
            with lock:
                record.constructed.append((model_size_or_path, kwargs))

        def transcribe(self, audio: str, **kwargs):
            record.transcribed.append({"audio": audio, **kwargs})
            segment = types.SimpleNamespace(start=0.0, end=1.5, text=text)
            info = types.SimpleNamespace(
                language=kwargs.get("language") or "id", language_probability=0.91234
            )
            return iter([segment]), info

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    return record


@dataclass
class EdgeRecord:
    communicated: list[dict] = field(default_factory=list)


def install_edge_tts(
    monkeypatch: pytest.MonkeyPatch, *, chunks: list[dict] | None = None
) -> EdgeRecord:
    record = EdgeRecord()
    stream_chunks = chunks if chunks is not None else []

    class Communicate:
        def __init__(self, text: str, voice: str, **kwargs) -> None:
            record.communicated.append({"text": text, "voice": voice, **kwargs})

        async def stream(self):
            for chunk in stream_chunks:
                yield chunk

    module = types.ModuleType("edge_tts")
    module.Communicate = Communicate
    monkeypatch.setitem(sys.modules, "edge_tts", module)

    return record


def block_imports(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make `import name` raise ImportError, as if the package were not installed."""
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)
