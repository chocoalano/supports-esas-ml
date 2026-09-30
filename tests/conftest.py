"""Test doubles.

The real engine needs ~300 MB of ONNX weights, so the suite swaps in a fake that
reads everything it needs out of a solid-colour image:

    pixel (0,0) blue  = identity of the main face (0 means "no face here")
    pixel (0,0) green = identity of a second face in the same frame (0 = none)
    pixel (0,0) red   = shot variant, so two photos of one person differ slightly
    pixel (0,1) blue  = pose state index into POSE_STATES (head turn, blink, ...)
    pixel (0,1) green = pose state of the second face

The fake engine then builds *geometrically real* 68-point landmarks for that pose,
so the liveness maths under test is the production one — only the neural net is
replaced.

"Videos" are short text files (``person=1;frames=10;states=neutral|turn_left``)
read by the fake sampler.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import (
    get_challenge_service,
    get_face_engine,
    get_frame_sampler,
    get_speech_telemetry,
    get_stt_engine,
    get_stt_limiter,
    get_tts_engine,
    get_tts_limiter,
)
from app.core.config import RuntimeRole, Settings, get_settings
from app.main import create_app
from app.services.challenge import ChallengeService
from app.services.face_engine import DetectedFace, normalise
from app.services.speech.stt import TranscribeOptions, Transcript, TranscriptSegment
from app.services.speech.tts import SynthesisRequest
from app.services.video import SampledFrame, SampledVideo, evenly_spaced

EMBEDDING_DIM = 128

#: The key the suite's client sends. Any string; it only has to match settings.
API_KEY = "suite-key"

#: name -> (yaw_ratio, pitch_offset, eye_aspect_ratio, mouth_aspect_ratio)
POSE_STATES: dict[str, tuple[float, float, float, float]] = {
    "neutral": (0.0, 0.0, 0.30, 0.10),
    "turn_left": (0.45, 0.0, 0.30, 0.10),
    "turn_right": (-0.45, 0.0, 0.30, 0.10),
    "look_up": (0.0, -0.35, 0.30, 0.10),
    "look_down": (0.0, 0.35, 0.30, 0.10),
    "open_mouth": (0.0, 0.0, 0.30, 0.45),
    "blink": (0.0, 0.0, 0.08, 0.10),
    "slight_turn": (0.12, 0.0, 0.30, 0.10),  # movement, but not enough to pass
}
STATE_ORDER = list(POSE_STATES)

EYE_DISTANCE = 60.0
NEUTRAL_PITCH = 0.55

MAIN_BBOX = (20, 20, 120, 140)
BYSTANDER_BBOX = (130, 10, 250, 190)


def embedding_for(person: int, variant: int = 0) -> np.ndarray:
    """Stable per-person vector; variants stay ~0.99 similar, other people ~0."""
    base = np.random.default_rng(person * 7919).normal(size=EMBEDDING_DIM)
    noise = np.random.default_rng(person * 7919 + variant + 1).normal(size=EMBEDDING_DIM)
    return normalise((normalise(base) + 0.05 * normalise(noise)).astype(np.float32))


def landmarks_for(state: str) -> tuple[np.ndarray, np.ndarray]:
    """Build 5 keypoints and 68 iBUG landmarks matching a pose state."""
    yaw, pitch, eye_ratio, mouth_ratio = POSE_STATES[state]

    right_eye_centre = np.array([100.0, 100.0])
    left_eye_centre = right_eye_centre + np.array([EYE_DISTANCE, 0.0])
    midpoint = (right_eye_centre + left_eye_centre) / 2.0
    nose = midpoint + np.array([yaw * EYE_DISTANCE, (NEUTRAL_PITCH + pitch) * EYE_DISTANCE])

    landmarks = np.zeros((68, 2), dtype=np.float32)
    landmarks[30] = nose

    # Eye corners 15 px apart horizontally -> EAR = 4 * half_height / 60.
    half_height = eye_ratio * 30.0 / 2.0
    for offset, centre in ((36, right_eye_centre), (42, left_eye_centre)):
        landmarks[offset + 0] = centre + [-15.0, 0.0]
        landmarks[offset + 1] = centre + [-7.0, -half_height]
        landmarks[offset + 2] = centre + [7.0, -half_height]
        landmarks[offset + 3] = centre + [15.0, 0.0]
        landmarks[offset + 4] = centre + [7.0, half_height]
        landmarks[offset + 5] = centre + [-7.0, half_height]

    # Inner lips: width 30 -> MAR = 2 * half_open / 30.
    mouth_centre = np.array([130.0, 160.0])
    half_open = mouth_ratio * 30.0 / 2.0
    landmarks[60] = mouth_centre + [-15.0, 0.0]
    landmarks[64] = mouth_centre + [15.0, 0.0]
    for index, dx in ((61, -7.0), (62, 0.0), (63, 7.0)):
        landmarks[index] = mouth_centre + [dx, -half_open]
    for index, dx in ((67, -7.0), (66, 0.0), (65, 7.0)):
        landmarks[index] = mouth_centre + [dx, half_open]

    keypoints = np.array(
        [right_eye_centre, left_eye_centre, nose, landmarks[60], landmarks[64]],
        dtype=np.float32,
    )
    return keypoints, landmarks


def solid_image(
    person: int,
    second_person: int = 0,
    variant: int = 0,
    state: str = "neutral",
    second_state: str = "neutral",
    size: int = 256,
) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = person  # BGR: blue
    image[:, :, 1] = second_person  # green
    image[:, :, 2] = variant % 256  # red
    image[0, 1, 0] = STATE_ORDER.index(state)
    image[0, 1, 1] = STATE_ORDER.index(second_state)
    return image


def png_bytes(
    person: int, second_person: int = 0, variant: int = 0, state: str = "neutral"
) -> bytes:
    ok, buffer = cv2.imencode(".png", solid_image(person, second_person, variant, state))
    assert ok
    return buffer.tobytes()


class FakeFaceEngine:
    name = "fake"

    def __init__(self, *, with_landmarks: bool = True) -> None:
        self.calls = 0
        self.with_landmarks = with_landmarks

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        self.calls += 1
        person = int(image[0, 0, 0])
        second = int(image[0, 0, 1])
        state = STATE_ORDER[int(image[0, 1, 0])]
        second_state = STATE_ORDER[int(image[0, 1, 1])]

        faces = []
        if person:
            faces.append(
                self._face(embedding_for(person, int(image[0, 0, 2])), state, 0.97, MAIN_BBOX)
            )
        if second:
            # Deliberately the bigger face: code that just grabs the largest one
            # would follow the bystander instead of the person being matched.
            faces.append(self._face(embedding_for(second), second_state, 0.81, BYSTANDER_BBOX))

        faces.sort(key=lambda face: face.area, reverse=True)  # as the real engine does
        return faces

    def detect_primary(self, image: np.ndarray) -> DetectedFace | None:
        faces = self.detect(image)
        return faces[0] if faces else None

    def _face(
        self, embedding: np.ndarray, state: str, det_score: float, bbox: tuple[int, int, int, int]
    ) -> DetectedFace:
        keypoints, landmarks = landmarks_for(state)
        return DetectedFace(
            embedding=embedding,
            bbox=bbox,
            det_score=det_score,
            keypoints=keypoints,
            landmarks_68=landmarks if self.with_landmarks else None,
        )


class FakeFrameSampler:
    """Reads `person=1;frames=10;second=0;states=neutral|blink` from the 'video'."""

    def __init__(self) -> None:
        self.requested_max_frames: list[int | None] = []

    def sample(self, path: Path, max_frames: int | None = None) -> SampledVideo:
        self.requested_max_frames.append(max_frames)
        spec = dict(
            part.split("=", 1) for part in path.read_text().strip().split(";") if "=" in part
        )
        person = int(spec.get("person", 1))
        second = int(spec.get("second", 0))
        states = [state for state in spec.get("states", "").split("|") if state]
        if not states:
            states = ["neutral"] * int(spec.get("frames", 10))
        second_states = [state for state in spec.get("second_states", "").split("|") if state]

        frames = [
            SampledFrame(
                index=position * 5,
                timestamp_seconds=round(position * 5 / 30.0, 3),
                image=solid_image(
                    person,
                    second,
                    variant=100 + position,
                    state=state,
                    second_state=(
                        second_states[position] if position < len(second_states) else "neutral"
                    ),
                ),
            )
            for position, state in enumerate(states)
        ]
        if max_frames is not None:
            frames = evenly_spaced(frames, max_frames)

        return SampledVideo(
            frames=frames,
            fps=30.0,
            total_frames=len(states) * 5,
            duration_seconds=len(states) / 6,
        )


def video_payload(
    person: int = 1,
    frames: int = 10,
    second: int = 0,
    states: list[str] | None = None,
    second_states: list[str] | None = None,
) -> bytes:
    spec = f"person={person};frames={frames};second={second}"
    if states:
        spec += ";states=" + "|".join(states)
    if second_states:
        spec += ";second_states=" + "|".join(second_states)
    return spec.encode()


#: Process-wide singletons the speech tests create. Cleared around every test so
#: none inherits another's engine, or a semaphore sized by another's settings.
SPEECH_SINGLETONS = (
    get_stt_engine,
    get_tts_engine,
    get_stt_limiter,
    get_tts_limiter,
    get_speech_telemetry,
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch):
    """Keep a developer's local .env out of the suite and skip model warm-up.

    The runtime role is pinned too: it decides which routers exist, so a local
    `.env` saying `speech` would otherwise turn every Face test into a 404.
    `all` is the development/test role; tests about the other two set their own.
    """
    monkeypatch.setenv("FSA_WARM_UP_ON_STARTUP", "false")
    monkeypatch.setenv("FSA_STT_WARM_UP_ON_STARTUP", "false")
    monkeypatch.setenv("FSA_RUNTIME_ROLE", RuntimeRole.ALL.value)
    get_settings.cache_clear()
    get_challenge_service.cache_clear()
    for singleton in SPEECH_SINGLETONS:
        singleton.cache_clear()
    yield
    get_settings.cache_clear()
    get_challenge_service.cache_clear()
    for singleton in SPEECH_SINGLETONS:
        singleton.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        warm_up_on_startup=False,
        allowed_video_types=["video/webm", "video/mp4", "text/plain"],
        challenge_secret="test-secret-key",
        # The suite runs against a keyed service, so every test proves the guard
        # lets ordinary traffic through as well as what it keeps out. The key
        # travels on the client below; tests/test_access.py sends its own.
        api_keys=[f"test:{API_KEY}"],
        # Off here, or a long test file trips the window and fails somewhere
        # unrelated to what it was checking. The limiter has tests of its own.
        rate_limit_per_minute=0,
    )


@pytest.fixture
def engine() -> FakeFaceEngine:
    return FakeFaceEngine()


@pytest.fixture
def sampler() -> FakeFrameSampler:
    return FakeFrameSampler()


@pytest.fixture
def challenges(settings: Settings) -> ChallengeService:
    return ChallengeService(settings)


@pytest.fixture
def client(
    settings: Settings,
    engine: FakeFaceEngine,
    sampler: FakeFrameSampler,
    challenges: ChallengeService,
):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_face_engine] = lambda: engine
    app.dependency_overrides[get_frame_sampler] = lambda: sampler
    app.dependency_overrides[get_challenge_service] = lambda: challenges
    # Sent on every request, so the existing tests read as what they are about
    # rather than about authentication.
    with TestClient(app, headers={"X-API-Key": API_KEY}) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def build_files(
    *,
    people: list[int] | None = None,
    video_person: int = 1,
    video_frames: int = 10,
    video_second: int = 0,
    video_states: list[str] | None = None,
    video_second_states: list[str] | None = None,
    video_type: str = "video/webm",
) -> list[tuple[str, tuple[str, bytes, str]]]:
    people = people if people is not None else [1] * 5
    files: list[tuple[str, tuple[str, bytes, str]]] = [
        (
            "images",
            (f"ref-{index}.png", png_bytes(person, variant=index + 1), "image/png"),
        )
        for index, person in enumerate(people)
    ]
    files.append(
        (
            "video",
            (
                "clip.webm",
                video_payload(
                    video_person,
                    video_frames,
                    video_second,
                    video_states,
                    video_second_states,
                ),
                video_type,
            ),
        )
    )
    return files


def build_image_files(
    *,
    people: list[int] | None = None,
    capture_person: int = 1,
    capture_second: int = 0,
    capture_variant: int = 99,
    capture_type: str = "image/png",
) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Five reference photos plus one capture, for `/face/verify-image`.

    `capture_variant` differs from every reference variant by default, so the
    capture is the same person photographed on a different occasion rather than
    one of the enrolment files sent back.
    """
    people = people if people is not None else [1] * 5
    files: list[tuple[str, tuple[str, bytes, str]]] = [
        (
            "images",
            (f"ref-{index}.png", png_bytes(person, variant=index + 1), "image/png"),
        )
        for index, person in enumerate(people)
    ]
    files.append(
        (
            "image",
            (
                "capture.png",
                png_bytes(capture_person, capture_second, variant=capture_variant),
                capture_type,
            ),
        )
    )
    return files


def clip_states(*actions: str, padding: int = 3) -> list[str]:
    """Neutral frames, with each action held for two frames, in order."""
    states = ["neutral"] * padding
    for action in actions:
        states += [action, action] + ["neutral"] * padding
    return states


# --- Speech ------------------------------------------------------------------
#
# The real providers are never imported by the suite: faster-whisper would need
# ~500 MB of weights and edge-tts a network. The fakes stand in behind the same
# Protocols. PyAV is real, because the audio validator is code of ours under test.

#: One MPEG-1 Layer III frame header (128 kbps, 44.1 kHz) padded to frame size.
#: Enough for the service's "is this MP3" check, which is all the fake needs.
MP3_FRAME = bytes([0xFF, 0xFB, 0x90, 0x64]) + bytes(413)


class FakeSttEngine:
    name = "fake-stt"

    def __init__(self, transcript: Transcript | None = None) -> None:
        self.transcript = transcript or Transcript(
            text="halo dunia",
            language="id",
            language_probability=0.97,
            segments=(
                TranscriptSegment(start=0.0, end=1.2, text="halo"),
                TranscriptSegment(start=1.2, end=2.0, text="dunia"),
            ),
        )
        self.error: Exception | None = None
        #: Times the "model" was actually loaded - not times `load()` was called.
        self.loads = 0
        self.calls: list[TranscribeOptions] = []
        #: Whether the upload was still on disk while the engine held it.
        self.path_existed: list[bool] = []

    def is_loaded(self) -> bool:
        return self.loads > 0

    def load(self) -> None:
        if not self.loads:
            self.loads += 1

    def transcribe(self, path: Path, options: TranscribeOptions) -> Transcript:
        self.load()
        self.calls.append(options)
        self.path_existed.append(path.exists())
        if self.error is not None:
            raise self.error
        return self.transcript


class FakeTtsEngine:
    """Yields `chunks`; can fail before chunk `fail_at`, or dawdle for `delay` seconds."""

    name = "fake-tts"

    def __init__(
        self,
        chunks: list[bytes] | None = None,
        *,
        error: Exception | None = None,
        fail_at: int = 0,
        delay: float = 0.0,
    ) -> None:
        self.chunks = chunks if chunks is not None else [MP3_FRAME, MP3_FRAME]
        self.error = error
        self.fail_at = fail_at
        self.delay = delay
        self.requests: list[SynthesisRequest] = []
        self.active = 0
        self.peak = 0
        #: Streams that ran their cleanup - i.e. were closed, not abandoned.
        self.closed = 0

    async def stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        self.requests.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            for index, chunk in enumerate(self.chunks):
                if self.error is not None and index == self.fail_at:
                    raise self.error
                if self.delay:
                    await anyio.sleep(self.delay)
                yield chunk
            if self.error is not None and self.fail_at >= len(self.chunks):
                raise self.error
        finally:
            self.active -= 1
            self.closed += 1


@pytest.fixture
def speech_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={"runtime_role": RuntimeRole.ALL, "stt_enabled": True, "tts_enabled": True}
    )


@pytest.fixture
def stt_engine() -> FakeSttEngine:
    return FakeSttEngine()


@pytest.fixture
def tts_engine() -> FakeTtsEngine:
    return FakeTtsEngine()


def make_speech_client(
    settings: Settings,
    stt: FakeSttEngine,
    tts: FakeTtsEngine,
    *,
    api_key: str | None = API_KEY,
) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_stt_engine] = lambda: stt
    app.dependency_overrides[get_tts_engine] = lambda: tts
    return TestClient(app, headers={"X-API-Key": api_key} if api_key else {})


@pytest.fixture
def speech_client(
    speech_settings: Settings, stt_engine: FakeSttEngine, tts_engine: FakeTtsEngine
):
    with make_speech_client(speech_settings, stt_engine, tts_engine) as test_client:
        yield test_client
