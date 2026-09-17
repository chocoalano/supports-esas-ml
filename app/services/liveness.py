"""Detecting the challenge actions in a recorded clip.

Everything is measured **relative to the person's own neutral pose** (the median
across the clip) rather than against absolute angles: face shapes and camera
placement vary far too much for fixed thresholds to hold.

Geometry, all in image coordinates (y grows downward):

* ``yaw_ratio``   = (nose.x - eye_midpoint.x) / eye_distance.
  Turning the head towards the person's own left moves the nose tip towards the
  right-hand side of the image, so a turn to their left is a *positive* delta.
* ``pitch_ratio`` = (nose.y - eye_midpoint.y) / eye_distance.
  Lifting the chin shortens the eye-to-nose drop, so looking up is a *negative*
  delta.
* ``eye_ratio``   = eye aspect ratio; collapses towards 0 when the eyes close.
* ``mouth_ratio`` = inner-lip opening / mouth width.

The last two need the 68-point landmark model (`FSA_ENABLE_LANDMARKS=true`).
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from anyio import Semaphore, to_thread

from app.core.config import Settings
from app.services.challenge import ACTION_CATALOGUE, Challenge
from app.services.face_engine import DetectedFace, FaceEngine
from app.services.video import FrameSampler, SampledFrame

logger = logging.getLogger(__name__)

# Standard iBUG 68-point groups.
RIGHT_EYE = (36, 37, 38, 39, 40, 41)
LEFT_EYE = (42, 43, 44, 45, 46, 47)
NOSE_TIP = 30
INNER_LIPS = (60, 61, 62, 63, 64, 65, 66, 67)


@dataclass(frozen=True)
class FaceMetrics:
    frame_index: int
    timestamp_seconds: float | None
    yaw_ratio: float
    pitch_ratio: float
    eye_ratio: float | None
    mouth_ratio: float | None


@dataclass
class ActionOutcome:
    action: str
    instruction: str
    performed: bool
    detected_at_seconds: float | None = None
    frame_index: int | None = None
    measured: float | None = None
    baseline: float | None = None
    reason: str | None = None


@dataclass
class LivenessOutcome:
    live: bool
    actions: list[ActionOutcome]
    frames_sampled: int
    frames_with_face: int
    movement_score: float
    warnings: list[str]
    challenge_id: str | None = None


def metrics_from_face(face: DetectedFace, frame: SampledFrame) -> FaceMetrics | None:
    """Reduce one detected face to the four numbers the checks care about."""
    landmarks = face.landmarks_68
    keypoints = face.keypoints

    if landmarks is not None and len(landmarks) >= 68:
        right_eye = landmarks[list(RIGHT_EYE)]
        left_eye = landmarks[list(LEFT_EYE)]
        eye_centres = (right_eye.mean(axis=0), left_eye.mean(axis=0))
        nose = landmarks[NOSE_TIP]
        eye_ratio = (_eye_aspect_ratio(right_eye) + _eye_aspect_ratio(left_eye)) / 2.0
        mouth_ratio = _mouth_aspect_ratio(landmarks)
    elif keypoints is not None and len(keypoints) >= 3:
        eye_centres = (keypoints[0], keypoints[1])
        nose = keypoints[2]
        eye_ratio = None
        mouth_ratio = None
    else:
        return None

    eye_distance = float(np.linalg.norm(eye_centres[1] - eye_centres[0]))
    if eye_distance < 1e-3:
        return None

    midpoint = (eye_centres[0] + eye_centres[1]) / 2.0
    return FaceMetrics(
        frame_index=frame.index,
        timestamp_seconds=frame.timestamp_seconds,
        yaw_ratio=float((nose[0] - midpoint[0]) / eye_distance),
        pitch_ratio=float((nose[1] - midpoint[1]) / eye_distance),
        eye_ratio=eye_ratio,
        mouth_ratio=mouth_ratio,
    )


def _eye_aspect_ratio(eye: np.ndarray) -> float:
    width = float(np.linalg.norm(eye[3] - eye[0]))
    if width < 1e-6:
        return 0.0
    height = float(np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4]))
    return height / (2.0 * width)


def _mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    width = float(np.linalg.norm(landmarks[64] - landmarks[60]))
    if width < 1e-6:
        return 0.0
    opening = (
        float(np.linalg.norm(landmarks[61] - landmarks[67]))
        + float(np.linalg.norm(landmarks[62] - landmarks[66]))
        + float(np.linalg.norm(landmarks[63] - landmarks[65]))
    )
    return opening / (3.0 * width)


class LivenessAnalyzer:
    """Turns a series of `FaceMetrics` into a per-action verdict."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(
        self,
        metrics: list[FaceMetrics],
        actions: list[str],
        *,
        frames_sampled: int,
        challenge_id: str | None = None,
    ) -> LivenessOutcome:
        warnings: list[str] = []
        outcomes: list[ActionOutcome] = []

        if len(metrics) < self._settings.liveness_min_frames_with_face:
            warnings.append(
                f"Only {len(metrics)} of {frames_sampled} sampled frames contained a usable "
                f"face (minimum {self._settings.liveness_min_frames_with_face}). Ask for a "
                "longer, better-lit recording."
            )
            return LivenessOutcome(
                live=False,
                actions=[
                    ActionOutcome(
                        action=action,
                        instruction=ACTION_CATALOGUE.get(action, ""),
                        performed=False,
                        reason="not_enough_frames",
                    )
                    for action in actions
                ],
                frames_sampled=frames_sampled,
                frames_with_face=len(metrics),
                movement_score=0.0,
                warnings=warnings,
                challenge_id=challenge_id,
            )

        baselines = self._baselines(metrics)
        movement = self._movement_score(metrics)
        if movement < 0.01:
            warnings.append(
                "The face barely moves across the clip, which is what a photo held up to "
                "the camera looks like."
            )

        # Scanning forward from the previous hit is what enforces the order.
        cursor = 0
        for action in actions:
            outcome = self._find(action, metrics, baselines, start=cursor)
            outcomes.append(outcome)
            if outcome.performed and outcome.frame_index is not None:
                cursor = next(
                    (
                        position
                        for position, item in enumerate(metrics)
                        if item.frame_index == outcome.frame_index
                    ),
                    cursor,
                )
                cursor += 1

        missing_landmarks = [item.reason == "landmarks_unavailable" for item in outcomes]
        if any(missing_landmarks):
            warnings.append(
                "Blink and mouth checks need the 68-point landmark model; enable "
                "FSA_ENABLE_LANDMARKS."
            )

        return LivenessOutcome(
            live=all(outcome.performed for outcome in outcomes) and bool(outcomes),
            actions=outcomes,
            frames_sampled=frames_sampled,
            frames_with_face=len(metrics),
            movement_score=round(movement, 4),
            warnings=warnings,
            challenge_id=challenge_id,
        )

    def _baselines(self, metrics: list[FaceMetrics]) -> dict[str, float | None]:
        def median_of(values: list[float]) -> float | None:
            return statistics.median(values) if values else None

        return {
            "yaw": statistics.median([item.yaw_ratio for item in metrics]),
            "pitch": statistics.median([item.pitch_ratio for item in metrics]),
            "eye": median_of([item.eye_ratio for item in metrics if item.eye_ratio is not None]),
            "mouth": median_of(
                [item.mouth_ratio for item in metrics if item.mouth_ratio is not None]
            ),
        }

    def _movement_score(self, metrics: list[FaceMetrics]) -> float:
        if len(metrics) < 2:
            return 0.0
        yaw = statistics.pstdev([item.yaw_ratio for item in metrics])
        pitch = statistics.pstdev([item.pitch_ratio for item in metrics])
        return float(yaw + pitch)

    def _find(
        self,
        action: str,
        metrics: list[FaceMetrics],
        baselines: dict[str, float | None],
        *,
        start: int,
    ) -> ActionOutcome:
        instruction = ACTION_CATALOGUE.get(action, "")
        settings = self._settings

        if action in {"turn_left", "turn_right"}:
            baseline = baselines["yaw"]
            sign = 1.0 if action == "turn_left" else -1.0
            return self._scan(
                action,
                instruction,
                metrics[start:],
                baseline=baseline,
                value=lambda item: item.yaw_ratio,
                satisfied=lambda value: sign * (value - baseline) >= settings.liveness_yaw_delta,
            )

        if action in {"look_up", "look_down"}:
            baseline = baselines["pitch"]
            sign = -1.0 if action == "look_up" else 1.0
            return self._scan(
                action,
                instruction,
                metrics[start:],
                baseline=baseline,
                value=lambda item: item.pitch_ratio,
                satisfied=lambda value: sign * (value - baseline) >= settings.liveness_pitch_delta,
            )

        if action == "open_mouth":
            baseline = baselines["mouth"]
            if baseline is None:
                return ActionOutcome(action, instruction, False, reason="landmarks_unavailable")
            return self._scan(
                action,
                instruction,
                metrics[start:],
                baseline=baseline,
                value=lambda item: item.mouth_ratio,
                satisfied=lambda value: value - baseline >= settings.liveness_mouth_delta,
            )

        if action == "blink":
            baseline = baselines["eye"]
            if baseline is None:
                return ActionOutcome(action, instruction, False, reason="landmarks_unavailable")
            return self._scan(
                action,
                instruction,
                metrics[start:],
                baseline=baseline,
                value=lambda item: item.eye_ratio,
                satisfied=lambda value: value <= baseline * settings.liveness_blink_ratio,
            )

        return ActionOutcome(action, instruction, False, reason="unsupported_action")

    def _scan(
        self,
        action: str,
        instruction: str,
        metrics: list[FaceMetrics],
        *,
        baseline: float | None,
        value,
        satisfied,
    ) -> ActionOutcome:
        best: tuple[float, FaceMetrics] | None = None
        for item in metrics:
            current = value(item)
            if current is None:
                continue
            if satisfied(current):
                return ActionOutcome(
                    action=action,
                    instruction=instruction,
                    performed=True,
                    detected_at_seconds=item.timestamp_seconds,
                    frame_index=item.frame_index,
                    measured=round(current, 4),
                    baseline=round(baseline, 4) if baseline is not None else None,
                )
            if best is None or abs(current - (baseline or 0.0)) > abs(best[0] - (baseline or 0.0)):
                best = (current, item)

        return ActionOutcome(
            action=action,
            instruction=instruction,
            performed=False,
            measured=round(best[0], 4) if best else None,
            baseline=round(baseline, 4) if baseline is not None else None,
            reason="not_detected",
        )


class LivenessService:
    """Standalone liveness check: no reference photos involved."""

    def __init__(
        self,
        *,
        engine: FaceEngine,
        sampler: FrameSampler,
        settings: Settings,
        limiter: Semaphore | None = None,
    ) -> None:
        self._engine = engine
        self._sampler = sampler
        self._settings = settings
        self._analyzer = LivenessAnalyzer(settings)
        self._limiter = limiter or Semaphore(max(1, settings.max_concurrent_inferences))

    async def verify(self, challenge: Challenge, video_path: Path) -> LivenessOutcome:
        async with self._limiter:
            return await to_thread.run_sync(self._run, challenge, video_path)

    def _run(self, challenge: Challenge, video_path: Path) -> LivenessOutcome:
        sampled = self._sampler.sample(
            video_path, max_frames=self._settings.liveness_max_sampled_frames
        )
        metrics: list[FaceMetrics] = []
        for frame in sampled.frames:
            faces = self._engine.detect(frame.image)
            if not faces:
                continue
            entry = metrics_from_face(faces[0], frame)
            if entry is not None:
                metrics.append(entry)

        return self._analyzer.analyze(
            metrics,
            challenge.actions,
            frames_sampled=len(sampled.frames),
            challenge_id=challenge.challenge_id,
        )
