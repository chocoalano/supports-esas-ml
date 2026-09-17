"""Liveness detection: metrics extraction, the analyzer, and both endpoints."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.challenge import ChallengeService
from app.services.liveness import FaceMetrics, LivenessAnalyzer, metrics_from_face
from app.services.video import SampledFrame
from tests.conftest import (
    NEUTRAL_PITCH,
    POSE_STATES,
    FakeFaceEngine,
    build_files,
    clip_states,
    landmarks_for,
    solid_image,
    video_payload,
)

CHALLENGE_URL = "/api/v1/liveness/challenge"
LIVENESS_URL = "/api/v1/liveness/verify"
VERIFY_URL = "/api/v1/face/verify"


def metrics_of(state: str, engine: FakeFaceEngine, index: int = 0) -> FaceMetrics:
    frame = SampledFrame(
        index=index, timestamp_seconds=index / 10.0, image=solid_image(1, state=state)
    )
    face = engine.detect(frame.image)[0]
    result = metrics_from_face(face, frame)
    assert result is not None
    return result


# --- metrics ----------------------------------------------------------------


@pytest.mark.parametrize("state", list(POSE_STATES))
def test_metrics_match_the_pose_they_were_built_from(state: str, engine: FakeFaceEngine):
    yaw, pitch, eye_ratio, mouth_ratio = POSE_STATES[state]

    metrics = metrics_of(state, engine)

    assert metrics.yaw_ratio == pytest.approx(yaw, abs=0.01)
    assert metrics.pitch_ratio == pytest.approx(NEUTRAL_PITCH + pitch, abs=0.01)
    assert metrics.eye_ratio == pytest.approx(eye_ratio, abs=0.01)
    assert metrics.mouth_ratio == pytest.approx(mouth_ratio, abs=0.01)


def test_turning_to_the_persons_left_moves_the_nose_right_in_the_image():
    """Sign convention: image x grows to the right, the person's left is that way."""
    _, neutral = landmarks_for("neutral")
    _, turned = landmarks_for("turn_left")

    assert turned[30][0] > neutral[30][0]


def test_metrics_fall_back_to_the_five_keypoints_without_landmarks():
    engine = FakeFaceEngine(with_landmarks=False)

    metrics = metrics_of("turn_right", engine)

    assert metrics.yaw_ratio == pytest.approx(-0.45, abs=0.01)
    assert metrics.eye_ratio is None  # blink/mouth need the 68-point model
    assert metrics.mouth_ratio is None


# --- analyzer ---------------------------------------------------------------


def series(states: list[str], engine: FakeFaceEngine) -> list[FaceMetrics]:
    return [metrics_of(state, engine, index=position) for position, state in enumerate(states)]


@pytest.mark.parametrize(
    "action", ["turn_left", "turn_right", "look_up", "look_down", "open_mouth", "blink"]
)
def test_each_action_is_detected(action: str, engine: FakeFaceEngine, settings: Settings):
    metrics = series(clip_states(action), engine)

    outcome = LivenessAnalyzer(settings).analyze(metrics, [action], frames_sampled=len(metrics))

    assert outcome.live is True
    assert outcome.actions[0].performed is True
    assert outcome.actions[0].detected_at_seconds is not None


def test_a_still_clip_fails_every_action(engine: FakeFaceEngine, settings: Settings):
    metrics = series(["neutral"] * 12, engine)

    outcome = LivenessAnalyzer(settings).analyze(
        metrics, ["turn_left", "blink"], frames_sampled=12
    )

    assert outcome.live is False
    assert [item.reason for item in outcome.actions] == ["not_detected", "not_detected"]
    assert outcome.movement_score == 0.0
    assert any("photo held up" in warning for warning in outcome.warnings)


def test_small_movements_do_not_count_as_a_turn(engine: FakeFaceEngine, settings: Settings):
    metrics = series(clip_states("slight_turn"), engine)

    outcome = LivenessAnalyzer(settings).analyze(
        metrics, ["turn_left"], frames_sampled=len(metrics)
    )

    assert outcome.live is False
    assert outcome.actions[0].measured == pytest.approx(0.12, abs=0.01)


def test_actions_must_happen_in_the_order_they_were_asked(
    engine: FakeFaceEngine, settings: Settings
):
    metrics = series(clip_states("open_mouth", "turn_left"), engine)
    analyzer = LivenessAnalyzer(settings)

    assert analyzer.analyze(
        metrics, ["open_mouth", "turn_left"], frames_sampled=len(metrics)
    ).live is True
    assert analyzer.analyze(
        metrics, ["turn_left", "open_mouth"], frames_sampled=len(metrics)
    ).live is False


def test_too_few_frames_with_a_face_is_reported_not_live(
    engine: FakeFaceEngine, settings: Settings
):
    metrics = series(["turn_left", "neutral"], engine)

    outcome = LivenessAnalyzer(settings).analyze(metrics, ["turn_left"], frames_sampled=20)

    assert outcome.live is False
    assert outcome.actions[0].reason == "not_enough_frames"
    assert any("minimum" in warning for warning in outcome.warnings)


def test_blink_needs_the_landmark_model(settings: Settings):
    engine = FakeFaceEngine(with_landmarks=False)
    metrics = series(clip_states("blink"), engine)

    outcome = LivenessAnalyzer(settings).analyze(metrics, ["blink"], frames_sampled=len(metrics))

    assert outcome.live is False
    assert outcome.actions[0].reason == "landmarks_unavailable"
    assert any("FSA_ENABLE_LANDMARKS" in warning for warning in outcome.warnings)


def test_thresholds_are_configurable(engine: FakeFaceEngine, settings: Settings):
    metrics = series(clip_states("slight_turn"), engine)
    settings.liveness_yaw_delta = 0.1

    outcome = LivenessAnalyzer(settings).analyze(
        metrics, ["turn_left"], frames_sampled=len(metrics)
    )

    assert outcome.live is True


# --- HTTP layer -------------------------------------------------------------


def liveness_files(actions: list[str], *, person: int = 1):
    payload = video_payload(person, states=clip_states(*actions))
    return {"video": ("clip.webm", payload, "video/webm")}


def test_liveness_endpoint_accepts_a_correct_recording(client, challenges: ChallengeService):
    challenge, token = challenges.issue(action_count=2)

    response = client.post(
        LIVENESS_URL, data={"token": token}, files=liveness_files(challenge.actions)
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["live"] is True
    assert body["challenge_id"] == challenge.challenge_id
    assert [item["action"] for item in body["actions"]] == challenge.actions
    assert body["frames_with_face"] == body["frames_sampled"]
    assert body["movement_score"] > 0
    assert body["processing_ms"] >= 0


def test_liveness_endpoint_rejects_the_wrong_actions(client, challenges: ChallengeService):
    challenge, token = challenges.issue(action_count=1)
    wrong = "turn_right" if challenge.actions[0] == "turn_left" else "turn_left"

    response = client.post(LIVENESS_URL, data={"token": token}, files=liveness_files([wrong]))

    assert response.status_code == 200, response.text
    assert response.json()["live"] is False


def test_liveness_endpoint_rejects_a_forged_token(client):
    response = client.post(
        LIVENESS_URL, data={"token": "abc.def"}, files=liveness_files(["turn_left"])
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_challenge"


def test_liveness_endpoint_rejects_an_expired_token(client, settings: Settings):
    settings.challenge_ttl_seconds = -1
    token = client.post(CHALLENGE_URL).json()["token"]

    response = client.post(
        LIVENESS_URL, data={"token": token}, files=liveness_files(["turn_left"])
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "challenge_expired"


def test_liveness_uses_the_denser_frame_budget(client, challenges, sampler, settings: Settings):
    _, token = challenges.issue(action_count=1)

    client.post(LIVENESS_URL, data={"token": token}, files=liveness_files(["turn_left"]))

    assert sampler.requested_max_frames == [settings.liveness_max_sampled_frames]


# --- combined with face similarity ------------------------------------------


def test_face_verify_accepts_a_challenge_token(client, challenges: ChallengeService):
    challenge, token = challenges.issue(action_count=2)
    files = build_files(video_states=clip_states(*challenge.actions))

    response = client.post(VERIFY_URL, files=files, data={"challenge_token": token})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["passed"] is True
    assert body["match"] is True
    assert body["liveness"]["live"] is True
    assert body["liveness"]["challenge_id"] == challenge.challenge_id


def test_face_verify_fails_when_the_face_matches_but_liveness_does_not(
    client, challenges: ChallengeService
):
    """A photo of the right person held to the camera: match, but not alive."""
    challenge, token = challenges.issue(action_count=1)
    files = build_files(video_states=["neutral"] * 12)

    response = client.post(VERIFY_URL, files=files, data={"challenge_token": token})

    body = response.json()
    assert body["match"] is True
    assert body["passed"] is False
    assert body["liveness"]["live"] is False
    assert any("Liveness challenge failed" in warning for warning in body["warnings"])


def test_liveness_follows_the_matched_face_not_the_bystander(
    client, challenges: ChallengeService
):
    """The bystander performs the actions while the matched person sits still."""
    challenge, token = challenges.issue(action_count=1)
    performed = clip_states(*challenge.actions)
    files = build_files(
        video_second=9,
        video_states=["neutral"] * len(performed),
        video_second_states=performed,
    )

    body = client.post(VERIFY_URL, files=files, data={"challenge_token": token}).json()

    assert body["match"] is True  # the right person is in the frame...
    assert body["liveness"]["live"] is False  # ...but they are the one who did nothing
    assert body["passed"] is False


def test_liveness_passes_when_the_matched_person_performs_the_actions(
    client, challenges: ChallengeService
):
    challenge, token = challenges.issue(action_count=1)
    files = build_files(video_second=9, video_states=clip_states(*challenge.actions))

    body = client.post(VERIFY_URL, files=files, data={"challenge_token": token}).json()

    assert body["liveness"]["live"] is True
    assert body["passed"] is True


def test_face_verify_rejects_a_bad_challenge_token(client):
    response = client.post(
        VERIFY_URL, files=build_files(), data={"challenge_token": "garbage"}
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_challenge"
