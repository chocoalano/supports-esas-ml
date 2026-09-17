"""End-to-end tests for POST /api/v1/face/verify (real HTTP layer, fake engine)."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from tests.conftest import build_files, png_bytes, video_payload

ENDPOINT = "/api/v1/face/verify"


def test_health_reports_the_engine(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "model": "fake",
        "model_loaded": True,
    }


def test_same_person_in_video_matches(client):
    response = client.post(ENDPOINT, files=build_files(video_person=1))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["passed"] is True
    assert body["match"] is True
    assert body["decision"] == "match"
    assert body["liveness"] is None  # no challenge token was sent
    assert body["score"] > body["thresholds"]["match_threshold"]
    assert body["score_percent"] > 90
    assert body["confidence"] > 0.9
    assert body["reference"]["accepted"] == 5
    assert body["reference"]["consistent"] is True
    assert body["video"]["frames_sampled"] == 10
    assert body["video"]["frames_with_face"] == 10
    assert body["video"]["match_ratio"] == 1.0
    assert body["warnings"] == []


def test_different_person_in_video_does_not_match(client):
    response = client.post(ENDPOINT, files=build_files(video_person=2))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["passed"] is False
    assert body["match"] is False
    assert body["decision"] == "no_match"
    assert body["score"] < body["thresholds"]["match_threshold"]
    assert body["video"]["frames_matched"] == 0


def test_response_reports_every_frame_and_reference(client):
    body = client.post(ENDPOINT, files=build_files()).json()

    assert len(body["video"]["frames"]) == 10
    assert body["video"]["best_frame"]["face_detected"] is True
    assert {frame["frame_index"] for frame in body["video"]["frames"]} == {
        i * 5 for i in range(10)
    }

    images = body["reference"]["images"]
    assert len(images) == 5
    assert all(image["face_detected"] for image in images)
    assert all(image["best_video_similarity"] > 0.9 for image in images)
    assert all(image["bbox"] == {"x": 20, "y": 20, "width": 100, "height": 120} for image in images)


def test_no_face_in_any_video_frame_is_inconclusive(client):
    response = client.post(ENDPOINT, files=build_files(video_person=0))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == "inconclusive"
    assert body["match"] is False
    assert body["video"]["frames_with_face"] == 0
    assert body["video"]["match_ratio"] == 0.0
    assert any("contained a face" in warning for warning in body["warnings"])


def test_bystander_in_frame_does_not_sink_the_score(client):
    """The frame holds the target plus a stranger; the closest face must win."""
    response = client.post(ENDPOINT, files=build_files(video_person=1, video_second=9))

    body = response.json()
    assert body["decision"] == "match"
    assert any("more than one face" in w.lower() for w in body["warnings"])


def test_mixed_reference_photos_flag_inconsistency(client):
    response = client.post(ENDPOINT, files=build_files(people=[1, 1, 1, 2, 3]))

    body = response.json()
    assert body["reference"]["consistent"] is False
    assert body["reference"]["cohesion"] < 0.35
    assert any("same person" in warning for warning in body["warnings"])


def test_reference_photo_without_a_face_is_rejected(client):
    """Four usable photos out of five is below the configured minimum.

    Its own code, not `no_face_detected`. The caller has to tell an unreadable
    *enrolment* apart from an unreadable *capture*, because only one of the two
    is something the person standing at the camera can do anything about.
    """
    response = client.post(ENDPOINT, files=build_files(people=[1, 1, 1, 1, 0]))

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "reference_faces_unusable"
    assert error["details"]["accepted"] == 4
    assert error["details"]["required"] == 5
    assert error["details"]["images"][4]["skip_reason"] == "no_face_detected"


def test_wrong_number_of_images_is_rejected(client):
    response = client.post(ENDPOINT, files=build_files(people=[1, 1, 1]))

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_upload"
    assert error["details"] == {"received": 3, "min": 5, "max": 5}


def test_video_content_type_is_validated(client):
    response = client.post(ENDPOINT, files=build_files(video_type="application/zip"))

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_oversized_image_is_rejected(client, settings):
    settings.max_image_bytes = 128

    response = client.post(ENDPOINT, files=build_files())

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_non_image_payload_is_rejected(client):
    files = [("images", (f"ref-{i}.png", png_bytes(1, variant=i), "image/png")) for i in range(4)]
    files.append(("images", ("broken.png", b"definitely-not-an-image", "image/png")))
    files.append(("video", ("clip.webm", video_payload(), "video/webm")))

    response = client.post(ENDPOINT, files=files)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_upload"


def test_thresholds_are_configurable(client, settings):
    """Raising the bar above a perfect-ish score flips the verdict."""
    settings.match_threshold = 0.999
    settings.frame_match_threshold = 0.999
    settings.inconclusive_band = 0.0

    body = client.post(ENDPOINT, files=build_files(video_person=1)).json()

    assert body["decision"] == "no_match"
    assert body["thresholds"]["match_threshold"] == 0.999


def test_the_scored_frame_is_returned_only_when_asked_for(client):
    """The picture worth keeping is the one the score was measured on.

    A caller filing a photograph beside an attendance record used to have to take
    its own, on the handset, before recording - an image that had never been
    compared with anything. This is the frame the similarity actually came from.
    """
    without = client.post("/api/v1/face/verify", files=build_files())
    assert without.status_code == 200
    assert without.json()["video"]["best_frame"]["image_base64"] is None

    with_frame = client.post(
        "/api/v1/face/verify",
        files=build_files(),
        data={"include_best_frame": "true"},
    )
    assert with_frame.status_code == 200

    encoded = with_frame.json()["video"]["best_frame"]["image_base64"]
    assert isinstance(encoded, str) and encoded

    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None, "the frame has to come back as a real JPEG"
    assert decoded.shape[2] == 3


def test_the_scored_frame_is_not_repeated_in_the_frame_list(client):
    """`best_frame` is one of the per-frame reports.

    Setting the image on it in place would repeat a base64 JPEG inside `frames`
    as well, for every response, which is a lot of bytes to send twice.
    """
    response = client.post(
        "/api/v1/face/verify",
        files=build_files(),
        data={"include_best_frame": "true"},
    )

    video = response.json()["video"]

    assert video["best_frame"]["image_base64"] is not None
    assert all(frame["image_base64"] is None for frame in video["frames"])


def test_a_clip_with_no_face_returns_no_frame_to_keep(client):
    response = client.post(
        "/api/v1/face/verify",
        files=build_files(video_person=0),
        data={"include_best_frame": "true"},
    )

    assert response.status_code == 200
    assert response.json()["video"]["best_frame"] is None
