"""`POST /face/verify-image` — five reference photos against one still.

The endpoint the attendance flow uses once liveness has been decided on the
handset. What it must get right is narrow and worth stating: it scores a
photograph, it says so, and it never implies anybody was alive when it did.
"""

from __future__ import annotations

from tests.conftest import build_image_files, png_bytes


def post(client, files, data=None):
    return client.post("/api/v1/face/verify-image", files=files, data=data or {})


def test_matches_the_enrolled_person(client) -> None:
    response = post(client, build_image_files())

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "match"
    assert body["passed"] is True
    assert body["match"] is True
    assert body["score"] > body["thresholds"]["match_threshold"]
    assert body["capture"]["face_detected"] is True
    assert body["capture"]["matched_references"] == 5
    assert body["reference"]["accepted"] == 5


def test_refuses_a_different_person(client) -> None:
    response = post(client, build_image_files(capture_person=2))

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "no_match"
    assert body["passed"] is False
    assert body["capture"]["matched_references"] == 0


def test_says_nothing_about_liveness(client) -> None:
    """A still cannot carry one, so the field is absent rather than false.

    A caller that read `liveness.live` off this response would be reading a
    field this endpoint never sets - and a `false` there would be worse than
    nothing, because it looks like a measurement.
    """
    body = post(client, build_image_files()).json()

    assert "liveness" not in body
    assert "video" not in body


def test_a_capture_with_no_face_is_refused_not_scored(client) -> None:
    """422, not `no_match`.

    Nothing was compared. Telling somebody "that is not you" about a picture
    with no face in it sends them away believing the wrong thing; telling them
    no face was found sends them one step closer to the camera.
    """
    response = post(client, build_image_files(capture_person=0))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_face_detected"


def test_the_closest_face_is_scored_not_the_largest(client) -> None:
    """A bystander stands nearer the lens and fills more of the frame."""
    response = post(client, build_image_files(capture_person=1, capture_second=2))

    body = response.json()
    assert body["decision"] == "match"
    assert body["capture"]["faces_found"] == 2
    assert any("faces appeared" in warning for warning in body["warnings"])


def test_a_short_reference_set_is_refused(client) -> None:
    files = build_image_files()
    files = [item for item in files if item[0] != "images"][:1]
    files = [
        ("images", ("ref-0.png", png_bytes(1, variant=1), "image/png")),
    ] + files

    response = post(client, files)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_upload"
    assert body["error"]["details"]["received"] == 1


def test_a_broken_enrolment_is_named_as_such(client) -> None:
    """Not `no_face_detected`, which is what a bad *capture* answers with.

    The two used to share a code, so an employee whose enrolment held one
    unreadable photo was told to move closer to the camera and try again - on
    every attempt, forever. Nothing they do in front of the lens changes a
    photograph an administrator uploaded months ago, so the caller has to be
    able to route them to HR instead.
    """
    response = post(client, build_image_files(people=[1, 1, 1, 1, 0]))

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "reference_faces_unusable"
    assert error["code"] != "no_face_detected"
    assert error["details"]["accepted"] == 4
    assert error["details"]["required"] == 5


def test_a_minority_of_references_cannot_carry_a_match(client) -> None:
    """Two photographs of somebody else must not clock in as the enrolled person.

    The score is the mean of the best three of five, so with the gate at two a
    set holding two planted photos scored above the threshold on their strength
    alone and the three genuine ones never got a vote. The gate matches the
    window now: the same three photos have to carry the mean AND clear the bar.
    """
    response = post(client, build_image_files(people=[1, 1, 2, 2, 2], capture_person=2))

    body = response.json()
    assert body["capture"]["matched_references"] == 3
    assert body["decision"] == "match", "three of five agreeing is still a match"

    # The other direction: the impostor's two photos alone do not carry it.
    response = post(client, build_image_files(people=[1, 1, 1, 2, 2], capture_person=2))

    body = response.json()
    assert body["capture"]["matched_references"] == 2
    assert body["passed"] is False
    assert body["decision"] != "match"


def test_each_reference_reports_its_own_similarity_to_the_capture(client) -> None:
    """Named for what it is, and not for frames that do not exist here.

    A deployment tuning `match_threshold` for its own cameras needs the spread
    the aggregate came from: one reference photo dragging every attempt down
    looks nothing like everybody scoring low under bad lighting, and the
    aggregate alone cannot tell them apart.
    """
    body = post(client, build_image_files()).json()

    for image in body["reference"]["images"]:
        assert isinstance(image["capture_similarity"], float)
        # The video-shaped fields stay empty on this endpoint rather than
        # quietly holding a different statistic under a name that lies.
        assert image["best_video_similarity"] is None
        assert image["mean_video_similarity"] is None


def test_the_capture_comes_back_when_asked_for(client) -> None:
    body = post(client, build_image_files(), {"include_capture": "true"}).json()

    assert isinstance(body["capture"]["image_base64"], str)
    assert body["capture"]["image_base64"] != ""


def test_the_capture_stays_out_of_the_response_by_default(client) -> None:
    body = post(client, build_image_files()).json()

    assert body["capture"]["image_base64"] is None


def test_a_stricter_threshold_travels_with_the_request(client) -> None:
    """One deployment answers for workspaces whose cameras are nothing alike."""
    body = post(client, build_image_files(), {"match_threshold": "0.999"}).json()

    assert body["thresholds"]["match_threshold"] == 0.999
    # The per-reference cut-off follows it, or `matched_references` would keep
    # counting agreement the caller has just said is not close enough.
    assert body["thresholds"]["reference_match_threshold"] == 0.999
    assert body["decision"] != "match"


def test_an_incoherent_enrolment_is_warned_about(client) -> None:
    """Five photos of four different people still score, and still say so."""
    body = post(client, build_image_files(people=[1, 1, 2, 3, 4])).json()

    assert body["reference"]["consistent"] is False
    assert any("same person" in warning for warning in body["warnings"])


def test_a_video_is_not_an_image(client) -> None:
    response = post(client, build_image_files(capture_type="video/mp4"))

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_the_endpoint_is_behind_the_api_key(client) -> None:
    response = client.post(
        "/api/v1/face/verify-image",
        files=build_image_files(),
        headers={"X-API-Key": "not-the-key"},
    )

    assert response.status_code == 401
