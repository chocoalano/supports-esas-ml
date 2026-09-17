"""Who gets in, how often, and by whose thresholds.

The service performs face recognition on whatever it is handed. These are the
tests for the difference between "reachable" and "usable by anybody".
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import (
    get_challenge_service,
    get_face_engine,
    get_frame_sampler,
    limiter_for,
)
from app.core.config import Settings, get_settings
from app.main import create_app
from tests.conftest import API_KEY, build_files

VERIFY = "/api/v1/face/verify"
CHALLENGE = "/api/v1/liveness/challenge"


def make_client(settings: Settings, engine, sampler, challenges) -> TestClient:
    """A client with no key of its own, so each test sends exactly what it means to."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_face_engine] = lambda: engine
    app.dependency_overrides[get_frame_sampler] = lambda: sampler
    app.dependency_overrides[get_challenge_service] = lambda: challenges

    return TestClient(app)


@pytest.fixture(autouse=True)
def _forget_counts():
    """Rate limiters are process-wide; a test must not inherit another's window."""
    limiter_for.cache_clear()
    yield
    limiter_for.cache_clear()


# -- the key ------------------------------------------------------------------


def test_a_request_without_a_key_is_refused(settings, engine, sampler, challenges):
    with make_client(settings, engine, sampler, challenges) as client:
        response = client.post(CHALLENGE, json={"action_count": 2})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_a_request_with_the_wrong_key_is_refused(settings, engine, sampler, challenges):
    with make_client(settings, engine, sampler, challenges) as client:
        response = client.post(CHALLENGE, headers={"X-API-Key": "not-the-key"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_a_request_with_a_configured_key_is_served(settings, engine, sampler, challenges):
    with make_client(settings, engine, sampler, challenges) as client:
        response = client.post(CHALLENGE, headers={"X-API-Key": API_KEY})

    assert response.status_code == 200
    assert response.json()["actions"]


def test_scoring_is_refused_without_a_key(settings, engine, sampler, challenges):
    """The expensive endpoint is guarded, not only the cheap one."""
    with make_client(settings, engine, sampler, challenges) as client:
        response = client.post(VERIFY, files=build_files())

    assert response.status_code == 401


def test_health_answers_without_a_key(settings, engine, sampler, challenges):
    """A readiness probe carries no credentials and discloses nothing that matters."""
    with make_client(settings, engine, sampler, challenges) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200


# -- the default when nothing is configured -----------------------------------


def test_a_service_with_no_keys_refuses_everything(settings, engine, sampler, challenges):
    """Closed by default: an unconfigured deployment is not an open one."""
    unconfigured = settings.model_copy(update={"api_keys": [], "debug": False})

    with make_client(unconfigured, engine, sampler, challenges) as client:
        response = client.post(CHALLENGE, headers={"X-API-Key": API_KEY})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "api_keys_not_configured"


def test_debug_admits_an_unkeyed_request(settings, engine, sampler, challenges):
    """The local-development path, and the only one that opens the door."""
    development = settings.model_copy(update={"api_keys": [], "debug": True})

    with make_client(development, engine, sampler, challenges) as client:
        response = client.post(CHALLENGE)

    assert response.status_code == 200


# -- the rate ------------------------------------------------------------------


def test_a_key_is_cut_off_once_it_is_over_its_rate(settings, engine, sampler, challenges):
    throttled = settings.model_copy(update={"rate_limit_per_minute": 2})

    with make_client(throttled, engine, sampler, challenges) as client:
        headers = {"X-API-Key": API_KEY}
        first = client.post(CHALLENGE, headers=headers)
        second = client.post(CHALLENGE, headers=headers)
        third = client.post(CHALLENGE, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["error"]["details"]["limit"] == 2


def test_one_key_cannot_spend_another_keys_allowance(settings, engine, sampler, challenges):
    """Counts are per caller, which is the point of labelling the keys."""
    two_keys = settings.model_copy(
        update={"api_keys": [f"one:{API_KEY}", "two:second-key"], "rate_limit_per_minute": 1}
    )

    with make_client(two_keys, engine, sampler, challenges) as client:
        assert client.post(CHALLENGE, headers={"X-API-Key": API_KEY}).status_code == 200
        assert client.post(CHALLENGE, headers={"X-API-Key": API_KEY}).status_code == 429
        assert client.post(CHALLENGE, headers={"X-API-Key": "second-key"}).status_code == 200


# -- per-request thresholds ----------------------------------------------------


def test_a_stricter_threshold_turns_a_match_into_a_refusal(client):
    """The same clip, judged by a workspace that asks for more."""
    lenient = client.post(VERIFY, files=build_files(video_person=1))
    strict = client.post(
        VERIFY,
        files=build_files(video_person=1),
        data={"match_threshold": "0.999"},
    )

    assert lenient.json()["decision"] == "match"
    assert strict.status_code == 200, strict.text
    assert strict.json()["decision"] != "match"
    # The response reports the thresholds it actually used, so a caller can see
    # which numbers produced the verdict it is being handed.
    assert strict.json()["thresholds"]["match_threshold"] == pytest.approx(0.999)


def test_the_frame_threshold_follows_the_aggregate_one(client):
    """Otherwise every frame still counts as matched and match_ratio means nothing."""
    response = client.post(
        VERIFY,
        files=build_files(video_person=1),
        data={"match_threshold": "0.999"},
    )

    body = response.json()
    assert body["thresholds"]["frame_match_threshold"] == pytest.approx(0.999)
    assert body["video"]["match_ratio"] == 0.0


def test_a_looser_ratio_is_honoured(client):
    """The other half of the override, and it reaches the response too."""
    response = client.post(
        VERIFY,
        files=build_files(video_person=1),
        data={"min_match_ratio": "0.1"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["thresholds"]["min_match_ratio"] == pytest.approx(0.1)


def test_an_out_of_range_threshold_is_refused(client):
    response = client.post(
        VERIFY,
        files=build_files(video_person=1),
        data={"match_threshold": "3"},
    )

    assert response.status_code == 422


# -- keys that are not keys ---------------------------------------------------


def test_a_key_with_non_ascii_bytes_is_refused_not_a_server_error(
    settings, engine, sampler, challenges
):
    """`hmac.compare_digest` refuses two strs unless both are ASCII-only.

    An ASGI header is decoded as latin-1, so a single byte above 127 turned an
    unrecognised key into a TypeError and a 500. A malformed key is a refusal.
    """
    with make_client(settings, engine, sampler, challenges) as client:
        response = client.post(
            CHALLENGE,
            json={"action_count": 2},
            headers={"X-API-Key": "kunci-café".encode()},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_an_entry_with_no_key_is_dropped_and_named(settings):
    """Empty is not a password.

    Dropping it is the safe direction and the silent one, so the labels are
    reported for the boot log - a key that was meant to work should look like a
    misconfiguration rather than a 401 nobody can explain.
    """
    configured = Settings(
        warm_up_on_startup=False,
        api_keys=["good:real-key", "broken:", "  ", "bare-key"],
    )

    assert configured.api_key_map == {"good": "real-key", "default-3": "bare-key"}
    assert configured.malformed_api_keys == ["broken", "entry #3"]


def test_a_key_that_is_only_whitespace_admits_nobody(settings, engine, sampler, challenges):
    configured = settings.model_copy(update={"api_keys": ["label:   "]})

    with make_client(configured, engine, sampler, challenges) as client:
        assert client.post(CHALLENGE, json={}).status_code == 503
        assert (
            client.post(CHALLENGE, json={}, headers={"X-API-Key": "   "}).status_code == 503
        )


# -- one key, many workspaces -------------------------------------------------


def test_the_limit_is_counted_per_workspace_not_per_installation(
    settings, engine, sampler, challenges
):
    """One installation of a multi-tenant caller is one API key.

    Counted per key alone, the first company clocking in at 08:00 spends the
    whole allowance and every other company is told the face service is
    unavailable. The caller is the only party that knows which workspace a
    request is for, so it travels in a header.
    """
    limited = settings.model_copy(update={"rate_limit_per_minute": 2})

    with make_client(limited, engine, sampler, challenges) as client:
        for _ in range(2):
            assert (
                client.post(
                    CHALLENGE,
                    json={},
                    headers={"X-API-Key": API_KEY, "X-Tenant": "acme"},
                ).status_code
                == 200
            )

        assert (
            client.post(
                CHALLENGE, json={}, headers={"X-API-Key": API_KEY, "X-Tenant": "acme"}
            ).status_code
            == 429
        )

        # A different workspace, the same key, its own allowance.
        assert (
            client.post(
                CHALLENGE, json={}, headers={"X-API-Key": API_KEY, "X-Tenant": "globex"}
            ).status_code
            == 200
        )


def test_a_caller_that_names_no_workspace_still_has_one_bucket(
    settings, engine, sampler, challenges
):
    limited = settings.model_copy(update={"rate_limit_per_minute": 1})

    with make_client(limited, engine, sampler, challenges) as client:
        assert client.post(CHALLENGE, json={}, headers={"X-API-Key": API_KEY}).status_code == 200
        assert client.post(CHALLENGE, json={}, headers={"X-API-Key": API_KEY}).status_code == 429


def test_a_workspace_header_cannot_mint_unbounded_buckets(
    settings, engine, sampler, challenges
):
    """The header comes off the wire, so what it can key is bounded."""
    limited = settings.model_copy(update={"rate_limit_per_minute": 1})
    long_name = "w" * 500

    with make_client(limited, engine, sampler, challenges) as client:
        first = client.post(
            CHALLENGE, json={}, headers={"X-API-Key": API_KEY, "X-Tenant": long_name}
        )
        second = client.post(
            CHALLENGE,
            json={},
            headers={"X-API-Key": API_KEY, "X-Tenant": long_name + "-different-tail"},
        )

    assert first.status_code == 200
    # Truncated to the same bucket: the first 64 characters are identical.
    assert second.status_code == 429
