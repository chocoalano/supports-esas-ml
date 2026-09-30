"""Every speech route is behind the same guard as Face - the two GETs included.

GET routes that cost nothing are the ones most easily left open, so each is
tested explicitly rather than assumed.
"""

from __future__ import annotations

import pytest

from app.api.deps import limiter_for
from tests import audio_samples
from tests.conftest import API_KEY, make_speech_client


def call(client, route: str, **kwargs):
    if route == "transcribe":
        return client.post(
            "/api/v1/speech/transcribe",
            files={"audio": ("clip.wav", audio_samples.wav(), "audio/wav")},
            **kwargs,
        )
    if route == "synthesize":
        return client.post("/api/v1/speech/synthesize", json={"text": "halo"}, **kwargs)
    return client.get(f"/api/v1/speech/{route}", **kwargs)


ROUTES = ["transcribe", "synthesize", "voices", "capabilities"]


@pytest.fixture(autouse=True)
def _forget_counts():
    """Rate limiters are process-wide; a test must not inherit another's window."""
    limiter_for.cache_clear()
    yield
    limiter_for.cache_clear()


@pytest.fixture
def unkeyed(speech_settings, stt_engine, tts_engine):
    """A client that sends no key unless a test says so."""

    def build(**overrides):
        settings = speech_settings.model_copy(update=overrides)
        return make_speech_client(settings, stt_engine, tts_engine, api_key=None)

    return build


@pytest.mark.parametrize("route", ROUTES)
def test_no_key_is_refused(unkeyed, route):
    with unkeyed() as client:
        response = call(client, route)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize("route", ROUTES)
def test_the_wrong_key_is_refused(unkeyed, route):
    with unkeyed() as client:
        response = call(client, route, headers={"X-API-Key": "not-the-key"})

    assert response.status_code == 401


@pytest.mark.parametrize("route", ROUTES)
def test_the_right_key_is_served(unkeyed, route):
    with unkeyed() as client:
        response = call(client, route, headers={"X-API-Key": API_KEY})

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("route", ROUTES)
def test_a_service_with_no_keys_refuses_everything(unkeyed, route):
    with unkeyed(api_keys=[], debug=False) as client:
        response = call(client, route, headers={"X-API-Key": API_KEY})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "api_keys_not_configured"


@pytest.mark.parametrize("route", ["transcribe", "synthesize", "voices"])
def test_the_key_is_checked_before_the_feature_flag(unkeyed, route):
    """An unknown caller learns nothing - not even which features are switched on."""
    with unkeyed(stt_enabled=False, tts_enabled=False) as client:
        response = call(client, route)

    assert response.status_code == 401


def test_the_key_is_checked_before_the_body(unkeyed):
    with unkeyed() as client:
        response = client.post("/api/v1/speech/synthesize", json={"format": "wav"})

    assert response.status_code == 401


def test_a_key_is_cut_off_once_it_is_over_its_rate(unkeyed):
    with unkeyed(rate_limit_per_minute=2) as client:
        headers = {"X-API-Key": API_KEY}
        statuses = [call(client, "capabilities", headers=headers).status_code for _ in range(3)]

    assert statuses == [200, 200, 429]


def test_the_rate_is_counted_per_workspace(unkeyed):
    with unkeyed(rate_limit_per_minute=1) as client:
        acme = {"X-API-Key": API_KEY, "X-Tenant": "acme"}
        globex = {"X-API-Key": API_KEY, "X-Tenant": "globex"}

        assert call(client, "voices", headers=acme).status_code == 200
        assert call(client, "voices", headers=acme).status_code == 429
        assert call(client, "voices", headers=globex).status_code == 200


def test_health_stays_open(unkeyed):
    with unkeyed() as client:
        assert client.get("/api/v1/health").status_code == 200
