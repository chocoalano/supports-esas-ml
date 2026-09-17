"""Challenge token issuing, signing and expiry."""

from __future__ import annotations

import time

import pytest

from app.core.config import Settings
from app.services.challenge import (
    ACTION_AXES,
    ACTION_CATALOGUE,
    ChallengeExpiredError,
    ChallengeService,
    ChallengeTokenError,
    _b64encode,
)


def test_issued_actions_come_from_the_configured_pool(challenges: ChallengeService):
    challenge, token = challenges.issue()

    assert len(challenge.actions) == 2
    assert len(set(challenge.actions)) == 2  # never asks for the same thing twice
    assert set(challenge.actions) <= set(challenges.available_actions)
    assert token.count(".") == 1


def test_a_token_round_trips(challenges: ChallengeService):
    challenge, token = challenges.issue(action_count=3)

    decoded = challenges.verify_token(token)

    assert decoded.challenge_id == challenge.challenge_id
    assert decoded.actions == challenge.actions
    assert decoded.expires_at == challenge.expires_at


def test_every_action_has_an_instruction(challenges: ChallengeService):
    _, token = challenges.issue(action_count=5)
    decoded = challenges.verify_token(token)

    instructions = challenges.instructions_for(decoded.actions)

    assert len(instructions) == len(decoded.actions)
    assert all(text for _, text in instructions)
    assert set(challenges.available_actions) <= set(ACTION_CATALOGUE)


def test_a_tampered_payload_is_rejected(challenges: ChallengeService):
    _, token = challenges.issue()
    payload, signature = token.split(".")
    forged = _b64encode(b'{"cid":"x","act":["turn_left"],"iat":0,"exp":99999999999}')

    with pytest.raises(ChallengeTokenError, match="signature"):
        challenges.verify_token(f"{forged}.{signature}")


def test_a_token_signed_with_another_key_is_rejected(settings: Settings):
    _, token = ChallengeService(settings.model_copy(update={"challenge_secret": "other"})).issue()

    with pytest.raises(ChallengeTokenError, match="signature"):
        ChallengeService(settings).verify_token(token)


def test_a_malformed_token_is_rejected(challenges: ChallengeService):
    with pytest.raises(ChallengeTokenError, match="malformed"):
        challenges.verify_token("not-a-token")


def test_an_expired_token_is_rejected(settings: Settings):
    settings.challenge_ttl_seconds = -1
    service = ChallengeService(settings)
    _, token = service.issue()

    with pytest.raises(ChallengeExpiredError):
        service.verify_token(token)


def test_action_count_is_clamped_to_the_number_of_axes(challenges: ChallengeService):
    """Not to the pool size: one action per axis is the real ceiling.

    The default pool holds five actions across three axes, so ninety-nine can
    only ever be answered with three.
    """
    challenge, _ = challenges.issue(action_count=99)

    axes = {ACTION_AXES[action] for action in challenges.available_actions}

    assert len(challenge.actions) == len(axes)
    assert len(challenge.actions) < len(challenges.available_actions)


def test_a_challenge_never_draws_two_actions_from_one_axis(challenges: ChallengeService):
    """`look_up` with `look_down` is a recording that cannot pass.

    The analyser takes the person's resting pose to be the median of the whole
    clip. Asked for both ends of one axis, the person spends the clip at those
    ends and the median lands between them, so each action is measured against
    the other rather than against rest. Both fall short of the threshold, and
    the attendance is refused naming actions that were performed correctly.

    Drawn a hundred times because the failure is a *sometimes*: a pool of five
    across three axes returns a same-axis pair often enough to be reported from
    the field, and rarely enough to survive a single-draw test.
    """
    for _ in range(100):
        challenge, _ = challenges.issue(action_count=2)

        axes = [ACTION_AXES[action] for action in challenge.actions]

        assert len(axes) == len(set(axes)), challenge.actions


def test_every_catalogued_action_declares_an_axis():
    """An action without an axis would crash the draw, not merely skip the rule."""
    assert set(ACTION_CATALOGUE) == set(ACTION_AXES)


def test_unknown_actions_are_dropped_from_the_pool(settings: Settings):
    settings.challenge_actions = ["turn_left", "do_a_backflip"]

    assert ChallengeService(settings).available_actions == ["turn_left"]


def test_ttl_counts_down_from_now(challenges: ChallengeService):
    challenge, _ = challenges.issue()

    assert 0 < challenge.ttl_seconds <= 120
    assert challenge.expires_at > int(time.time())


# --- HTTP layer -------------------------------------------------------------


def test_challenge_endpoint_returns_instructions(client):
    response = client.post("/api/v1/liveness/challenge", json={"action_count": 3})

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["actions"]) == 3
    assert all(item["instruction"] for item in body["actions"])
    assert body["ttl_seconds"] > 0
    assert body["challenge_id"]
    assert body["token"]


def test_challenge_endpoint_works_without_a_body(client):
    response = client.post("/api/v1/liveness/challenge")

    assert response.status_code == 200, response.text
    assert len(response.json()["actions"]) == 2
