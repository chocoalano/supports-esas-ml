"""Stateless liveness challenges.

The service holds no database, so a challenge travels with the client as a signed
token: `base64url(payload).base64url(hmac_sha256(secret, payload))`. The server
can therefore trust the actions and the expiry it gets back without having stored
anything. What a signature cannot do is stop the *same* valid token being used
twice — see `challenge_id` and the replay note in the README.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import AppError

logger = logging.getLogger(__name__)


class ChallengeTokenError(AppError):
    code = "invalid_challenge"
    status_code = 400


class ChallengeExpiredError(AppError):
    code = "challenge_expired"
    status_code = 400


#: Action code -> instruction shown to the person being verified.
ACTION_CATALOGUE: dict[str, str] = {
    "turn_left": "Tolehkan kepala ke kiri Anda, tahan sebentar, lalu kembali menghadap kamera.",
    "turn_right": "Tolehkan kepala ke kanan Anda, tahan sebentar, lalu kembali menghadap kamera.",
    "look_up": "Angkat dagu ke atas sebentar, lalu kembali menghadap kamera.",
    "look_down": "Tundukkan kepala sebentar, lalu kembali menghadap kamera.",
    "open_mouth": "Buka mulut lebar-lebar sebentar, lalu tutup kembali.",
    "blink": "Pejamkan kedua mata sekitar satu detik, lalu buka lagi.",
}

#: Action code -> the facial measurement it moves.
#:
#: One challenge must never draw two actions sharing an axis, and the reason is
#: in how the analyser finds them: the pose it compares against is the *median
#: of the whole clip*, not a neutral frame it was handed. Draw `look_up` with
#: `look_down` and the person spends the clip at the two extremes of one axis
#: with barely any neutral in between — so the median lands roughly midway
#: between them, and each action is then measured from the other one instead of
#: from the person's resting pose. Both excursions are halved, both fall under
#: the threshold, and the recording is rejected with `aksi liveness tidak
#: terlihat` naming actions the person performed correctly.
#:
#: Two actions on *different* axes do not interfere: while the head turns, pitch
#: stays at rest, so each axis still sees a clear majority of neutral frames.
ACTION_AXES: dict[str, str] = {
    "turn_left": "yaw",
    "turn_right": "yaw",
    "look_up": "pitch",
    "look_down": "pitch",
    "open_mouth": "mouth",
    "blink": "eye",
}


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    actions: list[str]
    issued_at: int
    expires_at: int

    @property
    def ttl_seconds(self) -> int:
        return max(0, self.expires_at - int(time.time()))


class ChallengeService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        secret = settings.challenge_secret
        if not secret:
            secret = secrets.token_urlsafe(32)
            logger.warning(
                "FSA_CHALLENGE_SECRET is not set; a random key was generated. Challenge "
                "tokens will stop verifying after a restart and across workers. Set it "
                "explicitly before deploying."
            )
        self._secret = secret.encode()

    @property
    def available_actions(self) -> list[str]:
        """Configured actions, minus anything not in the catalogue."""
        return [action for action in self._settings.challenge_actions if action in ACTION_CATALOGUE]

    def issue(self, *, action_count: int | None = None) -> tuple[Challenge, str]:
        """Draw a random, ordered set of actions and sign them.

        At most one action per axis — see `ACTION_AXES`. That caps a challenge at
        the number of distinct axes in the pool, which is why the requested count
        is clamped to *that* rather than to the pool size: asking for four
        actions from a five-action pool spanning three axes can only ever be
        answered with three.
        """
        pool = self.available_actions
        if not pool:
            raise ChallengeTokenError("No liveness actions are configured.")

        count = action_count or self._settings.challenge_action_count
        count = max(1, min(count, len({ACTION_AXES[action] for action in pool})))

        remaining = list(pool)
        actions: list[str] = []
        for _ in range(count):
            drawn = remaining.pop(secrets.randbelow(len(remaining)))
            actions.append(drawn)

            axis = ACTION_AXES[drawn]
            remaining = [action for action in remaining if ACTION_AXES[action] != axis]

        now = int(time.time())
        challenge = Challenge(
            challenge_id=secrets.token_urlsafe(16),
            actions=actions,
            issued_at=now,
            expires_at=now + self._settings.challenge_ttl_seconds,
        )
        return challenge, self._encode(challenge)

    def verify_token(self, token: str) -> Challenge:
        """Decode a token, or raise if it was tampered with or has expired."""
        try:
            payload_part, signature_part = token.strip().split(".", 1)
            payload = _b64decode(payload_part)
            signature = _b64decode(signature_part)
        except (ValueError, TypeError) as exc:
            raise ChallengeTokenError("The challenge token is malformed.") from exc

        expected = hmac.new(self._secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise ChallengeTokenError("The challenge token signature does not match.")

        try:
            data = json.loads(payload)
            challenge = Challenge(
                challenge_id=str(data["cid"]),
                actions=[str(action) for action in data["act"]],
                issued_at=int(data["iat"]),
                expires_at=int(data["exp"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ChallengeTokenError("The challenge token payload is invalid.") from exc

        if challenge.expires_at <= int(time.time()):
            raise ChallengeExpiredError(
                "The challenge has expired; request a new one.",
                details={"challenge_id": challenge.challenge_id},
            )
        unknown = [action for action in challenge.actions if action not in ACTION_CATALOGUE]
        if unknown:
            raise ChallengeTokenError(
                "The challenge token asks for unknown actions.",
                details={"actions": unknown},
            )
        return challenge

    def instructions_for(self, actions: list[str]) -> list[tuple[str, str]]:
        return [(action, ACTION_CATALOGUE[action]) for action in actions]

    def _encode(self, challenge: Challenge) -> str:
        payload = json.dumps(
            {
                "cid": challenge.challenge_id,
                "act": challenge.actions,
                "iat": challenge.issued_at,
                "exp": challenge.expires_at,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        signature = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return f"{_b64encode(payload)}.{_b64encode(signature)}"


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
