"""Liveness challenge: hand out random actions, then check the recording."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict

from fastapi import APIRouter, File, Form, UploadFile

from app.api.deps import ChallengeServiceDep, GuardDep, LivenessServiceDep, SettingsDep
from app.schemas.liveness import (
    ChallengeAction,
    ChallengeRequest,
    ChallengeResponse,
    LivenessResponse,
)
from app.schemas.verification import ErrorResponse
from app.services.media import ensure_content_type, temp_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/liveness", tags=["liveness"])


@router.post(
    "/challenge",
    response_model=ChallengeResponse,
    summary="Issue a random, signed set of actions for the person to perform",
    responses={
        400: {"model": ErrorResponse, "description": "No actions configured"},
        401: {"model": ErrorResponse, "description": "Missing or unrecognised API key"},
        429: {"model": ErrorResponse, "description": "Too many requests for this key"},
    },
)
async def create_challenge(
    caller: GuardDep,
    challenges: ChallengeServiceDep,
    payload: ChallengeRequest | None = None,
) -> ChallengeResponse:
    """Nothing is stored: the actions and the expiry are signed into the token."""
    challenge, token = challenges.issue(
        action_count=payload.action_count if payload else None
    )
    return ChallengeResponse(
        challenge_id=challenge.challenge_id,
        token=token,
        actions=[
            ChallengeAction(action=action, instruction=instruction)
            for action, instruction in challenges.instructions_for(challenge.actions)
        ],
        issued_at=challenge.issued_at,
        expires_at=challenge.expires_at,
        ttl_seconds=challenge.ttl_seconds,
    )


@router.post(
    "/verify",
    response_model=LivenessResponse,
    summary="Check that the recording performs the challenge actions, in order",
    responses={
        400: {"model": ErrorResponse, "description": "Token invalid or expired"},
        401: {"model": ErrorResponse, "description": "Missing or unrecognised API key"},
        413: {"model": ErrorResponse, "description": "Upload too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        422: {"model": ErrorResponse, "description": "Video could not be read"},
        429: {"model": ErrorResponse, "description": "Too many requests for this key"},
        503: {"model": ErrorResponse, "description": "Face engine unavailable"},
    },
)
async def verify_liveness(
    caller: GuardDep,
    settings: SettingsDep,
    challenges: ChallengeServiceDep,
    liveness: LivenessServiceDep,
    token: str = Form(..., description="Token returned by /liveness/challenge."),
    video: UploadFile = File(..., description="Recording of the person doing the actions."),
) -> LivenessResponse:
    started = time.perf_counter()
    challenge = challenges.verify_token(token)

    ensure_content_type(video, settings.allowed_video_types, label="Video")
    async with temp_file(video, max_bytes=settings.max_video_bytes, label="Video") as path:
        outcome = await liveness.verify(challenge, path)

    logger.info(
        "liveness caller=%s challenge=%s live=%s actions=%s",
        caller,
        challenge.challenge_id,
        outcome.live,
        ",".join(f"{item.action}:{'ok' if item.performed else 'fail'}" for item in outcome.actions),
    )

    return LivenessResponse(
        **asdict(outcome),
        processing_ms=int((time.perf_counter() - started) * 1000),
    )
