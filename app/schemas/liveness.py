"""Request/response models for the liveness challenge endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChallengeRequest(BaseModel):
    action_count: int | None = Field(
        default=None,
        ge=1,
        le=6,
        description="How many actions to ask for. Defaults to FSA_CHALLENGE_ACTION_COUNT.",
    )


class ChallengeAction(BaseModel):
    action: str = Field(description="Machine-readable action code, e.g. `turn_left`.")
    instruction: str = Field(description="Sentence to show the person, in order.")


class ChallengeResponse(BaseModel):
    challenge_id: str = Field(
        description="Store this to reject a second use of the same challenge."
    )
    token: str = Field(description="Signed token; send it back with the recording.")
    actions: list[ChallengeAction]
    issued_at: int
    expires_at: int = Field(description="Unix timestamp after which the token is refused.")
    ttl_seconds: int


class ActionResult(BaseModel):
    action: str
    instruction: str
    performed: bool
    detected_at_seconds: float | None = Field(
        default=None, description="Where in the clip the action was observed."
    )
    frame_index: int | None = None
    measured: float | None = Field(
        default=None, description="Most extreme value seen for this action's metric."
    )
    baseline: float | None = Field(
        default=None, description="The person's own neutral value for that metric."
    )
    reason: str | None = Field(
        default=None,
        description="Why it failed: not_detected, not_enough_frames, landmarks_unavailable.",
    )


class LivenessReport(BaseModel):
    live: bool = Field(description="True when every action was performed, in the order asked.")
    challenge_id: str | None = None
    actions: list[ActionResult]
    frames_sampled: int
    frames_with_face: int
    movement_score: float = Field(
        description="Spread of head pose across the clip. Near zero means a still image."
    )
    warnings: list[str] = Field(default_factory=list)


class LivenessResponse(LivenessReport):
    processing_ms: int
