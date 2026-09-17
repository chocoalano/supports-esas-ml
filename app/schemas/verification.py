"""Response models for the verification endpoint."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.liveness import LivenessReport


class Decision(str, Enum):
    MATCH = "match"
    NO_MATCH = "no_match"
    INCONCLUSIVE = "inconclusive"


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ReferenceImageReport(BaseModel):
    index: int = Field(description="Position of the file in the uploaded `images` list.")
    filename: str | None = None
    face_detected: bool
    detection_score: float | None = Field(
        default=None, description="Detector confidence, 0..1."
    )
    bbox: BoundingBox | None = None
    similarity_to_group: float | None = Field(
        default=None,
        description="Mean cosine similarity against the other accepted reference photos.",
    )
    best_video_similarity: float | None = Field(
        default=None, description="Highest cosine similarity found across video frames."
    )
    mean_video_similarity: float | None = Field(
        default=None, description="Mean cosine similarity across face-bearing frames."
    )
    capture_similarity: float | None = Field(
        default=None,
        description="Cosine similarity between this reference photo and the single "
        "capture. Set by /face/verify-image only - the two video fields above stay "
        "null there, because a still has no frames and a field named after one that "
        "quietly held a different statistic is worse than an absent field.",
    )
    skip_reason: str | None = None


class ReferenceReport(BaseModel):
    submitted: int
    accepted: int
    rejected: int
    cohesion: float | None = Field(
        default=None,
        description="Mean pairwise cosine similarity between the accepted photos. "
        "Low values suggest the photos are not all the same person.",
    )
    consistent: bool = Field(
        description="True when `cohesion` clears the configured cohesion threshold."
    )
    images: list[ReferenceImageReport]


class FrameReport(BaseModel):
    frame_index: int = Field(description="Frame number inside the decoded video.")
    timestamp_seconds: float | None = None
    face_detected: bool
    detection_score: float | None = None
    bbox: BoundingBox | None = None
    similarity: float | None = Field(
        default=None, description="Best cosine similarity against any reference photo."
    )
    matched: bool = False
    best_reference_index: int | None = None
    image_base64: str | None = Field(
        default=None,
        description="The frame itself, JPEG, base64. Present only on `best_frame`, and "
        "only when `include_best_frame` was requested: it is the picture the score was "
        "actually measured on, which is the one worth filing beside an attendance record.",
    )


class VideoReport(BaseModel):
    duration_seconds: float | None = None
    fps: float | None = None
    total_frames: int | None = None
    frames_sampled: int
    frames_with_face: int
    frames_matched: int
    match_ratio: float = Field(description="frames_matched / frames_with_face (0 when no faces).")
    best_frame: FrameReport | None = None
    frames: list[FrameReport]


class Thresholds(BaseModel):
    match_threshold: float
    frame_match_threshold: float
    min_match_ratio: float
    inconclusive_band: float


class VerificationResponse(BaseModel):
    passed: bool = Field(
        description="The single field to gate on: the faces match AND, when a challenge "
        "token was sent, the liveness actions were performed."
    )
    match: bool = Field(description="True only when the decision is `match`.")
    decision: Decision
    score: float = Field(description="Aggregated cosine similarity, -1..1.")
    score_percent: float = Field(description="Aggregated score mapped to 0..100.")
    confidence: float = Field(
        description="0..1 logistic confidence in the returned decision."
    )
    thresholds: Thresholds
    reference: ReferenceReport
    video: VideoReport
    liveness: LivenessReport | None = Field(
        default=None, description="Present only when a challenge token was sent."
    )
    warnings: list[str] = Field(default_factory=list)
    model: str
    processing_ms: int


class ImageThresholds(BaseModel):
    match_threshold: float
    reference_match_threshold: float = Field(
        description="Cut-off a single reference photo must clear on its own."
    )
    min_reference_matches: int
    top_k_references: int
    inconclusive_band: float


class CaptureReport(BaseModel):
    """What was found in the one still that was sent to be checked."""

    face_detected: bool
    faces_found: int = Field(
        default=0, description="How many faces the detector found in the capture."
    )
    detection_score: float | None = None
    bbox: BoundingBox | None = None
    face_pixels: int | None = Field(
        default=None,
        description="Shortest side of the face box, in pixels. A small number is "
        "why a genuine person can score badly: they stood too far back.",
    )
    similarity: float | None = Field(
        default=None, description="Best cosine similarity against any reference photo."
    )
    matched: bool = False
    best_reference_index: int | None = None
    matched_references: int = Field(
        default=0,
        description="How many reference photos this capture cleared individually.",
    )
    image_base64: str | None = Field(
        default=None,
        description="The capture itself, JPEG, base64 — returned only when "
        "`include_capture` was requested, so the caller files the picture the "
        "score was measured on rather than one it kept separately.",
    )


class ImageVerificationResponse(BaseModel):
    """The answer for a still.

    Deliberately the same field names as `VerificationResponse` wherever the two
    mean the same thing, so a caller reading `passed` / `decision` / `score` does
    not have to branch on which endpoint answered. What is absent is as
    meaningful: there is no `liveness` block, because a still cannot carry one and
    this service will not pretend otherwise. Whoever calls this owns that
    question.
    """

    passed: bool = Field(
        description="The single field to gate on: the capture matches the "
        "reference set. It says nothing about whether the person was alive."
    )
    match: bool = Field(description="True only when the decision is `match`.")
    decision: Decision
    score: float = Field(description="Aggregated cosine similarity, -1..1.")
    score_percent: float = Field(description="Aggregated score mapped to 0..100.")
    confidence: float = Field(
        description="0..1 logistic confidence in the returned decision."
    )
    thresholds: ImageThresholds
    reference: ReferenceReport
    capture: CaptureReport
    warnings: list[str] = Field(default_factory=list)
    model: str
    processing_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    model_loaded: bool


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody
