"""Similarity endpoints: reference photos against a clip, or against one still."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, UploadFile

from app.api.deps import ChallengeServiceDep, GuardDep, SettingsDep, VerificationServiceDep
from app.core.errors import InvalidUploadError
from app.schemas.verification import (
    ErrorResponse,
    ImageVerificationResponse,
    VerificationResponse,
)
from app.services.media import ensure_content_type, read_upload, temp_file
from app.services.verification import ReferenceUpload, ThresholdOverrides

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face", tags=["face"])


@router.post(
    "/verify",
    response_model=VerificationResponse,
    summary="Measure how similar the faces in a video are to 5 reference photos",
    responses={
        400: {"model": ErrorResponse, "description": "Challenge token invalid or expired"},
        401: {"model": ErrorResponse, "description": "Missing or unrecognised API key"},
        413: {"model": ErrorResponse, "description": "Upload too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        422: {"model": ErrorResponse, "description": "Bad upload or no usable face"},
        429: {"model": ErrorResponse, "description": "Too many requests for this key"},
        503: {"model": ErrorResponse, "description": "Face engine unavailable"},
    },
)
async def verify_face(
    caller: GuardDep,
    settings: SettingsDep,
    service: VerificationServiceDep,
    challenges: ChallengeServiceDep,
    images: list[UploadFile] = File(
        ..., description="Reference photos of one person (5 by default)."
    ),
    video: UploadFile = File(..., description="Recorded clip of the person to check."),
    challenge_token: str | None = Form(
        default=None,
        description="Token from /liveness/challenge. When present, the same clip is also "
        "checked for the challenge actions and `passed` requires both.",
    ),
    match_threshold: float | None = Form(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Override the configured similarity threshold for this request. "
        "One deployment answers for several workspaces, whose cameras and lighting "
        "are not alike.",
    ),
    min_match_ratio: float | None = Form(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the share of face-bearing frames that must match.",
    ),
    include_best_frame: bool = Form(
        default=False,
        description="Return the frame the score was measured on, as base64 JPEG, on "
        "`video.best_frame.image_base64`. A caller keeping a picture beside an attendance "
        "record wants that frame rather than one its own client chose.",
    ),
) -> VerificationResponse:
    """Stateless similarity check — nothing is stored, everything is returned inline."""
    _ensure_image_count(images, settings)

    service = service.with_thresholds(
        ThresholdOverrides(match_threshold=match_threshold, min_match_ratio=min_match_ratio)
    )

    challenge = challenges.verify_token(challenge_token) if challenge_token else None

    references: list[ReferenceUpload] = []
    for index, upload in enumerate(images):
        label = f"Reference image #{index + 1}"
        ensure_content_type(upload, settings.allowed_image_types, label=label)
        payload = await read_upload(upload, max_bytes=settings.max_image_bytes, label=label)
        references.append(
            ReferenceUpload(index=index, filename=upload.filename, payload=payload)
        )

    ensure_content_type(video, settings.allowed_video_types, label="Video")
    async with temp_file(video, max_bytes=settings.max_video_bytes, label="Video") as path:
        result = await service.verify(references, path, challenge, include_best_frame)

    logger.info(
        "verify caller=%s decision=%s passed=%s score=%.4f frames=%d/%d in %dms",
        caller,
        result.decision.value,
        result.passed,
        result.score,
        result.video.frames_with_face,
        result.video.frames_sampled,
        result.processing_ms,
    )
    return result


@router.post(
    "/verify-image",
    response_model=ImageVerificationResponse,
    summary="Measure how similar one captured photo is to 5 reference photos",
    responses={
        401: {"model": ErrorResponse, "description": "Missing or unrecognised API key"},
        413: {"model": ErrorResponse, "description": "Upload too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        422: {"model": ErrorResponse, "description": "Bad upload or no usable face"},
        429: {"model": ErrorResponse, "description": "Too many requests for this key"},
        503: {"model": ErrorResponse, "description": "Face engine unavailable"},
    },
)
async def verify_face_image(
    caller: GuardDep,
    settings: SettingsDep,
    service: VerificationServiceDep,
    images: list[UploadFile] = File(
        ..., description="Reference photos of one person (5 by default)."
    ),
    image: UploadFile = File(..., description="The captured photo of the person to check."),
    match_threshold: float | None = Form(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Override the configured similarity threshold for this request.",
    ),
    include_capture: bool = Form(
        default=False,
        description="Return the capture the score was measured on, as base64 JPEG, on "
        "`capture.image_base64`. A caller filing a picture beside an attendance record "
        "wants the one that was actually compared, downscaled by this service, rather "
        "than the full-size upload it sent.",
    ),
) -> ImageVerificationResponse:
    """Similarity only — this endpoint makes no claim that anybody was alive.

    A still cannot carry a liveness signal: head pose is measured across frames
    and a photograph has one. Whoever calls this owns that question and must
    answer it some other way; sending a photograph here and treating `passed` as
    proof of presence would be trusting whatever produced the file.

    `min_match_ratio` is absent for the same kind of reason: there is one probe,
    so a ratio over probes has nothing to range over. The agreement that a clip
    gets from its frames comes from the reference set here instead - see
    `FSA_MIN_REFERENCE_MATCHES` and `FSA_IMAGE_TOP_K_REFERENCES`.
    """
    _ensure_image_count(images, settings)

    service = service.with_thresholds(ThresholdOverrides(match_threshold=match_threshold))

    references: list[ReferenceUpload] = []
    for index, upload in enumerate(images):
        label = f"Reference image #{index + 1}"
        ensure_content_type(upload, settings.allowed_image_types, label=label)
        payload = await read_upload(upload, max_bytes=settings.max_image_bytes, label=label)
        references.append(
            ReferenceUpload(index=index, filename=upload.filename, payload=payload)
        )

    ensure_content_type(image, settings.allowed_image_types, label="Capture")
    capture = await read_upload(image, max_bytes=settings.max_image_bytes, label="Capture")

    result = await service.verify_image(references, capture, include_capture)

    # The per-reference similarities, not only the aggregate. A deployment tuning
    # `match_threshold` for its own cameras has to see the spread the number came
    # from: an aggregate alone cannot tell "everybody scores 0.35 under this
    # lighting" from "one reference photo is dragging every attempt down".
    logger.info(
        "verify-image caller=%s decision=%s passed=%s score=%.4f refs=%d/%d "
        "similarities=%s in %dms",
        caller,
        result.decision.value,
        result.passed,
        result.score,
        result.capture.matched_references,
        result.reference.accepted,
        [image.capture_similarity for image in result.reference.images],
        result.processing_ms,
    )
    return result


def _ensure_image_count(images: list[UploadFile], settings: SettingsDep) -> None:
    count = len(images)
    if settings.min_reference_images <= count <= settings.max_reference_images:
        return

    expected = (
        f"exactly {settings.min_reference_images}"
        if settings.min_reference_images == settings.max_reference_images
        else f"between {settings.min_reference_images} and {settings.max_reference_images}"
    )
    raise InvalidUploadError(
        f"Expected {expected} reference images, got {count}.",
        details={
            "received": count,
            "min": settings.min_reference_images,
            "max": settings.max_reference_images,
        },
    )
