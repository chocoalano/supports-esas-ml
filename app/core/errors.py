"""Domain errors and their HTTP representation."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Plain integers rather than `starlette.status` constants: the constant names for
# 413/422 were renamed and the old ones now emit deprecation warnings.


class AppError(Exception):
    """Base class for errors that map to a clean JSON response."""

    code: str = "error"
    status_code: int = 400

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidUploadError(AppError):
    code = "invalid_upload"
    status_code = 422


class PayloadTooLargeError(AppError):
    code = "payload_too_large"
    status_code = 413


class UnsupportedMediaError(AppError):
    code = "unsupported_media_type"
    status_code = 415


class NoFaceDetectedError(AppError):
    code = "no_face_detected"
    status_code = 422


class ReferenceFacesUnusableError(AppError):
    """The person's enrolment photos cannot be scored — not their capture.

    Its own code, because the caller has to tell these two apart and cannot.
    Both used to be `no_face_detected`, so an employee whose enrolment held one
    unreadable photo was told to move closer to the camera and try again, on
    every attempt, forever. The instruction was impossible: nothing they do in
    front of the lens changes a photograph an administrator uploaded months ago.
    """

    code = "reference_faces_unusable"
    status_code = 422


class VideoDecodeError(AppError):
    code = "video_decode_failed"
    status_code = 422


class EngineUnavailableError(AppError):
    code = "engine_unavailable"
    status_code = 503


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )
