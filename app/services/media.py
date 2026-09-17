"""Upload handling: size-guarded reads, image decoding, temp files."""

from __future__ import annotations

import base64
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import UploadFile

from app.core.errors import InvalidUploadError, PayloadTooLargeError, UnsupportedMediaError

CHUNK_SIZE = 1024 * 1024

IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"BM", "image/bmp"),
    (b"GIF8", "image/gif"),
)


async def read_upload(upload: UploadFile, *, max_bytes: int, label: str) -> bytes:
    """Read an upload into memory, aborting as soon as it exceeds `max_bytes`."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLargeError(
                f"{label} exceeds the maximum size of {max_bytes} bytes.",
                details={"filename": upload.filename, "max_bytes": max_bytes},
            )
        chunks.append(chunk)
    await upload.close()

    if total == 0:
        raise InvalidUploadError(
            f"{label} is empty.", details={"filename": upload.filename}
        )
    return b"".join(chunks)


@asynccontextmanager
async def temp_file(upload: UploadFile, *, max_bytes: int, label: str) -> AsyncIterator[Path]:
    """Stream an upload to a temp file that is removed on exit."""
    suffix = Path(upload.filename or "").suffix or ".bin"
    # Not a `with` block: the file must outlive the write loop and be handed to
    # OpenCV by path, so cleanup is done in the `finally` below instead.
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)  # noqa: SIM115
    path = Path(handle.name)
    total = 0
    try:
        while True:
            chunk = await upload.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise PayloadTooLargeError(
                    f"{label} exceeds the maximum size of {max_bytes} bytes.",
                    details={"filename": upload.filename, "max_bytes": max_bytes},
                )
            handle.write(chunk)
        handle.close()
        await upload.close()

        if total == 0:
            raise InvalidUploadError(
                f"{label} is empty.", details={"filename": upload.filename}
            )
        yield path
    finally:
        if not handle.closed:
            handle.close()
        path.unlink(missing_ok=True)


def ensure_content_type(upload: UploadFile, allowed: list[str], *, label: str) -> None:
    """Reject obviously wrong content types; `*` in `allowed` disables the check."""
    if "*" in allowed:
        return
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in [item.lower() for item in allowed]:
        raise UnsupportedMediaError(
            f"{label} has unsupported content type '{content_type}'.",
            details={"filename": upload.filename, "allowed": allowed},
        )


def looks_like_image(payload: bytes) -> bool:
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return True
    return any(payload.startswith(magic) for magic, _ in IMAGE_MAGIC)


def decode_image(payload: bytes, *, max_side: int, label: str) -> np.ndarray:
    """Decode image bytes into a BGR ndarray, downscaled to `max_side`."""
    import cv2

    if not looks_like_image(payload):
        raise InvalidUploadError(f"{label} is not a recognisable image file.")

    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidUploadError(f"{label} could not be decoded as an image.")
    return downscale(image, max_side=max_side)


def downscale(image: np.ndarray, *, max_side: int) -> np.ndarray:
    """Shrink an image so its longest side is at most `max_side`."""
    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def encode_jpeg(image: np.ndarray, *, max_side: int, quality: int) -> str | None:
    """Encode a frame as base64 JPEG, or None when it cannot be encoded.

    Downscaled first, because this travels inline in a JSON response: the caller
    wants a picture of the person the score was measured on, not the frame at
    detection resolution.
    """
    import cv2

    shrunk = downscale(image, max_side=max_side)
    ok, buffer = cv2.imencode(".jpg", shrunk, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

    if not ok:
        return None

    return base64.b64encode(buffer.tobytes()).decode()
