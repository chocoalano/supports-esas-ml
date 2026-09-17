"""Frame sampling from an uploaded video clip."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import Settings
from app.core.errors import VideoDecodeError
from app.services.media import downscale

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampledFrame:
    index: int
    timestamp_seconds: float | None
    image: np.ndarray


@dataclass
class SampledVideo:
    frames: list[SampledFrame] = field(default_factory=list)
    fps: float | None = None
    total_frames: int | None = None
    duration_seconds: float | None = None


class FrameSampler(Protocol):
    def sample(self, path: Path, max_frames: int | None = None) -> SampledVideo: ...


class OpenCVFrameSampler:
    """Decodes sequentially and keeps a bounded, evenly spread set of frames.

    Sequential decoding is used instead of frame seeking because seeking is
    unreliable for the VP8/VP9 WebM files produced by browser MediaRecorder.
    Memory stays bounded: whenever the kept list grows past twice the target the
    stride is doubled and every other frame is dropped.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def sample(self, path: Path, max_frames: int | None = None) -> SampledVideo:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise VideoDecodeError(
                "The video could not be opened. Supported containers: mp4, webm, mov, mkv, avi."
            )

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = fps if 0 < fps < 240 else None
            total_frames = total_frames if total_frames > 0 else None

            duration = total_frames / fps if (fps and total_frames) else None
            self._enforce_duration(duration)

            target = max(1, max_frames or self._settings.max_sampled_frames)
            stride = self._initial_stride(total_frames, fps, target)
            kept = self._decode(capture, stride=stride, target=target)
        finally:
            capture.release()

        if not kept:
            raise VideoDecodeError("No frames could be decoded from the video.")

        frames = evenly_spaced(kept, target)
        if duration is None and fps and kept:
            duration = (kept[-1].index + 1) / fps
            # Checked twice on purpose. A WebM written by a browser's
            # MediaRecorder routinely reports neither a frame count nor a
            # duration, so the check above sees nothing to check and the
            # documented 60-second limit silently stopped applying to exactly
            # the files this service is sent most. Once the clip has been
            # decoded its length is known, and the limit applies again.
            self._enforce_duration(duration)

        return SampledVideo(
            frames=frames,
            fps=fps,
            total_frames=total_frames or (kept[-1].index + 1),
            duration_seconds=round(duration, 3) if duration else None,
        )

    def _enforce_duration(self, duration: float | None) -> None:
        """Refuse a clip longer than the configured maximum.

        A clip whose container reports neither frame count nor frame rate cannot
        be measured before decoding; `max_decoded_frames` is what bounds that
        case, and it is a bound on work rather than on length.
        """
        if self._settings.max_video_seconds <= 0 or duration is None:
            return

        if duration > self._settings.max_video_seconds:
            raise VideoDecodeError(
                f"The video is {duration:.1f}s long, the maximum is "
                f"{self._settings.max_video_seconds:.0f}s.",
                details={"duration_seconds": round(duration, 2)},
            )

    def _initial_stride(self, total_frames: int | None, fps: float | None, target: int) -> int:
        if total_frames:
            return max(1, total_frames // target)
        if fps:
            # Unknown length: start at ~4 fps and let the decimator adapt.
            return max(1, int(round(fps / 4.0)))
        return 5

    def _decode(self, capture, *, stride: int, target: int) -> list[SampledFrame]:
        import cv2

        kept: list[SampledFrame] = []
        index = 0
        while index < self._settings.max_decoded_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0 and frame is not None and frame.size:
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                kept.append(
                    SampledFrame(
                        index=index,
                        timestamp_seconds=round(timestamp / 1000.0, 3) if timestamp else None,
                        image=downscale(frame, max_side=self._settings.frame_max_side),
                    )
                )
                if len(kept) > target * 2:
                    kept = kept[::2]
                    stride *= 2
            index += 1

        if index >= self._settings.max_decoded_frames:
            logger.info(
                "Stopped decoding at the %d frame cap.", self._settings.max_decoded_frames
            )
        return kept


def evenly_spaced(frames: list[SampledFrame], target: int) -> list[SampledFrame]:
    """Reduce `frames` to at most `target` items, evenly spread across the clip."""
    if len(frames) <= target:
        return frames
    step = len(frames) / float(target)
    return [frames[min(len(frames) - 1, int(i * step))] for i in range(target)]
