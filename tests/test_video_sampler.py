"""Sampling from a real (synthetic) video file, through OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.core.errors import VideoDecodeError
from app.services.video import OpenCVFrameSampler


def write_video(
    path: Path, *, frames: int, fps: int = 30, size: tuple[int, int] = (320, 240)
) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV was built without an mp4 writer"
    for index in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :, 0] = index % 256
        writer.write(frame)
    writer.release()
    return path


def test_sampling_spreads_frames_across_the_clip(tmp_path: Path, settings: Settings):
    settings.max_sampled_frames = 8
    path = write_video(tmp_path / "clip.mp4", frames=120)

    result = OpenCVFrameSampler(settings).sample(path)

    assert len(result.frames) <= 8
    assert result.fps == pytest.approx(30.0)
    assert result.total_frames == 120
    assert result.duration_seconds == pytest.approx(4.0, abs=0.1)
    indices = [frame.index for frame in result.frames]
    assert indices == sorted(indices)
    assert indices[0] < 20 and indices[-1] > 80  # first and last thirds are covered
    assert all(frame.image.shape[2] == 3 for frame in result.frames)


def test_short_clips_are_returned_whole(tmp_path: Path, settings: Settings):
    settings.max_sampled_frames = 24
    path = write_video(tmp_path / "short.mp4", frames=5)

    result = OpenCVFrameSampler(settings).sample(path)

    assert len(result.frames) == 5


def test_large_frames_are_downscaled(tmp_path: Path, settings: Settings):
    settings.frame_max_side = 160
    path = write_video(tmp_path / "big.mp4", frames=10, size=(640, 480))

    result = OpenCVFrameSampler(settings).sample(path)

    assert max(result.frames[0].image.shape[:2]) == 160


def test_long_clips_are_rejected(tmp_path: Path, settings: Settings):
    settings.max_video_seconds = 1.0
    path = write_video(tmp_path / "long.mp4", frames=120)

    with pytest.raises(VideoDecodeError, match="maximum"):
        OpenCVFrameSampler(settings).sample(path)


def test_unreadable_file_raises(tmp_path: Path, settings: Settings):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video at all")

    with pytest.raises(VideoDecodeError):
        OpenCVFrameSampler(settings).sample(path)


def test_a_clip_whose_container_reports_no_length_is_still_measured(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    """The documented 60-second limit has to apply to the files this service is sent.

    A WebM written by a browser's MediaRecorder routinely reports neither a frame
    count nor a duration, so the check before decoding saw nothing to check and
    the limit silently stopped applying to exactly those clips. Once the frames
    are decoded the length is known, and it applies again.
    """
    settings.max_video_seconds = 2.0
    settings.max_sampled_frames = 8
    path = write_video(tmp_path / "unmarked.mp4", frames=150, fps=30)

    real_get = cv2.VideoCapture.get

    def blind(self, prop):  # noqa: ANN001 - patching an OpenCV method
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0
        return real_get(self, prop)

    monkeypatch.setattr(cv2.VideoCapture, "get", blind)

    with pytest.raises(VideoDecodeError) as refusal:
        OpenCVFrameSampler(settings).sample(path)

    assert refusal.value.code == "video_decode_failed"
    assert refusal.value.details["duration_seconds"] > 2.0


def test_a_clip_inside_the_limit_still_passes_when_unmarked(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    settings.max_video_seconds = 60.0
    settings.max_sampled_frames = 8
    path = write_video(tmp_path / "short-unmarked.mp4", frames=60, fps=30)

    real_get = cv2.VideoCapture.get

    def blind(self, prop):  # noqa: ANN001 - patching an OpenCV method
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0
        return real_get(self, prop)

    monkeypatch.setattr(cv2.VideoCapture, "get", blind)

    result = OpenCVFrameSampler(settings).sample(path)

    assert len(result.frames) > 0
    assert result.duration_seconds == pytest.approx(2.0, abs=0.2)
