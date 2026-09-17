"""Face detection + embedding extraction.

The heavy dependencies (`insightface`, `onnxruntime`) are imported lazily so the
application, its OpenAPI schema and the test-suite can run without the models
being present.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.core.config import Settings
from app.core.errors import EngineUnavailableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedFace:
    """A single detected face, its embedding and (optionally) its landmarks."""

    embedding: np.ndarray
    bbox: tuple[int, int, int, int]
    det_score: float
    #: 5 keypoints from the detector: left eye, right eye, nose, mouth corners.
    keypoints: np.ndarray | None = None
    #: 68 landmarks in the standard iBUG order, when the landmark model is on.
    landmarks_68: np.ndarray | None = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def shortest_side(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return min(max(0, x2 - x1), max(0, y2 - y1))


class FaceEngine(Protocol):
    """Interface the API depends on — trivially fakeable in tests."""

    name: str

    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Return every face found in a BGR image, ordered largest first."""

    def detect_primary(self, image: np.ndarray) -> DetectedFace | None:
        """Return the largest usable face, or None."""


class InsightFaceEngine:
    """`FaceEngine` backed by InsightFace's ArcFace models (ONNX Runtime)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._app = None
        self._lock = threading.Lock()
        self.name = settings.model_pack

    def is_loaded(self) -> bool:
        return self._app is not None

    def load(self) -> None:
        if self._app is not None:
            return
        with self._lock:
            if self._app is not None:
                return
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:  # pragma: no cover - environment issue
                raise EngineUnavailableError(
                    "insightface is not installed. Run `pip install -r requirements.txt`.",
                    details={"import_error": str(exc)},
                ) from exc

            logger.info("Loading InsightFace model pack '%s'...", self._settings.model_pack)
            providers = (
                ["CPUExecutionProvider"]
                if self._settings.ctx_id < 0
                else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            modules = ["detection", "recognition"]
            if self._settings.enable_landmarks:
                # Gives `face.landmark_3d_68` (iBUG order) — eyes and lips are
                # what the liveness checks measure.
                modules.append("landmark_3d_68")

            try:
                app = FaceAnalysis(
                    name=self._settings.model_pack,
                    root=self._settings.model_root or "~/.insightface",
                    allowed_modules=modules,
                    providers=providers,
                )
                app.prepare(
                    ctx_id=self._settings.ctx_id,
                    det_size=(self._settings.det_size, self._settings.det_size),
                )
            except Exception as exc:  # pragma: no cover - depends on local models
                raise EngineUnavailableError(
                    "Failed to initialise the face engine.",
                    details={"reason": str(exc)},
                ) from exc

            self._app = app
            logger.info("Face engine ready (providers=%s).", providers)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        self.load()
        assert self._app is not None

        faces = []
        for face in self._app.get(image):
            det_score = float(getattr(face, "det_score", 0.0))
            if det_score < self._settings.min_det_score:
                continue

            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                raw = getattr(face, "embedding", None)
                if raw is None:
                    continue
                embedding = normalise(np.asarray(raw, dtype=np.float32))

            x1, y1, x2, y2 = (int(v) for v in face.bbox[:4])
            keypoints = getattr(face, "kps", None)
            landmarks = getattr(face, "landmark_3d_68", None)
            detected = DetectedFace(
                embedding=np.asarray(embedding, dtype=np.float32),
                bbox=(x1, y1, x2, y2),
                det_score=det_score,
                keypoints=None if keypoints is None else np.asarray(keypoints, dtype=np.float32),
                landmarks_68=(
                    None if landmarks is None else np.asarray(landmarks, dtype=np.float32)[:, :2]
                ),
            )
            if detected.shortest_side < self._settings.min_face_pixels:
                continue
            faces.append(detected)

        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def detect_primary(self, image: np.ndarray) -> DetectedFace | None:
        faces = self.detect(image)
        return faces[0] if faces else None


def normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings, clipped to [-1, 1]."""
    value = float(np.dot(normalise(a), normalise(b)))
    return max(-1.0, min(1.0, value))
