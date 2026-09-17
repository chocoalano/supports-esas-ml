"""Core comparison logic: 5 reference photos vs. the frames of a video clip."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from anyio import Semaphore, to_thread

from app.core.config import Settings
from app.core.errors import NoFaceDetectedError, ReferenceFacesUnusableError
from app.schemas.liveness import LivenessReport
from app.schemas.verification import (
    BoundingBox,
    CaptureReport,
    Decision,
    FrameReport,
    ImageThresholds,
    ImageVerificationResponse,
    ReferenceImageReport,
    ReferenceReport,
    Thresholds,
    VerificationResponse,
    VideoReport,
)
from app.services.challenge import Challenge
from app.services.face_engine import DetectedFace, FaceEngine
from app.services.liveness import FaceMetrics, LivenessAnalyzer, metrics_from_face
from app.services.media import decode_image, encode_jpeg
from app.services.video import FrameSampler, SampledFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceUpload:
    index: int
    filename: str | None
    payload: bytes


@dataclass(frozen=True)
class ThresholdOverrides:
    """Per-request thresholds, sent by the caller.

    One deployment of this service answers for several workspaces, and their
    cameras are nothing alike: a warehouse gate at night and an office lobby do
    not share a sensible cut-off. The caller knows which workspace a request is
    for; this service does not and should not, so the numbers travel with the
    request rather than being configured per tenant here.

    Only the two thresholds that decide the verdict are overridable. The frame
    threshold, the top-K window and the inconclusive band are properties of how
    the score is computed, and letting a caller move those would let it move the
    meaning of the score it is being handed.
    """

    match_threshold: float | None = None
    min_match_ratio: float | None = None

    def is_empty(self) -> bool:
        return self.match_threshold is None and self.min_match_ratio is None

    def as_update(self) -> dict[str, float]:
        """The settings fields these overrides replace."""
        update: dict[str, float] = {}

        if self.match_threshold is not None:
            update["match_threshold"] = self.match_threshold
            # The per-frame cut-off follows the aggregate one. Left behind, a
            # caller asking for a stricter score would still have every frame
            # counted as a match, and `match_ratio` would stop meaning anything.
            update["frame_match_threshold"] = self.match_threshold

        if self.min_match_ratio is not None:
            update["min_match_ratio"] = self.min_match_ratio

        return update


@dataclass
class _Reference:
    index: int
    filename: str | None
    face: DetectedFace


class VerificationService:
    def __init__(
        self,
        *,
        engine: FaceEngine,
        sampler: FrameSampler,
        settings: Settings,
        limiter: Semaphore | None = None,
    ) -> None:
        self._engine = engine
        self._sampler = sampler
        self._settings = settings
        self._analyzer = LivenessAnalyzer(settings)
        self._limiter = limiter or Semaphore(max(1, settings.max_concurrent_inferences))

    def with_thresholds(self, overrides: ThresholdOverrides | None) -> VerificationService:
        """A view of this service judging by the caller's thresholds.

        A new instance over the *same* engine, sampler and semaphore: the model
        is expensive and process-wide, and the concurrency limit exists to
        protect the CPU, so a per-request copy of either would defeat both.
        """
        if overrides is None or overrides.is_empty():
            return self

        return VerificationService(
            engine=self._engine,
            sampler=self._sampler,
            settings=self._settings.model_copy(update=overrides.as_update()),
            limiter=self._limiter,
        )

    async def verify(
        self,
        references: list[ReferenceUpload],
        video_path: Path,
        challenge: Challenge | None = None,
        include_best_frame: bool = False,
    ) -> VerificationResponse:
        """Run the (CPU-bound) pipeline in a worker thread, bounded by a semaphore."""
        async with self._limiter:
            return await to_thread.run_sync(
                self._run, references, video_path, challenge, include_best_frame
            )

    async def verify_image(
        self,
        references: list[ReferenceUpload],
        capture: bytes,
        include_capture: bool = False,
    ) -> ImageVerificationResponse:
        """Compare one still against the reference set.

        The same worker-thread and semaphore discipline as `verify`: the work is
        smaller — one detection instead of twenty-four — but it is the same
        CPU-bound model, and letting stills bypass the limiter would let them
        starve the clips sharing the process.
        """
        async with self._limiter:
            return await to_thread.run_sync(
                self._run_image, references, capture, include_capture
            )

    # -- pipeline -------------------------------------------------------------

    def _run(
        self,
        references: list[ReferenceUpload],
        video_path: Path,
        challenge: Challenge | None = None,
        include_best_frame: bool = False,
    ) -> VerificationResponse:
        started = time.perf_counter()
        warnings: list[str] = []

        accepted, reports = self._embed_references(references)
        if len(accepted) < self._settings.min_reference_images:
            raise ReferenceFacesUnusableError(
                "Not enough reference photos contain a usable face "
                f"({len(accepted)}/{self._settings.min_reference_images} accepted).",
                details={
                    "accepted": len(accepted),
                    "required": self._settings.min_reference_images,
                    "images": [report.model_dump() for report in reports],
                },
            )

        cohesion = self._apply_cohesion(accepted, reports)
        consistent = (
            cohesion is None or cohesion >= self._settings.reference_cohesion_threshold
        )
        if not consistent:
            warnings.append(
                "The reference photos do not look like the same person "
                f"(cohesion {cohesion:.2f} < {self._settings.reference_cohesion_threshold:.2f})."
            )
        if len(accepted) < len(references):
            warnings.append(
                f"{len(references) - len(accepted)} reference photo(s) were skipped; "
                "see reference.images for the reason."
            )

        # A challenge needs a denser sample (a blink is ~0.2s), and the frames are
        # decoded and detected once for both checks.
        sampled = self._sampler.sample(
            video_path,
            max_frames=(
                self._settings.liveness_max_sampled_frames if challenge is not None else None
            ),
        )
        reference_matrix = np.vstack([ref.face.embedding for ref in accepted])
        frame_reports, similarity_rows, multi_face_frames, metrics = self._score_frames(
            sampled.frames, reference_matrix
        )

        if multi_face_frames:
            warnings.append(
                f"More than one face appeared in {multi_face_frames} sampled frame(s); "
                "the closest match was used for each frame."
            )

        self._apply_per_reference_stats(accepted, reports, similarity_rows)

        video_report, score, decision, confidence = self._decide(
            sampled, frame_reports, warnings
        )

        if include_best_frame:
            self._attach_best_frame(video_report, sampled.frames)

        liveness: LivenessReport | None = None
        if challenge is not None:
            outcome = self._analyzer.analyze(
                metrics,
                challenge.actions,
                frames_sampled=len(frame_reports),
                challenge_id=challenge.challenge_id,
            )
            liveness = LivenessReport(**asdict(outcome))
            warnings.extend(outcome.warnings)
            if not outcome.live:
                failed = [item.action for item in outcome.actions if not item.performed]
                warnings.append(
                    "Liveness challenge failed; actions not observed: " + ", ".join(failed) + "."
                )

        return VerificationResponse(
            passed=decision is Decision.MATCH and (liveness is None or liveness.live),
            match=decision is Decision.MATCH,
            decision=decision,
            score=round(score, 4),
            score_percent=round(max(0.0, score) * 100.0, 2),
            confidence=round(confidence, 4),
            thresholds=Thresholds(
                match_threshold=self._settings.match_threshold,
                frame_match_threshold=self._settings.frame_match_threshold,
                min_match_ratio=self._settings.min_match_ratio,
                inconclusive_band=self._settings.inconclusive_band,
            ),
            reference=ReferenceReport(
                submitted=len(references),
                accepted=len(accepted),
                rejected=len(references) - len(accepted),
                cohesion=round(cohesion, 4) if cohesion is not None else None,
                consistent=consistent,
                images=reports,
            ),
            video=video_report,
            liveness=liveness,
            warnings=warnings,
            model=self._engine.name,
            processing_ms=int((time.perf_counter() - started) * 1000),
        )

    def _run_image(
        self,
        references: list[ReferenceUpload],
        capture: bytes,
        include_capture: bool = False,
    ) -> ImageVerificationResponse:
        """The still-image pipeline.

        The reference half is shared with `_run` verbatim — the same embedding,
        the same cohesion check, the same refusal when too few photos hold a
        usable face. Only the probe differs, and with it the aggregation: a clip
        is scored by averaging its best frames, and a still has one.
        """
        started = time.perf_counter()
        warnings: list[str] = []

        accepted, reports = self._embed_references(references)
        if len(accepted) < self._settings.min_reference_images:
            raise ReferenceFacesUnusableError(
                "Not enough reference photos contain a usable face "
                f"({len(accepted)}/{self._settings.min_reference_images} accepted).",
                details={
                    "accepted": len(accepted),
                    "required": self._settings.min_reference_images,
                    "images": [report.model_dump() for report in reports],
                },
            )

        cohesion = self._apply_cohesion(accepted, reports)
        consistent = (
            cohesion is None or cohesion >= self._settings.reference_cohesion_threshold
        )
        if not consistent:
            warnings.append(
                "The reference photos do not look like the same person "
                f"(cohesion {cohesion:.2f} < {self._settings.reference_cohesion_threshold:.2f})."
            )
        if len(accepted) < len(references):
            warnings.append(
                f"{len(references) - len(accepted)} reference photo(s) were skipped; "
                "see reference.images for the reason."
            )

        image = decode_image(
            capture, max_side=self._settings.image_max_side, label="Capture"
        )
        faces = self._engine.detect(image)

        if not faces:
            # Refused rather than scored as `no_match`. Nothing was compared, and
            # answering "that is not you" about a picture with no face in it is a
            # different — and wrong — statement. The caller needs to tell the
            # person to stand closer and try again, not that they failed.
            raise NoFaceDetectedError(
                "No usable face was found in the capture.",
                details={
                    "min_det_score": self._settings.min_det_score,
                    "min_face_pixels": self._settings.min_face_pixels,
                },
            )

        reference_matrix = np.vstack([ref.face.embedding for ref in accepted])

        # Pick the face closest to any reference rather than the largest, for the
        # same reason the clip does: somebody walking past behind the person
        # being checked must not be the one that gets scored.
        embeddings = np.vstack([face.embedding for face in faces])
        similarity_grid = np.clip(embeddings @ reference_matrix.T, -1.0, 1.0)
        face_position = int(np.argmax(similarity_grid.max(axis=1)))
        row = similarity_grid[face_position]
        face = faces[face_position]

        if len(faces) > 1:
            warnings.append(
                f"{len(faces)} faces appeared in the capture; the closest match was used. "
                "Ask for a picture with nobody else in frame."
            )

        best_reference = int(np.argmax(row))
        best_similarity = float(row[best_reference])
        matched_references = int(
            np.count_nonzero(row >= self._settings.frame_match_threshold)
        )

        # Written back per reference photo, so an enrolment holding one unusable
        # angle is visible as such instead of showing up as an employee who
        # intermittently fails to be themselves.
        by_index = {report.index: report for report in reports}
        for position, ref in enumerate(accepted):
            by_index[ref.index].capture_similarity = round(float(row[position]), 4)

        score = self._aggregate_references(row)
        decision = self._decide_image(score, matched_references, warnings)

        capture_report = CaptureReport(
            face_detected=True,
            faces_found=len(faces),
            detection_score=round(face.det_score, 4),
            bbox=to_bbox(face),
            face_pixels=face.shortest_side,
            similarity=round(best_similarity, 4),
            matched=best_similarity >= self._settings.frame_match_threshold,
            best_reference_index=best_reference,
            matched_references=matched_references,
        )

        if include_capture:
            encoded = encode_jpeg(
                image,
                max_side=self._settings.best_frame_max_side,
                quality=self._settings.best_frame_quality,
            )
            if encoded is not None:
                capture_report = capture_report.model_copy(
                    update={"image_base64": encoded}
                )

        return ImageVerificationResponse(
            passed=decision is Decision.MATCH,
            match=decision is Decision.MATCH,
            decision=decision,
            score=round(score, 4),
            score_percent=round(max(0.0, score) * 100.0, 2),
            confidence=round(self._confidence(score, decision), 4),
            thresholds=ImageThresholds(
                match_threshold=self._settings.match_threshold,
                reference_match_threshold=self._settings.frame_match_threshold,
                min_reference_matches=self._settings.min_reference_matches,
                top_k_references=self._settings.image_top_k_references,
                inconclusive_band=self._settings.inconclusive_band,
            ),
            reference=ReferenceReport(
                submitted=len(references),
                accepted=len(accepted),
                rejected=len(references) - len(accepted),
                cohesion=round(cohesion, 4) if cohesion is not None else None,
                consistent=consistent,
                images=reports,
            ),
            capture=capture_report,
            warnings=warnings,
            model=self._engine.name,
            processing_ms=int((time.perf_counter() - started) * 1000),
        )

    def _aggregate_references(self, similarities: np.ndarray) -> float:
        """Mean of the best K reference similarities.

        The still-image counterpart of `_aggregate`. A clip averages its best
        frames because most frames are blurred or badly angled; a still has one
        frame, so the spread that has to be averaged out is the reference set's
        instead — enrolment photos differ in light, age and angle just as much.
        """
        if similarities.size == 0:
            return 0.0

        top_k = max(1, min(self._settings.image_top_k_references, int(similarities.size)))
        best = np.sort(similarities)[::-1][:top_k]

        return float(best.mean())

    def _decide_image(
        self, score: float, matched_references: int, warnings: list[str]
    ) -> Decision:
        """Turn a score and a count of agreeing references into a verdict.

        Two conditions rather than one, mirroring the clip's score-plus-ratio
        rule: the mean has to clear the threshold *and* enough individual photos
        have to agree, so a high average resting on a single outlier is not a
        match. Between them sits `inconclusive`, which is a question for a person
        rather than an accusation.
        """
        threshold = self._settings.match_threshold

        if score >= threshold and matched_references >= self._settings.min_reference_matches:
            return Decision.MATCH

        if abs(score - threshold) <= self._settings.inconclusive_band:
            return Decision.INCONCLUSIVE

        if score >= threshold:
            warnings.append(
                f"Only {matched_references} reference photo(s) matched individually "
                f"(minimum {self._settings.min_reference_matches}); the capture may be "
                "at an unusual angle or badly lit."
            )
            return Decision.INCONCLUSIVE

        return Decision.NO_MATCH

    # -- steps ----------------------------------------------------------------

    def _embed_references(
        self, references: list[ReferenceUpload]
    ) -> tuple[list[_Reference], list[ReferenceImageReport]]:
        accepted: list[_Reference] = []
        reports: list[ReferenceImageReport] = []

        for upload in references:
            label = f"Reference image #{upload.index + 1}"
            image = decode_image(
                upload.payload, max_side=self._settings.image_max_side, label=label
            )
            faces = self._engine.detect(image)
            if not faces:
                reports.append(
                    ReferenceImageReport(
                        index=upload.index,
                        filename=upload.filename,
                        face_detected=False,
                        skip_reason="no_face_detected",
                    )
                )
                continue

            face = faces[0]
            reports.append(
                ReferenceImageReport(
                    index=upload.index,
                    filename=upload.filename,
                    face_detected=True,
                    detection_score=round(face.det_score, 4),
                    bbox=to_bbox(face),
                    skip_reason=(
                        "multiple_faces_largest_used" if len(faces) > 1 else None
                    ),
                )
            )
            accepted.append(
                _Reference(index=upload.index, filename=upload.filename, face=face)
            )

        return accepted, reports

    def _apply_cohesion(
        self, accepted: list[_Reference], reports: list[ReferenceImageReport]
    ) -> float | None:
        """Mean pairwise similarity between reference photos, written back per image."""
        if len(accepted) < 2:
            return None

        matrix = np.vstack([ref.face.embedding for ref in accepted])
        gram = np.clip(matrix @ matrix.T, -1.0, 1.0)
        np.fill_diagonal(gram, np.nan)
        per_image = np.nanmean(gram, axis=1)

        by_index = {report.index: report for report in reports}
        for ref, value in zip(accepted, per_image, strict=True):
            by_index[ref.index].similarity_to_group = round(float(value), 4)

        return float(np.nanmean(gram))

    def _score_frames(
        self, frames: list[SampledFrame], reference_matrix: np.ndarray
    ) -> tuple[list[FrameReport], list[np.ndarray], int, list[FaceMetrics]]:
        reports: list[FrameReport] = []
        similarity_rows: list[np.ndarray] = []
        metrics: list[FaceMetrics] = []
        multi_face_frames = 0

        for frame in frames:
            faces = self._engine.detect(frame.image)
            if not faces:
                reports.append(
                    FrameReport(
                        frame_index=frame.index,
                        timestamp_seconds=frame.timestamp_seconds,
                        face_detected=False,
                    )
                )
                continue

            if len(faces) > 1:
                multi_face_frames += 1

            # Pick the face closest to any reference, not just the biggest one:
            # a bystander must not sink the score.
            embeddings = np.vstack([face.embedding for face in faces])
            sims = np.clip(embeddings @ reference_matrix.T, -1.0, 1.0)
            face_position = int(np.argmax(sims.max(axis=1)))
            row = sims[face_position]
            best_reference = int(np.argmax(row))
            similarity = float(row[best_reference])

            # Liveness is measured on the same face the score came from, so a
            # bystander cannot perform the actions on someone else's behalf.
            entry = metrics_from_face(faces[face_position], frame)
            if entry is not None:
                metrics.append(entry)

            similarity_rows.append(row)
            reports.append(
                FrameReport(
                    frame_index=frame.index,
                    timestamp_seconds=frame.timestamp_seconds,
                    face_detected=True,
                    detection_score=round(faces[face_position].det_score, 4),
                    bbox=to_bbox(faces[face_position]),
                    similarity=round(similarity, 4),
                    matched=similarity >= self._settings.frame_match_threshold,
                    best_reference_index=best_reference,
                )
            )

        return reports, similarity_rows, multi_face_frames, metrics

    def _apply_per_reference_stats(
        self,
        accepted: list[_Reference],
        reports: list[ReferenceImageReport],
        similarity_rows: list[np.ndarray],
    ) -> None:
        if not similarity_rows:
            return

        matrix = np.vstack(similarity_rows)  # frames x references
        by_index = {report.index: report for report in reports}
        for position, ref in enumerate(accepted):
            column = matrix[:, position]
            report = by_index[ref.index]
            report.best_video_similarity = round(float(column.max()), 4)
            report.mean_video_similarity = round(float(column.mean()), 4)

    def _decide(
        self,
        sampled,
        frame_reports: list[FrameReport],
        warnings: list[str],
    ) -> tuple[VideoReport, float, Decision, float]:
        with_face = [report for report in frame_reports if report.face_detected]
        similarities = [report.similarity or -1.0 for report in with_face]
        matched = [report for report in with_face if report.matched]
        match_ratio = len(matched) / len(with_face) if with_face else 0.0

        best_frame = max(with_face, key=lambda r: r.similarity or -1.0, default=None)
        score = self._aggregate(similarities)

        if len(with_face) < self._settings.min_frames_with_face:
            decision = Decision.INCONCLUSIVE
            warnings.append(
                f"Only {len(with_face)} of {len(frame_reports)} sampled frames contained a "
                f"face (minimum {self._settings.min_frames_with_face}). Ask the user to "
                "re-record with the face centred and well lit."
            )
        elif (
            score >= self._settings.match_threshold
            and match_ratio >= self._settings.min_match_ratio
        ):
            decision = Decision.MATCH
        elif abs(score - self._settings.match_threshold) <= self._settings.inconclusive_band:
            decision = Decision.INCONCLUSIVE
        elif score >= self._settings.match_threshold:
            decision = Decision.INCONCLUSIVE
            warnings.append(
                f"Only {match_ratio:.0%} of the frames matched (minimum "
                f"{self._settings.min_match_ratio:.0%}); the person may have left the frame."
            )
        else:
            decision = Decision.NO_MATCH

        video_report = VideoReport(
            duration_seconds=sampled.duration_seconds,
            fps=round(sampled.fps, 2) if sampled.fps else None,
            total_frames=sampled.total_frames,
            frames_sampled=len(frame_reports),
            frames_with_face=len(with_face),
            frames_matched=len(matched),
            match_ratio=round(match_ratio, 4),
            best_frame=best_frame,
            frames=frame_reports,
        )
        return video_report, score, decision, self._confidence(score, decision)

    def _attach_best_frame(self, report: VideoReport, frames: list[SampledFrame]) -> None:
        """Put the frame the score came from into the response.

        A copy rather than the object already in `frames`: `best_frame` is one of
        the per-frame reports, so setting the image on it in place would repeat a
        base64 JPEG inside the full frame list as well.
        """
        best = report.best_frame

        if best is None:
            return

        source = next((frame for frame in frames if frame.index == best.frame_index), None)

        if source is None:
            return

        encoded = encode_jpeg(
            source.image,
            max_side=self._settings.best_frame_max_side,
            quality=self._settings.best_frame_quality,
        )

        if encoded is not None:
            report.best_frame = best.model_copy(update={"image_base64": encoded})

    def _aggregate(self, similarities: list[float]) -> float:
        """Mean of the best K frames — robust to blurred or badly angled frames."""
        if not similarities:
            return 0.0
        top_k = max(
            self._settings.top_k_min_frames,
            math.ceil(self._settings.top_k_ratio * len(similarities)),
        )
        top_k = min(top_k, len(similarities))
        best = sorted(similarities, reverse=True)[:top_k]
        return float(sum(best) / len(best))

    def _confidence(self, score: float, decision: Decision) -> float:
        probability = 1.0 / (1.0 + math.exp(-12.0 * (score - self._settings.match_threshold)))
        if decision is Decision.MATCH:
            return probability
        if decision is Decision.NO_MATCH:
            return 1.0 - probability
        # Inconclusive: certainty collapses to 0 exactly at the threshold.
        return abs(2.0 * probability - 1.0)


def to_bbox(face: DetectedFace) -> BoundingBox:
    x1, y1, x2, y2 = face.bbox
    return BoundingBox(x=x1, y=y1, width=max(0, x2 - x1), height=max(0, y2 - y1))
