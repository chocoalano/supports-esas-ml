"""Unit tests for the scoring maths and frame selection."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import Settings
from app.schemas.verification import Decision
from app.services.face_engine import cosine_similarity, normalise
from app.services.verification import VerificationService
from app.services.video import SampledFrame, evenly_spaced


@pytest.fixture
def service(settings: Settings) -> VerificationService:
    return VerificationService(engine=object(), sampler=object(), settings=settings)


def test_cosine_similarity_is_bounded_and_scale_free():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert cosine_similarity(a, a * 7) == pytest.approx(1.0)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)
    assert cosine_similarity(a, np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)


def test_normalise_leaves_a_zero_vector_alone():
    zero = np.zeros(4, dtype=np.float32)

    assert np.array_equal(normalise(zero), zero)
    assert float(np.linalg.norm(normalise(np.array([3.0, 4.0])))) == pytest.approx(1.0)


def test_aggregate_uses_the_best_frames_only(service: VerificationService):
    """Blurred frames at the tail must not drag a genuine match down."""
    similarities = [0.9, 0.85, 0.8] + [0.05] * 7

    assert service._aggregate(similarities) == pytest.approx(0.85)
    assert service._aggregate([]) == 0.0


def test_aggregate_never_uses_fewer_than_the_minimum(service: VerificationService):
    service._settings.top_k_ratio = 0.01
    service._settings.top_k_min_frames = 3

    assert service._aggregate([1.0, 0.5, 0.0, -1.0]) == pytest.approx(0.5)


def test_confidence_follows_the_decision(service: VerificationService):
    high = service._confidence(0.9, Decision.MATCH)
    low = service._confidence(0.0, Decision.NO_MATCH)
    borderline = service._confidence(service._settings.match_threshold, Decision.INCONCLUSIVE)

    assert high > 0.99
    assert low > 0.98
    assert borderline == pytest.approx(0.0)  # maximum uncertainty at the threshold


def test_evenly_spaced_keeps_the_span_of_the_clip():
    frames = [
        SampledFrame(index=i, timestamp_seconds=None, image=np.zeros((2, 2, 3)))
        for i in range(100)
    ]

    picked = evenly_spaced(frames, 10)

    assert len(picked) == 10
    assert picked[0].index == 0
    assert picked[-1].index == 90
    assert evenly_spaced(frames[:5], 10) == frames[:5]
