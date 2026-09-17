"""Dependency wiring.

Heavy objects (the ONNX session, the frame sampler, the concurrency limiter) are
process-wide singletons; the service that ties them together is resolved through
`Depends` so tests can override any single piece.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from anyio import Semaphore
from fastapi import Depends, Header

from app.api.security import require_api_key
from app.api.throttle import FixedWindowLimiter
from app.core.config import Settings, get_settings
from app.services.challenge import ChallengeService
from app.services.face_engine import FaceEngine, InsightFaceEngine
from app.services.liveness import LivenessService
from app.services.verification import VerificationService
from app.services.video import FrameSampler, OpenCVFrameSampler

SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def get_face_engine() -> FaceEngine:
    return InsightFaceEngine(get_settings())


@lru_cache
def get_frame_sampler() -> FrameSampler:
    return OpenCVFrameSampler(get_settings())


@lru_cache
def get_inference_limiter() -> Semaphore:
    return Semaphore(max(1, get_settings().max_concurrent_inferences))


@lru_cache
def limiter_for(requests_per_minute: int) -> FixedWindowLimiter:
    """One limiter per configured rate, shared by every request that uses it.

    Keyed on the number rather than being a plain singleton so a test - or a
    process reconfigured between runs - gets a limiter that matches its own
    settings instead of the first ones ever seen.
    """
    return FixedWindowLimiter(requests_per_minute)


#: The workspace a request is being made for, when the caller says.
#:
#: One installation of a multi-tenant application is one API key, so counting
#: per key counts every company as one caller: the first company to clock in at
#: 08:00 spends the whole allowance and every other company is told the face
#: service is unavailable. The caller is the only party that knows which
#: workspace a request belongs to, so it travels in a header and the limit is
#: counted per key *and* workspace.
#:
#: It is not access control and is not treated as any: an unknown value buys a
#: caller its own bucket inside its own key's ceiling and nothing else.
TENANT_HEADER = "X-Tenant"


def guard(
    settings: SettingsDep,
    caller: Annotated[str | None, Depends(require_api_key)],
    x_tenant: Annotated[str | None, Header(alias=TENANT_HEADER)] = None,
) -> str:
    """Admit a request: known key first, then within its rate.

    One dependency rather than two on every route, because the order matters and
    a route that got it the other way round would let an unknown caller spend a
    known one's allowance.
    """
    identity = caller or "anonymous"
    limiter_for(settings.rate_limit_per_minute).hit(scope(identity, x_tenant))

    return identity


def scope(caller: str, tenant: str | None) -> str:
    """The bucket a request is counted in.

    Trimmed and truncated because it comes off the wire: an unbounded header
    value would let a caller mint unbounded buckets inside a dict this process
    keeps in memory.
    """
    workspace = (tenant or "").strip()[:64]

    return f"{caller}|{workspace}" if workspace else caller


#: Every route that costs CPU depends on this. It is the whole of the service's
#: access control, so a new route without it is a new open door.
GuardDep = Annotated[str, Depends(guard)]


FaceEngineDep = Annotated[FaceEngine, Depends(get_face_engine)]
FrameSamplerDep = Annotated[FrameSampler, Depends(get_frame_sampler)]


def get_verification_service(
    engine: FaceEngineDep,
    sampler: FrameSamplerDep,
    settings: SettingsDep,
) -> VerificationService:
    return VerificationService(
        engine=engine,
        sampler=sampler,
        settings=settings,
        limiter=get_inference_limiter(),
    )


VerificationServiceDep = Annotated[VerificationService, Depends(get_verification_service)]


@lru_cache
def get_challenge_service() -> ChallengeService:
    """Cached: it holds the HMAC key, which must not be regenerated per request."""
    return ChallengeService(get_settings())


def get_liveness_service(
    engine: FaceEngineDep,
    sampler: FrameSamplerDep,
    settings: SettingsDep,
) -> LivenessService:
    return LivenessService(
        engine=engine,
        sampler=sampler,
        settings=settings,
        limiter=get_inference_limiter(),
    )


ChallengeServiceDep = Annotated[ChallengeService, Depends(get_challenge_service)]
LivenessServiceDep = Annotated[LivenessService, Depends(get_liveness_service)]
