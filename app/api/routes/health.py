"""Liveness / readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import FaceEngineDep, SettingsDep
from app.schemas.verification import HealthResponse

router = APIRouter(tags=["health"])

# Deliberately not behind `GuardDep`. A readiness probe is what a load balancer
# and a container orchestrator call, neither of which carries an API key, and
# what it discloses - up, and whether the model finished loading - is what those
# callers already learn from the fact that it answers at all.


@router.get("/health", response_model=HealthResponse, summary="Service health")
async def health(settings: SettingsDep, engine: FaceEngineDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        model=engine.name,
        model_loaded=engine.is_loaded(),
    )
