"""Application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app.api.deps import get_challenge_service, get_face_engine
from app.api.routes import health, liveness, verify
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # Instantiated here so a missing FSA_CHALLENGE_SECRET is warned about at boot,
    # not on the first challenge request.
    get_challenge_service()

    if not settings.api_key_map:
        if settings.debug:
            logger.warning(
                "FSA_API_KEYS is not set and FSA_DEBUG is on: every request is admitted. "
                "This is the local-development path and must not be deployed."
            )
        else:
            logger.error(
                "FSA_API_KEYS is not set. Every request will be refused with 503 until "
                "it is. This service performs face recognition on whatever it is sent, "
                "so it does not open itself by default."
            )
    if malformed := settings.malformed_api_keys:
        # Dropped rather than admitted as an empty password, which is the safe
        # direction and the silent one. Named here so a key that was meant to
        # work shows up as a misconfiguration rather than as a 401 nobody can
        # explain. The labels, never the values: this is a log.
        logger.error(
            "FSA_API_KEYS has %d entr%s with no usable key, which will be ignored: %s",
            len(malformed),
            "y" if len(malformed) == 1 else "ies",
            ", ".join(malformed),
        )

    if settings.warm_up_on_startup:
        try:
            get_face_engine().load()
        except Exception:  # pragma: no cover - startup must not hard-fail
            logger.exception("Face engine warm-up failed; it will retry on first request.")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Stateless face-similarity API. Send 5 reference photos plus either a "
            "captured still (`/face/verify-image`) or a recorded video clip "
            "(`/face/verify`); the response reports how closely the person in it "
            "matches the photos. Nothing is persisted.\n\n"
            "Liveness is a property of a clip and never of a still: `/face/verify` "
            "can check a signed challenge across frames, and `/face/verify-image` "
            "reports no liveness at all. A caller sending a photograph owns that "
            "question and has to answer it some other way."
        ),
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(verify.router, prefix=settings.api_prefix)
    app.include_router(liveness.router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/demo")

    @app.get("/demo", include_in_schema=False)
    async def demo() -> FileResponse:
        """Browser playground: pick 5 photos, record a clip, see the score."""
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
