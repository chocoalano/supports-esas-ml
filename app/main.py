"""Application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app.api.deps import get_challenge_service, get_face_engine, get_stt_engine
from app.api.routes import health, liveness, speech, verify
from app.core.config import RuntimeRole, Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.services.speech.providers import stt_provisioning_problem

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

FACE_DESCRIPTION = (
    "Stateless face-similarity API. Send 5 reference photos plus either a "
    "captured still (`/face/verify-image`) or a recorded video clip "
    "(`/face/verify`); the response reports how closely the person in it "
    "matches the photos. Nothing is persisted.\n\n"
    "Liveness is a property of a clip and never of a still: `/face/verify` "
    "can check a signed challenge across frames, and `/face/verify-image` "
    "reports no liveness at all. A caller sending a photograph owns that "
    "question and has to answer it some other way."
)

SPEECH_DESCRIPTION = (
    "Speech: `/speech/transcribe` turns a recording into text and "
    "`/speech/synthesize` turns text into MP3. Voices are this service's own "
    "aliases, listed by `/speech/voices`; no provider's identifiers are part of "
    "the contract. Nothing is persisted."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    if settings.serves_face:
        # Instantiated here so a missing FSA_CHALLENGE_SECRET is warned about at
        # boot, not on the first challenge request. Only where Face is served: a
        # Speech runtime has no challenges to sign.
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

    # `FSA_WARM_UP_ON_STARTUP` defaults to true, so without the role check a
    # Speech runtime would load InsightFace - and OpenCV with it - into the
    # process that holds PyAV: two FFmpeg builds in one address space, which is
    # one of the things running Speech apart exists to prevent.
    if settings.serves_face and settings.warm_up_on_startup:
        try:
            get_face_engine().load()
        except Exception:  # pragma: no cover - startup must not hard-fail
            logger.exception("Face engine warm-up failed; it will retry on first request.")

    if settings.serves_speech:
        _start_speech(settings)
    elif settings.stt_enabled or settings.tts_enabled:
        logger.warning(
            "FSA_STT_ENABLED/FSA_TTS_ENABLED are set, but FSA_RUNTIME_ROLE=%s mounts no "
            "speech endpoints, so they have no effect here. Speech is served by its own "
            "process with FSA_RUNTIME_ROLE=speech.",
            settings.runtime_role.value,
        )
    yield


def _start_speech(settings: Settings) -> None:
    """Say at boot what would otherwise surface as a 503 on the first request.

    Warnings, not failures: invalid Speech values already stopped the boot when
    `Settings` was built. What is left here is legal but probably not meant.
    """
    if settings.runtime_role is RuntimeRole.ALL:
        logger.warning(
            "FSA_RUNTIME_ROLE=all serves Face and Speech in one process. That is for "
            "development and tests; in production each runs as its own process, so "
            "that Speech cannot take attendance down with it."
        )

    if not settings.stt_enabled and not settings.tts_enabled:
        logger.warning(
            "FSA_RUNTIME_ROLE=%s mounts the speech endpoints, but FSA_STT_ENABLED and "
            "FSA_TTS_ENABLED are both off: every speech request will be refused with 503.",
            settings.runtime_role.value,
        )

    if settings.stt_enabled and settings.stt_warm_up_on_startup:
        try:
            get_stt_engine().load()
        except Exception:  # pragma: no cover - startup must not hard-fail
            logger.exception("Speech-to-text warm-up failed; it will retry on first request.")
    elif settings.stt_enabled and (problem := stt_provisioning_problem(settings)):
        # A disk lookup, not a model load. Logged rather than fatal: TTS in the
        # same runtime does not need the model, and every transcription will
        # say what is missing with a fast 503 rather than a download.
        logger.error(
            "Speech-to-text is enabled but cannot load as configured: %s. Transcriptions "
            "will be refused with 503 until this is fixed.",
            problem,
        )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="\n\n".join(
            part
            for part, served in (
                (FACE_DESCRIPTION, settings.serves_face),
                (SPEECH_DESCRIPTION, settings.serves_speech),
            )
            if served
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

    # The role is enforced here, and not only in nginx. A route that is not
    # mounted cannot be reached by a routing mistake: a Face request that lands
    # on the Speech runtime is a 404, not a model loading.
    if settings.serves_face:
        app.include_router(verify.router, prefix=settings.api_prefix)
        app.include_router(liveness.router, prefix=settings.api_prefix)
        _mount_demo(app)

    if settings.serves_speech:
        app.include_router(speech.router, prefix=settings.api_prefix)

    return app


def _mount_demo(app: FastAPI) -> None:
    """The Face playground. It calls /face/*, so it exists where those do."""

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/demo")

    @app.get("/demo", include_in_schema=False)
    async def demo() -> FileResponse:
        """Browser playground: pick 5 photos, record a clip, see the score."""
        return FileResponse(STATIC_DIR / "index.html")


app = create_app()
