"""Application settings, loaded from environment / .env file."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FSA_",
        extra="ignore",
    )

    # --- App -----------------------------------------------------------------
    app_name: str = "Face Similarity API"
    app_version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    log_level: str = "INFO"

    # --- Access ---------------------------------------------------------------
    #: Shared keys, as `label:key` pairs — the label appears in log lines so a
    #: request can be attributed without the key itself ever being written down.
    #: A bare `key` is accepted and labelled `default`.
    #:
    #: Empty means the service refuses everything, unless `debug` is on. CORS is
    #: not a substitute: it is a rule browsers apply to themselves, and this
    #: service is called by servers.
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    #: Requests allowed per key per minute, counted in this process. Set to 0 to
    #: turn it off. See `app/api/throttle.py` for what a multi-worker
    #: deployment has to do instead.
    rate_limit_per_minute: int = 60

    # --- Face engine ---------------------------------------------------------
    #: InsightFace model pack. `buffalo_l` = ArcFace R100 (best), `buffalo_s` = lighter.
    model_pack: str = "buffalo_l"
    #: Root dir for downloaded ONNX models (~/.insightface by default).
    model_root: str | None = None
    #: -1 = CPU, >= 0 = CUDA device id.
    ctx_id: int = -1
    #: Detector input size (square). Lower = faster, misses small faces.
    det_size: int = 640
    #: Minimum detector confidence for a face to be used at all.
    min_det_score: float = 0.55
    #: Faces smaller than this (in pixels, shortest bbox side) are ignored.
    min_face_pixels: int = 40
    #: Max concurrent inference calls (protects CPU / memory).
    max_concurrent_inferences: int = 2
    #: Load the model at startup instead of on first request.
    warm_up_on_startup: bool = True
    #: Load the 68-point landmark model. Required by the liveness endpoints;
    #: turn it off to save memory if you only need similarity scoring.
    enable_landmarks: bool = True

    # --- Reference images ----------------------------------------------------
    min_reference_images: int = 5
    max_reference_images: int = 5
    max_image_bytes: int = 8 * 1024 * 1024
    #: Downscale reference photos whose longest side exceeds this before detection.
    image_max_side: int = 1600
    allowed_image_types: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    )
    #: Reference photos below this mutual cosine similarity are flagged as
    #: "probably not the same person" (warning only, does not block).
    reference_cohesion_threshold: float = 0.35

    # --- Video ---------------------------------------------------------------
    max_video_bytes: int = 64 * 1024 * 1024
    allowed_video_types: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "video/mp4",
            "video/webm",
            "video/quicktime",
            "video/x-matroska",
            "video/x-msvideo",
            "video/3gpp",
        ]
    )
    #: Number of frames actually fed to the face engine.
    max_sampled_frames: int = 24
    #: Hard cap on frames decoded from the container (protects against long clips).
    max_decoded_frames: int = 1800
    #: Reject clips longer than this (0 disables the check).
    max_video_seconds: float = 60.0
    #: Downscale frames whose longest side exceeds this before detection.
    frame_max_side: int = 1280
    #: Longest side of the scored frame returned to the caller, when asked for.
    #: Small on purpose: it travels inline, base64-encoded, in a JSON response.
    best_frame_max_side: int = 640
    #: JPEG quality of that frame.
    best_frame_quality: int = 85

    # --- Matching ------------------------------------------------------------
    #: Cosine similarity above which a single frame counts as "matched".
    frame_match_threshold: float = 0.38
    #: Cosine similarity the aggregated score must reach to return match=true.
    match_threshold: float = 0.38
    #: Fraction of face-bearing frames that must match.
    min_match_ratio: float = 0.6
    #: Aggregated score is the mean of the best `top_k_ratio` share of frames.
    top_k_ratio: float = 0.3
    top_k_min_frames: int = 3
    #: Scores within +/- this band of `match_threshold` are reported inconclusive.
    inconclusive_band: float = 0.03
    #: Minimum frames containing a face before a verdict is trustworthy.
    min_frames_with_face: int = 3

    # --- Single-image matching ------------------------------------------------
    #: A still has no frames to average over, so the robustness that `top_k_ratio`
    #: buys across a clip has to come from the reference set instead: the score is
    #: the mean of the best `image_top_k_references` of the accepted photos.
    #:
    #: Averaging over all five punishes an enrolment that holds one bad angle;
    #: taking the single best lets one lucky photograph carry a stranger. Three of
    #: five asks a majority to agree and lets the two worst references go.
    image_top_k_references: int = 3
    #: How many individual reference photos must clear `frame_match_threshold` on
    #: their own. The still-image analogue of `min_match_ratio`.
    #:
    #: Equal to `image_top_k_references` on purpose: the score is the mean of the
    #: best three, so requiring three to clear the bar individually means the
    #: same three photos both carry the mean and pass the gate. At two, a
    #: minority carried the majority - similarities of
    #: [0.62, 0.60, 0.05, 0.05, 0.05] score 0.42 against a 0.38 threshold with
    #: two matches, so two photographs of the wrong person in an enrolment set
    #: were enough to clock in as its owner.
    min_reference_matches: int = 3

    # --- Liveness challenge --------------------------------------------------
    #: HMAC key for challenge tokens. MUST be set in production, and must be the
    #: same across every worker/instance, otherwise tokens fail to verify.
    challenge_secret: str | None = None
    #: How long a challenge stays valid. Short is good: it is the window in which
    #: a recording could be replayed.
    challenge_ttl_seconds: int = 120
    #: How many actions a challenge asks for.
    challenge_action_count: int = 2
    #: Pool the actions are drawn from, in order of the catalogue in
    #: `app/services/challenge.py`.
    challenge_actions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "turn_left",
            "turn_right",
            "look_up",
            "look_down",
            "open_mouth",
        ]
    )
    #: Liveness needs a denser sample than similarity: a blink lasts ~0.2s.
    #:
    #: This is the single biggest cost in a verification, and it is close to the
    #: only one. Measured on CPU (buffalo_l, ctx_id=-1) with a real 720x480 clip
    #: and five reference photos:
    #:
    #:     60 frames -> 14.9s      36 frames -> 11.2s
    #:
    #: Clip *length* does not matter at all - a 5s clip and an 11s clip both
    #: sample this many frames and both took 14.9s - so a shorter recording buys
    #: nothing here. Neither does reference photo size: downscaling them from
    #: 1280px to 640px changed nothing measurable. It is ~154ms per frame on top
    #: of a fixed ~5.6s.
    #:
    #: The default stays at 60 because `blink` needs it: a blink is over in
    #: ~0.2s and a sparser sample walks straight past one. A deployment whose
    #: `challenge_actions` leave blink out - head turns and an open mouth are
    #: all held far longer - can safely lower this and get the seconds back.
    #: See FSA_LIVENESS_MAX_SAMPLED_FRAMES in .env.example.
    liveness_max_sampled_frames: int = 60
    liveness_min_frames_with_face: int = 8
    #: Deviation from the person's own neutral pose that counts as a head turn.
    liveness_yaw_delta: float = 0.30
    liveness_pitch_delta: float = 0.22
    #: Eye aspect ratio must fall to this share of its baseline to count as a blink.
    liveness_blink_ratio: float = 0.7
    #: Mouth aspect ratio must rise this much above baseline to count as open.
    liveness_mouth_delta: float = 0.12

    @property
    def api_key_map(self) -> dict[str, str]:
        """Configured keys as label -> key.

        A property rather than a field, so the raw list stays exactly what was
        configured and nothing rewrites it on the way in.

        Entries with no key at all — a stray comma, `label:` with nothing after
        it — are dropped rather than admitted as an empty password. That is the
        safe direction and the silent one, so `malformed_api_keys` names them and
        the application logs them at boot: a key that was meant to work and does
        not should be visible as a misconfiguration, not as a 401 nobody can
        explain.
        """
        mapping: dict[str, str] = {}

        for index, entry in enumerate(self.api_keys):
            label, separator, key = entry.partition(":")
            if separator:
                mapping[label.strip() or f"key-{index}"] = key.strip()
            else:
                mapping[f"default-{index}" if index else "default"] = label.strip()

        return {label: key for label, key in mapping.items() if key}

    @property
    def malformed_api_keys(self) -> list[str]:
        """Configured entries that carry no usable key.

        Returns the labels, never the values: this is written to a log.
        """
        malformed: list[str] = []

        for index, entry in enumerate(self.api_keys):
            label, separator, key = entry.partition(":")
            if separator and not key.strip():
                malformed.append(label.strip() or f"entry #{index + 1}")
            elif not separator and not label.strip():
                malformed.append(f"entry #{index + 1}")

        return malformed

    @field_validator(
        "cors_origins",
        "allowed_image_types",
        "allowed_video_types",
        "challenge_actions",
        "api_keys",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow `FSA_CORS_ORIGINS=a,b` in addition to JSON lists.

        The four fields are annotated `NoDecode` because this validator alone
        is not enough: pydantic-settings JSON-decodes list-typed fields inside
        the settings source, *before* any validator runs. Without `NoDecode`
        the documented comma-separated syntax — and even the `*` shipped in
        `.env.example` — raises `SettingsError` at import time, so the service
        cannot boot with its own example configuration.

        `NoDecode` turns the decoding off for both forms, so the JSON branch is
        parsed here rather than passed through. Leaving it to pydantic instead
        would hand it a `str` where a `list` is declared, which trades one
        unbootable syntax for the other.
        """
        if not isinstance(value, str):
            return value

        text = value.strip()

        if text.startswith("["):
            return json.loads(text)

        return [item.strip() for item in text.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
