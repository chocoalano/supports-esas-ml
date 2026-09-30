"""Application settings, loaded from environment / .env file."""

from __future__ import annotations

import json
import re
from enum import Enum
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class RuntimeRole(str, Enum):
    """Which workloads this process serves.

    One enum rather than a boolean per workload: two booleans have a state that
    means nothing (both off) and a state nobody meant (face on in the speech
    runtime), and neither says so. Three named values cover the only three
    deployments there are.
    """

    #: Face + Liveness only. The default, so an existing deployment that has
    #: never heard of this setting keeps doing exactly what it did.
    FACE = "face"
    #: Speech only. Face and Liveness are not mounted at all, so a request
    #: routed here by mistake is a 404 - not InsightFace and OpenCV loading
    #: into the process that already holds PyAV (two FFmpeg builds, one
    #: address space).
    SPEECH = "speech"
    #: Everything in one process. Development and tests only.
    ALL = "all"


#: The public contract for TTS `rate` and `volume`: a sign, one to three
#: digits, a percent sign. The value is spliced into an SSML attribute by the
#: Edge provider, so it is parsed into an integer here and only that integer
#: travels further - nothing a caller typed ever reaches a provider verbatim.
PROSODY_PATTERN = re.compile(r"[+-]\d{1,3}%")
#: Narrower than what providers accept, on purpose. The contract is what every
#: future provider has to honour, and "three times faster" is not a feature
#: anybody asked for.
PROSODY_RANGE = (-100, 100)

#: `stt_device` values faster-whisper understands.
STT_DEVICES = ("cpu", "cuda", "auto")
#: What `/speech/voices` may say about a voice. Closed, so a client can switch on it.
VOICE_GENDERS = ("male", "female", "neutral")

# Always used with `fullmatch`: `$` also matches before a trailing newline, so
# `re.match("^...$")` accepts "+0%\n" - which then reaches `int()`, or a header.
_VOICE_ALIAS_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
LANGUAGE_CODE = re.compile(r"[a-z]{2,3}")


class SpeechConfigError(RuntimeError):
    """The Speech settings of a Speech runtime cannot be served. Raised at boot.

    Deliberately not a `ValueError`. pydantic wraps a `ValueError` from a model
    validator in a `ValidationError` whose message repeats the whole settings
    input, truncated in the middle - and which keys land at either end of that
    repr depends on field order and on pydantic's truncation width. That input
    holds `FSA_API_KEYS` and `FSA_CHALLENGE_SECRET`, and this message goes to
    the boot log. Any other exception type propagates untouched, carrying only
    the text below, which names env vars and never their neighbours' values.
    """


def parse_prosody_percent(value: str) -> int | None:
    """`"+10%"` -> 10, or None when it is not a contract-conforming value.

    Syntax and range are both checked: the pattern alone accepts `+999%`.
    """
    if not PROSODY_PATTERN.fullmatch(value):
        return None

    number = int(value[:-1])
    low, high = PROSODY_RANGE

    return number if low <= number <= high else None


class VoiceAlias(BaseModel):
    """One voice a caller may ask for, by a name that belongs to this service.

    `provider_voice` is the only provider-specific thing in it and it never
    leaves the service: not in `/speech/voices`, not in a response header, not
    in an error. Changing providers is then a change to this list, and the
    callers - which only ever held `id` - do not notice.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    provider_voice: str
    label: str
    language: str
    gender: str


def _default_voice_aliases() -> list[VoiceAlias]:
    return [
        VoiceAlias(
            id="default",
            provider_voice="id-ID-ArdiNeural",
            label="Bahasa Indonesia - Default",
            language="id-ID",
            gender="male",
        ),
        VoiceAlias(
            id="id_male",
            provider_voice="id-ID-ArdiNeural",
            label="Bahasa Indonesia - Pria",
            language="id-ID",
            gender="male",
        ),
        VoiceAlias(
            id="id_female",
            provider_voice="id-ID-GadisNeural",
            label="Bahasa Indonesia - Wanita",
            language="id-ID",
            gender="female",
        ),
    ]


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
    #: Which routers this process mounts. See `RuntimeRole`. An unknown value
    #: fails the boot rather than falling back to anything: a runtime that
    #: guessed its own role is worse than one that did not start.
    runtime_role: RuntimeRole = RuntimeRole.FACE

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

    # --- Speech: speech-to-text ------------------------------------------------
    #
    # Namespaced STT_/TTS_ on purpose, unlike the Face settings above: nothing
    # here may ever be read as widening what a Face setting means.
    # `FSA_MAX_CONCURRENT_INFERENCES` stays Face's alone.

    #: Off by default: upgrading a deployment must not start a new workload.
    #: Only consulted when `runtime_role` mounts Speech at all.
    stt_enabled: bool = False
    stt_provider: Literal["faster_whisper"] = "faster_whisper"
    #: A benchmark baseline, not a production decision. The production value is
    #: the smallest model whose Indonesian transcripts are good enough, chosen
    #: from peak RSS, latency and real-time factor measured on the target host.
    stt_model: str = "small"
    #: Where model weights are cached (HuggingFace cache when unset). Set it in
    #: systemd, or `PrivateTmp`/a home-less service user loses the download.
    stt_model_root: str | None = None
    stt_device: str = "cpu"
    #: int8 is the smallest footprint on CPU.
    stt_compute_type: str = "int8"
    #: 0 = the library default. Multiplies with `stt_max_concurrent`:
    #: `stt_max_concurrent * max(1, stt_cpu_threads)` must fit the cores the
    #: Speech runtime is given.
    stt_cpu_threads: int = 0
    #: 1 = greedy: fastest and smallest. Not exposed to callers.
    stt_beam_size: int = 1
    #: Skip silence before inference. Not exposed to callers.
    stt_vad_filter: bool = True
    #: Empty = detect. Whisper regularly hears Indonesian as Malay, so a
    #: deployment that knows its language should say so.
    stt_default_language: str | None = None
    #: Empty = any language. When set, a requested language outside it is
    #: refused before any work, and a detected one after.
    stt_allowed_languages: Annotated[list[str], NoDecode] = Field(default_factory=list)
    #: Off: upgrading must not add memory at boot. The model loads on the first
    #: transcription instead.
    stt_warm_up_on_startup: bool = False
    #: Its own semaphore, never Face's. CPU-bound, so 1 until measured.
    stt_max_concurrent: int = 1
    #: First bound, enforced while the upload streams to disk.
    stt_max_audio_bytes: int = 25 * 1024 * 1024
    #: Second, different bound. Bytes do not bound CPU - 25 MB of Opus is hours
    #: of speech - so length is measured by decoding and the decode stops the
    #: moment it passes this. It is also what keeps a synchronous request under
    #: the gateway timeout: when a clip this long does not finish in time on
    #: the target host, lower this, do not raise the timeout.
    stt_max_audio_seconds: float = 300.0
    #: Declared Content-Type, checked only when the caller sends one. Not a
    #: security boundary: the decoder is.
    stt_allowed_audio_types: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
            "audio/mpeg",
            "audio/mp3",
            "audio/mp4",
            "audio/m4a",
            "audio/x-m4a",
            "audio/aac",
            "audio/ogg",
            "audio/opus",
            "audio/webm",
            "audio/flac",
            "audio/x-flac",
            "audio/3gpp",
            # What HTTP clients declare by file extension - Guzzle, under
            # Laravel's `attach()`, sends `.webm` as video/webm and `.aac` as
            # audio/x-aac - and what they declare when they do not know. None of
            # these is refused by a check that is not an authority anyway.
            "audio/x-aac",
            "video/webm",
            "video/mp4",
            "video/3gpp",
            "application/octet-stream",
        ]
    )

    # --- Speech: text-to-speech -------------------------------------------------
    tts_enabled: bool = False
    tts_provider: Literal["edge"] = "edge"
    #: The alias used when a request names none. Must be one of the ids below.
    tts_default_voice: str = "default"
    #: The voices callers may ask for, as a JSON list of
    #: `{"id", "provider_voice", "label", "language", "gender"}`.
    #:
    #: JSON rather than `alias:voice` CSV because each entry carries five
    #: fields, one of them free text; a delimiter-based format would break on
    #: the first label with a comma in it. This list is also the whole of
    #: `/speech/voices` - there is no network discovery, so what callers see is
    #: exactly what was approved here.
    tts_voice_aliases: list[VoiceAlias] = Field(default_factory=_default_voice_aliases)
    tts_default_rate: str = "+0%"
    tts_default_volume: str = "+0%"
    #: ~3 minutes of speech. Bounds the request without guessing a bitrate.
    tts_max_text_chars: int = 3000
    #: A remote provider's output is not bounded by anything we accepted, so it
    #: is bounded here. Past this the response is an error, never a truncated
    #: MP3.
    tts_max_output_bytes: int = 10 * 1024 * 1024
    #: Wall clock for one synthesis, provider round trip included.
    tts_timeout_seconds: float = 30.0
    #: Network-bound, not CPU-bound, hence higher than STT.
    tts_max_concurrent: int = 4

    @property
    def serves_face(self) -> bool:
        return self.runtime_role in (RuntimeRole.FACE, RuntimeRole.ALL)

    @property
    def serves_speech(self) -> bool:
        return self.runtime_role in (RuntimeRole.SPEECH, RuntimeRole.ALL)

    @property
    def voice_alias_map(self) -> dict[str, VoiceAlias]:
        return {alias.id: alias for alias in self.tts_voice_aliases}

    @field_validator("stt_default_language", "stt_model_root", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """`FSA_STT_DEFAULT_LANGUAGE=` as shipped means "not set", not the empty language."""
        if isinstance(value, str) and not value.strip():
            return None

        return value

    @model_validator(mode="after")
    def _validate_speech(self) -> Settings:
        """Refuse to boot a Speech runtime whose Speech configuration is wrong.

        Only when this process serves Speech. A Face runtime never reads these
        values, and a typo in one must not be able to stop attendance from
        booting - that is the whole reason the two run apart.
        """
        if self.serves_speech and (problems := speech_config_problems(self)):
            raise SpeechConfigError("Invalid speech configuration: " + "; ".join(problems))

        return self

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
        "stt_allowed_languages",
        "stt_allowed_audio_types",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow `FSA_CORS_ORIGINS=a,b` in addition to JSON lists.

        These fields are annotated `NoDecode` because this validator alone
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


def speech_config_problems(settings: Settings) -> list[str]:
    """Every reason the Speech settings cannot be served, named by env var."""
    problems: list[str] = []

    for name in (
        "stt_max_audio_bytes",
        "stt_max_audio_seconds",
        "stt_beam_size",
        "tts_max_text_chars",
        "tts_max_output_bytes",
        "tts_timeout_seconds",
    ):
        if getattr(settings, name) <= 0:
            problems.append(f"FSA_{name.upper()} must be greater than 0")

    if settings.stt_device not in STT_DEVICES:
        problems.append(f"FSA_STT_DEVICE must be one of {', '.join(STT_DEVICES)}")

    bad_languages = [
        code for code in settings.stt_allowed_languages if not LANGUAGE_CODE.fullmatch(code)
    ]
    if bad_languages:
        problems.append(
            "FSA_STT_ALLOWED_LANGUAGES holds codes that are not lowercase ISO 639 "
            f"codes: {', '.join(bad_languages)}"
        )

    default_language = settings.stt_default_language
    if default_language is not None:
        if not LANGUAGE_CODE.fullmatch(default_language):
            problems.append("FSA_STT_DEFAULT_LANGUAGE must be a lowercase ISO 639 code")
        elif settings.stt_allowed_languages and (
            default_language not in settings.stt_allowed_languages
        ):
            problems.append("FSA_STT_DEFAULT_LANGUAGE is not in FSA_STT_ALLOWED_LANGUAGES")

    for name in ("tts_default_rate", "tts_default_volume"):
        if parse_prosody_percent(getattr(settings, name)) is None:
            low, high = PROSODY_RANGE
            problems.append(f"FSA_{name.upper()} must look like +0% and lie in {low}..{high}")

    aliases = settings.tts_voice_aliases
    if not aliases:
        problems.append("FSA_TTS_VOICE_ALIASES must define at least one voice")

    seen: set[str] = set()
    for alias in aliases:
        if not _VOICE_ALIAS_ID.fullmatch(alias.id):
            problems.append(
                f"voice alias id {alias.id!r} must be lowercase letters, digits, '_' or '-'"
            )
        if alias.id in seen:
            problems.append(f"voice alias id {alias.id!r} is defined twice")
        seen.add(alias.id)
        if not alias.provider_voice.strip():
            problems.append(f"voice alias {alias.id!r} has no provider_voice")
        if not alias.label.strip() or not alias.language.strip():
            problems.append(f"voice alias {alias.id!r} needs a label and a language")
        if alias.gender not in VOICE_GENDERS:
            problems.append(
                f"voice alias {alias.id!r} gender must be one of {', '.join(VOICE_GENDERS)}"
            )

    if aliases and settings.tts_default_voice not in seen:
        problems.append(
            f"FSA_TTS_DEFAULT_VOICE={settings.tts_default_voice!r} is not an id in "
            "FSA_TTS_VOICE_ALIASES"
        )

    return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()
