"""Settings parsing.

The four list-typed settings accept a comma-separated string, which is what
`.env.example` documents and ships. That syntax is not free: pydantic-settings
JSON-decodes list-typed fields *inside the settings source*, before any
validator runs, so a bare `a,b` — and even the `*` in the shipped example —
raises `SettingsError` while the module is still being imported.

`NoDecode` on those fields is what keeps the raw string reaching `_split_csv`.
Dropping it does not fail loudly in one place; it makes the service refuse to
boot with its own example configuration, and the traceback names pydantic
rather than the line that caused it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings

CSV_FIELDS = {
    "FSA_CORS_ORIGINS": ("cors_origins", ["http://a", "http://b"]),
    "FSA_CHALLENGE_ACTIONS": ("challenge_actions", ["turn_left", "open_mouth"]),
    "FSA_ALLOWED_IMAGE_TYPES": ("allowed_image_types", ["image/jpeg", "image/png"]),
    "FSA_ALLOWED_VIDEO_TYPES": ("allowed_video_types", ["video/mp4", "video/webm"]),
}


def _settings_from(tmp_path: Path, line: str) -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text(line + "\n")
    return Settings(_env_file=str(env_file))


@pytest.mark.parametrize(("key", "expected"), CSV_FIELDS.items(), ids=list(CSV_FIELDS))
def test_reads_comma_separated_lists(
    tmp_path: Path, key: str, expected: tuple[str, list[str]]
) -> None:
    field, values = expected
    settings = _settings_from(tmp_path, f"{key}={','.join(values)}")

    assert getattr(settings, field) == values


def test_reads_the_wildcard_origin_shipped_in_the_example(tmp_path: Path) -> None:
    """`.env.example` ships `FSA_CORS_ORIGINS=*`; it must load."""
    assert _settings_from(tmp_path, "FSA_CORS_ORIGINS=*").cors_origins == ["*"]


def test_still_accepts_json_lists(tmp_path: Path) -> None:
    """The JSON form documented alongside the CSV one keeps working."""
    settings = _settings_from(tmp_path, 'FSA_CORS_ORIGINS=["http://a", "http://b"]')

    assert settings.cors_origins == ["http://a", "http://b"]


def test_the_shipped_example_file_loads(tmp_path: Path) -> None:
    """Whatever else changes, `.env.example` must remain bootable as written."""
    example = Path(__file__).resolve().parent.parent / ".env.example"
    settings = Settings(_env_file=str(example))

    assert settings.cors_origins == ["*"]
    assert "turn_left" in settings.challenge_actions
