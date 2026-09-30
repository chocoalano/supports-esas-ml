"""Speech settings: parsed like the rest, validated only where they are used.

A Speech runtime refuses to boot on Speech settings it cannot serve. A Face
runtime does not look at them at all - a typo in a Speech value must never be
what stops attendance from starting.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import NoDecode

from app.core.config import Settings, SpeechConfigError, parse_prosody_percent

EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


def settings_from(tmp_path: Path, *lines: str) -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(lines) + "\n")
    return Settings(_env_file=str(env_file))


@pytest.fixture(autouse=True)
def _no_role_from_the_environment(monkeypatch):
    """These tests say their role in the file they load."""
    monkeypatch.delenv("FSA_RUNTIME_ROLE", raising=False)


# -- parsing -----------------------------------------------------------------------------


def test_every_no_decode_list_reads_comma_separated_values(tmp_path):
    """The trap `tests/test_settings.py` describes, closed for every field at once.

    A list field marked `NoDecode` but missing from `_split_csv` - or the other
    way round - refuses to boot from its own documented syntax. Rather than a
    list someone has to remember to extend, every such field is found here.
    """
    fields = [name for name, field in Settings.model_fields.items() if NoDecode in field.metadata]
    assert {"stt_allowed_languages", "stt_allowed_audio_types"} <= set(fields)

    for name in fields:
        loaded = settings_from(tmp_path, f"FSA_{name.upper()}=aa,bb")
        assert getattr(loaded, name) == ["aa", "bb"], name


def test_voice_aliases_read_as_json(tmp_path):
    loaded = settings_from(
        tmp_path,
        "FSA_RUNTIME_ROLE=speech",
        "FSA_TTS_DEFAULT_VOICE=narrator",
        'FSA_TTS_VOICE_ALIASES=[{"id":"narrator","provider_voice":"x-Neural",'
        '"label":"Narator, pelan","language":"id-ID","gender":"neutral"}]',
    )

    (alias,) = loaded.tts_voice_aliases
    assert alias.id == "narrator"
    assert alias.provider_voice == "x-Neural"
    assert alias.label == "Narator, pelan"


def test_a_misspelt_voice_alias_field_is_refused(tmp_path):
    with pytest.raises(ValidationError, match="extra"):
        settings_from(
            tmp_path,
            'FSA_TTS_VOICE_ALIASES=[{"id":"a","provider_voice":"x","label":"A",'
            '"language":"id-ID","gender":"male","lable":"typo"}]',
        )


def test_a_blank_default_language_means_unset(tmp_path):
    loaded = settings_from(tmp_path, "FSA_RUNTIME_ROLE=speech", "FSA_STT_DEFAULT_LANGUAGE=")

    assert loaded.stt_default_language is None


@pytest.mark.parametrize("role", ["face", "speech", "all"])
def test_the_shipped_example_loads_under_every_role(monkeypatch, role):
    monkeypatch.setenv("FSA_RUNTIME_ROLE", role)

    loaded = Settings(_env_file=str(EXAMPLE))

    assert loaded.runtime_role.value == role
    assert loaded.stt_enabled is False
    assert loaded.tts_enabled is False
    assert [alias.id for alias in loaded.tts_voice_aliases] == ["default", "id_male", "id_female"]


def test_the_shipped_example_says_face(tmp_path):
    """Copying `.env.example` must never turn a Face deployment into anything else."""
    assert Settings(_env_file=str(EXAMPLE)).runtime_role.value == "face"


# -- validation, where Speech is served ----------------------------------------------


@pytest.mark.parametrize(
    ("line", "named"),
    [
        ("FSA_TTS_DEFAULT_VOICE=nobody", "FSA_TTS_DEFAULT_VOICE"),
        ("FSA_TTS_DEFAULT_RATE=+150%", "FSA_TTS_DEFAULT_RATE"),
        ("FSA_TTS_DEFAULT_VOLUME=loud", "FSA_TTS_DEFAULT_VOLUME"),
        ("FSA_STT_MAX_AUDIO_SECONDS=0", "FSA_STT_MAX_AUDIO_SECONDS"),
        ("FSA_TTS_TIMEOUT_SECONDS=-1", "FSA_TTS_TIMEOUT_SECONDS"),
        ("FSA_STT_DEVICE=tpu", "FSA_STT_DEVICE"),
        ("FSA_STT_ALLOWED_LANGUAGES=id,English", "FSA_STT_ALLOWED_LANGUAGES"),
        ("FSA_STT_DEFAULT_LANGUAGE=Indonesian", "FSA_STT_DEFAULT_LANGUAGE"),
        ("FSA_TTS_VOICE_ALIASES=[]", "FSA_TTS_VOICE_ALIASES"),
    ],
)
@pytest.mark.parametrize("role", ["speech", "all"])
def test_a_speech_runtime_refuses_to_boot_on_bad_speech_settings(tmp_path, role, line, named):
    with pytest.raises(SpeechConfigError, match=named):
        settings_from(tmp_path, f"FSA_RUNTIME_ROLE={role}", line)


def test_a_default_language_outside_the_allow_list_is_refused(tmp_path):
    with pytest.raises(SpeechConfigError, match="not in FSA_STT_ALLOWED_LANGUAGES"):
        settings_from(
            tmp_path,
            "FSA_RUNTIME_ROLE=speech",
            "FSA_STT_DEFAULT_LANGUAGE=fr",
            "FSA_STT_ALLOWED_LANGUAGES=id,en",
        )


@pytest.mark.parametrize(
    ("alias", "complaint"),
    [
        ('{"id":"Default","provider_voice":"x","label":"A","language":"id","gender":"male"}',
         "lowercase"),
        # `$` matches before a trailing newline; this id would end up in a header.
        ('{"id":"a\\n","provider_voice":"x","label":"A","language":"id","gender":"male"}',
         "lowercase"),
        ('{"id":"a","provider_voice":" ","label":"A","language":"id","gender":"male"}',
         "no provider_voice"),
        ('{"id":"a","provider_voice":"x","label":"A","language":"id","gender":"robot"}',
         "gender"),
    ],
)
def test_a_bad_voice_alias_is_refused(tmp_path, alias, complaint):
    with pytest.raises(SpeechConfigError, match=complaint):
        settings_from(
            tmp_path,
            "FSA_RUNTIME_ROLE=speech",
            "FSA_TTS_DEFAULT_VOICE=a",
            f"FSA_TTS_VOICE_ALIASES=[{alias}]",
        )


def test_an_alias_defined_twice_is_refused(tmp_path):
    entry = '{"id":"a","provider_voice":"x","label":"A","language":"id","gender":"male"}'

    with pytest.raises(SpeechConfigError, match="defined twice"):
        settings_from(
            tmp_path,
            "FSA_RUNTIME_ROLE=speech",
            "FSA_TTS_DEFAULT_VOICE=a",
            f"FSA_TTS_VOICE_ALIASES=[{entry},{entry}]",
        )


def test_every_problem_is_reported_at_once(tmp_path):
    """One boot, one list - not one fix-and-restart per mistake."""
    with pytest.raises(SpeechConfigError) as refusal:
        settings_from(
            tmp_path,
            "FSA_RUNTIME_ROLE=speech",
            "FSA_TTS_DEFAULT_VOICE=nobody",
            "FSA_STT_DEVICE=tpu",
        )

    assert "FSA_TTS_DEFAULT_VOICE" in str(refusal.value)
    assert "FSA_STT_DEVICE" in str(refusal.value)


def test_the_boot_error_never_carries_a_secret(tmp_path):
    """pydantic's own error would repeat the settings input, keys included."""
    with pytest.raises(SpeechConfigError) as refusal:
        settings_from(
            tmp_path,
            "FSA_API_KEYS=laravel:the-real-api-key-value",
            "FSA_CHALLENGE_SECRET=the-real-challenge-secret",
            "FSA_RUNTIME_ROLE=all",
            "FSA_TTS_DEFAULT_VOICE=nobody",
        )

    assert "the-real-api-key-value" not in str(refusal.value)
    assert "the-real-challenge-secret" not in str(refusal.value)
    assert "api_keys" not in str(refusal.value)


# -- ...and only there -----------------------------------------------------------------


def test_a_face_runtime_ignores_speech_settings_it_never_uses(tmp_path):
    loaded = settings_from(
        tmp_path,
        "FSA_RUNTIME_ROLE=face",
        "FSA_TTS_DEFAULT_VOICE=nobody",
        "FSA_TTS_DEFAULT_RATE=+150%",
        "FSA_STT_DEVICE=tpu",
    )

    assert loaded.runtime_role.value == "face"


# -- the prosody contract ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("+0%", 0), ("-100%", -100), ("+100%", 100), ("+7%", 7)],
)
def test_prosody_inside_the_contract_parses(value, expected):
    assert parse_prosody_percent(value) == expected


@pytest.mark.parametrize("value", ["+101%", "-101%", "+999%", "0%", "+10", "10%", "", "+0%\n"])
def test_prosody_outside_the_contract_does_not(value):
    assert parse_prosody_percent(value) is None
