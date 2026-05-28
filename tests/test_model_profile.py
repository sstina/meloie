"""Tests for the Stage 2F model profile loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.engine.model_profile import (
    ModelProfile,
    ModelProfileError,
    load_model_profile,
)


# ---------------------------------------------------------------------------
# Happy-path loading
# ---------------------------------------------------------------------------

def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_minimum_required_fields(tmp_path):
    p = _write(tmp_path / "min.json", {"model_path": "x.pth"})
    profile = load_model_profile(str(p))
    assert isinstance(profile, ModelProfile)
    assert profile.model_path == "x.pth"
    # Defaults for fields not in the JSON.
    assert profile.f0_method == "rmvpe"
    assert profile.index_rate == 0.5
    assert profile.protect == 0.33
    assert profile.pitch_shift == 0


def test_loads_full_profile(tmp_path):
    payload = {
        "name": "kiki",
        "model_path": "models/kiki/kikiV1.pth",
        "index_path": "models/kiki/kikiV1.index",
        "hubert_path": "models/kiki/hubert_base.pt",
        "rmvpe_path":  "models/kiki/rmvpe.pt",
        "f0_method": "rmvpe",
        "index_rate": 0.5,
        "protect": 0.33,
        "filter_radius": 3,
        "rms_mix_rate": 0.25,
        "pitch_shift": 0,
        "resample_sr": 0,
        "notes": "example",
    }
    profile = load_model_profile(str(_write(tmp_path / "full.json", payload)))
    assert profile.name == "kiki"
    assert profile.model_path == "models/kiki/kikiV1.pth"
    assert profile.notes == "example"


def test_relative_paths_pass_through_verbatim(tmp_path):
    """The loader does NOT resolve relative paths — callers do, against cwd."""
    p = _write(tmp_path / "p.json", {"model_path": "models/kiki/kikiV1.pth"})
    profile = load_model_profile(str(p))
    assert profile.model_path == "models/kiki/kikiV1.pth"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_missing_model_path_raises(tmp_path):
    p = _write(tmp_path / "nomp.json", {"name": "kiki"})
    with pytest.raises(ModelProfileError) as exc:
        load_model_profile(str(p))
    assert "model_path" in str(exc.value)


def test_unknown_keys_rejected(tmp_path):
    p = _write(tmp_path / "weird.json",
               {"model_path": "x.pth", "secret_tuning_knob": 999})
    with pytest.raises(ModelProfileError) as exc:
        load_model_profile(str(p))
    assert "secret_tuning_knob" in str(exc.value)


def test_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid", encoding="utf-8")
    with pytest.raises(ModelProfileError):
        load_model_profile(str(bad))


def test_non_object_json_raises(tmp_path):
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ModelProfileError):
        load_model_profile(str(arr))


def test_missing_file_raises(tmp_path):
    with pytest.raises(ModelProfileError) as exc:
        load_model_profile(str(tmp_path / "nope.json"))
    assert "not found" in str(exc.value)


# ---------------------------------------------------------------------------
# Real on-disk example profile must remain loadable
# ---------------------------------------------------------------------------

def test_kiki_example_profile_loads_and_has_expected_shape():
    """The committed example profile must always be a valid profile."""
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "config" / "model_profiles" / "kiki.example.json"
    assert p.exists(), f"missing example profile at {p}"
    profile = load_model_profile(str(p))
    assert profile.name == "kiki"
    assert profile.model_path.endswith(".pth")
    # Recommended params for the kiki model documented in the profile.
    assert profile.f0_method == "rmvpe"
    assert profile.resample_sr == 0
