"""Pure tests for the GUI model-discovery + recipe-lookup (no torch, no Qt).

Covers the models/ folder discovery and the per-model recipe (a matching
``<stem>.json`` profile supplies pitch/index/...; otherwise neutral defaults).
Uses tmp dirs + monkeypatch so it never depends on real model files.
"""

from __future__ import annotations

import json

from src.ui import config_assembly as ca


def test_list_model_files_lists_pth_by_stem(tmp_path, monkeypatch):
    (tmp_path / "B.pth").write_bytes(b"x")
    (tmp_path / "A.pth").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("nope", encoding="utf-8")
    monkeypatch.setattr(ca, "models_dir", lambda: str(tmp_path))
    ms = ca.list_model_files()
    assert [m["name"] for m in ms] == ["A", "B"]      # sorted, .pth only, stem = name
    assert ms[0]["path"].endswith("A.pth")


def test_list_model_files_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "models_dir", lambda: str(tmp_path / "nope"))
    assert ca.list_model_files() == []


def _write_profile(tmp_path, stem, **fields):
    payload = {"name": stem, "model_path": f"models/{stem}.pth", **fields}
    (tmp_path / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_model_default_params_from_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path))
    _write_profile(tmp_path, "X", pitch_shift=12, index_rate=0.5,
                   index_path="models/V2.index", protect=0.33)
    p = ca.model_default_params("anywhere/X.pth")
    assert p["pitch_shift"] == 12 and p["index_rate"] == 0.5 and p["has_index"] is True


def test_model_default_params_defaults_without_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path))
    p = ca.model_default_params("anywhere/NOPROFILE.pth")
    assert p["pitch_shift"] == 0 and p["index_rate"] == 0.0 and p["has_index"] is False
    assert p["formant_on"] is False                 # neutral default -> formant off


def test_model_default_params_derives_formant_on(tmp_path, monkeypatch):
    # a saved gender shift (timbre != 1.0) must surface as formant_on=True so the GUI
    # checkbox reflects it; the dead formant_qfrency key is no longer emitted.
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path))
    _write_profile(tmp_path, "F", formant_timbre=1.25)
    p = ca.model_default_params("anywhere/F.pth")
    assert p["formant_on"] is True and p["formant_timbre"] == 1.25
    assert "formant_qfrency" not in p
    _write_profile(tmp_path, "N", formant_timbre=1.0)
    assert ca.model_default_params("anywhere/N.pth")["formant_on"] is False


def test_build_configs_for_model_applies_profile_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path))
    _write_profile(tmp_path, "X", pitch_shift=12, index_rate=0.5,
                   index_path="models/V2.index")
    scfg, acfg = ca.build_configs_for_model("models/X.pth", None, None, "fcpe")
    assert scfg.model_path == "models/X.pth"          # the discovered .pth, not the profile's
    assert scfg.pitch_shift == 12
    assert scfg.index_path == "models/V2.index"       # recipe from the profile
    assert scfg.f0_method == "fcpe"
    assert acfg.channels == 1


def test_build_configs_for_model_defaults_without_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path))
    scfg, _ = ca.build_configs_for_model("models/NOPROF.pth", None, None, "rmvpe")
    assert scfg.model_path == "models/NOPROF.pth"
    assert scfg.pitch_shift == 0 and scfg.index_path == "" and scfg.index_rate == 0.0
