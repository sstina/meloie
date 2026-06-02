"""Pure tests for the 捏脸 per-model save (no Qt, no torch).

Covers ``save_model_profile``: writes a profile that the strict ModelProfile
loader accepts, updates an existing one preserving its index_path / legacy
fields, folds formant on/off into formant_timbre, clamps out-of-range values,
and never leaks non-profile keys.
"""

from __future__ import annotations

import json

from src.engine.model_profile import load_model_profile
from src.ui import presets as pr


def test_save_creates_loadable_profile(tmp_path):
    path = pr.save_model_profile("models/MyVoice.pth", {
        "pitch_shift": 12, "index_rate": 0.3, "protect": 0.4,
        "formant_timbre": 1.2, "formant_on": True,
    }, str(tmp_path))
    prof = load_model_profile(path)             # strict loader: rejects unknown keys
    assert prof.pitch_shift == 12
    assert prof.index_rate == 0.3
    assert prof.protect == 0.4
    assert prof.formant_timbre == 1.2
    assert prof.model_path == "models/MyVoice.pth"


def test_save_formant_off_writes_neutral(tmp_path):
    path = pr.save_model_profile("models/X.pth",
                                 {"formant_timbre": 1.5, "formant_on": False},
                                 str(tmp_path))
    assert load_model_profile(path).formant_timbre == 1.0   # off -> neutral 1.0


def test_save_updates_existing_preserving_index_path(tmp_path):
    (tmp_path / "A.json").write_text(json.dumps({
        "name": "A", "model_path": "models/A.pth",
        "index_path": "models/V2.index", "f0_method": "rmvpe",
        "index_rate": 0.0, "pitch_shift": 0, "rms_mix_rate": 1.0,
    }), encoding="utf-8")
    pr.save_model_profile("models/A.pth", {"pitch_shift": 12}, str(tmp_path))
    prof = load_model_profile(str(tmp_path / "A.json"))
    assert prof.pitch_shift == 12                 # updated
    assert prof.index_path == "models/V2.index"   # preserved
    assert prof.rms_mix_rate == 1.0               # legacy field preserved


def test_save_clamps_and_rejects_nonfinite(tmp_path):
    # out-of-range / NaN slider values are clamped/dropped on write so the saved
    # profile is always loadable (the strict loader doesn't range-check floats).
    path = pr.save_model_profile("models/Z.pth", {
        "index_rate": 1.5, "protect": 2.0,
        "formant_timbre": float("nan"), "formant_on": True,
    }, str(tmp_path))
    prof = load_model_profile(path)
    assert prof.index_rate == 1.0            # clamped to [0, 1]
    assert prof.protect == 0.5               # clamped to [0, 0.5]
    assert prof.formant_timbre == 1.0        # NaN -> neutral default


def test_save_only_writes_valid_keys(tmp_path):
    pr.save_model_profile("models/Y.pth", {
        "pitch_shift": 5, "formant_on": True, "formant_timbre": 1.1, "junk_key": 999,
    }, str(tmp_path))
    raw = json.loads((tmp_path / "Y.json").read_text(encoding="utf-8"))
    assert "formant_on" not in raw and "junk_key" not in raw
    load_model_profile(str(tmp_path / "Y.json"))  # must not raise
