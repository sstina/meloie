"""Pure tests for the StreamingRvcEngine live setters (Phase 0 control API).

Constructs the engine but NEVER calls load() — __init__ does not import torch,
so this runs in any venv with numpy. The object-building setters
(set_formant/set_denoise) have their build helpers monkeypatched to no-ops so we
test the setter logic (validation + flag/cfg mutation) without the optional
stftpitchshift/noisereduce deps.
"""

from __future__ import annotations

import math

import pytest

from src.engine.streaming_engine import (
    StreamingEngineConfig,
    StreamingEngineError,
    StreamingRvcEngine,
)


def _engine(**cfg_kw):
    cfg = StreamingEngineConfig(model_path="x.pth", **cfg_kw)
    return StreamingRvcEngine(cfg)


# --------------------------------------------------------------------------
# scalar input-side setters (read fresh per block by _convert -> live)
# --------------------------------------------------------------------------

def test_set_pitch_shift_mutates_and_coerces_int():
    e = _engine()
    e.set_pitch_shift(7)
    assert e.pitch_shift == 7
    e.set_pitch_shift(3.9)        # coerced to int
    assert e.pitch_shift == 3


def test_set_protect_range():
    e = _engine()
    e.set_protect(0.4)
    assert e.protect == pytest.approx(0.4)
    with pytest.raises(ValueError):
        e.set_protect(0.6)
    with pytest.raises(ValueError):
        e.set_protect(-0.1)


def test_set_index_rate_validates_and_guards_on_index():
    e = _engine()
    # 0.0 is always fine (no retrieval needed)
    e.set_index_rate(0.0)
    assert e.index_rate == 0.0
    # out of range
    with pytest.raises(ValueError):
        e.set_index_rate(1.1)
    # > 0 with no index loaded -> engine error (not loaded => no pipeline/index)
    with pytest.raises(StreamingEngineError):
        e.set_index_rate(0.5)


def test_set_index_path_requires_loaded():
    e = _engine()
    with pytest.raises(StreamingEngineError):
        e.set_index_path("models/V2.index")


def test_set_sid_requires_loaded_and_defaults_one_speaker():
    e = _engine()
    assert e.num_speakers == 1                      # set in load() from emb_g
    with pytest.raises(StreamingEngineError):       # no pipeline yet
        e.set_sid(0)


def test_set_autotune_pair():
    e = _engine()
    e.set_autotune(True, 0.5)
    assert e.f0_autotune is True
    assert e.f0_autotune_strength == pytest.approx(0.5)
    e.set_autotune(False)
    assert e.f0_autotune is False
    with pytest.raises(ValueError):
        e.set_autotune(True, 2.0)


def test_set_auto_pitch_pair():
    e = _engine()
    e.set_auto_pitch(True, 200.0)
    assert e.proposed_pitch is True
    assert e.proposed_pitch_threshold == pytest.approx(200.0)
    with pytest.raises(ValueError):
        e.set_auto_pitch(True, 0.0)


# --------------------------------------------------------------------------
# object-building setters (flag-gated; build helper stubbed in pure tests)
# --------------------------------------------------------------------------

def test_set_formant_flag_and_cfg(monkeypatch):
    e = _engine()
    monkeypatch.setattr(e, "_build_formant", lambda: None)  # avoid stftpitchshift
    e.set_formant(True, timbre=0.25, qfrency=1.5)
    assert e._formant_on is True
    assert e.cfg.formant_timbre == pytest.approx(0.25)
    assert e.cfg.formant_qfrency == pytest.approx(1.5)
    e.set_formant(False)
    assert e._formant_on is False
    with pytest.raises(ValueError):
        e.set_formant(True, timbre=-1.0)


def test_set_denoise_flag_and_cfg(monkeypatch):
    e = _engine()
    monkeypatch.setattr(e, "_build_denoiser", lambda: None)  # avoid noisereduce
    e.set_denoise(True, strength=0.7)
    assert e._denoise_on is True
    assert e.cfg.denoise_strength == pytest.approx(0.7)
    e.set_denoise(False)
    assert e._denoise_on is False
    with pytest.raises(ValueError):
        e.set_denoise(True, strength=2.0)


def test_set_silence_gate():
    e = _engine()  # default block_ms == 250
    e.set_silence_gate(-50.0, 300.0)
    assert e._silence_thresh_lin == pytest.approx(10.0 ** (-50.0 / 20.0))
    assert e.cfg.silence_threshold_dbfs == pytest.approx(-50.0)
    assert e._silence_hangover_blocks == math.ceil(300.0 / 250.0)  # == 2
    e.set_silence_gate(None)
    assert e._silence_thresh_lin is None
    with pytest.raises(ValueError):
        e.set_silence_gate(-40.0, -1.0)


# --------------------------------------------------------------------------
# faithful-carrier contract: NO output-shaping setter may exist
# --------------------------------------------------------------------------

def test_no_output_shaping_setters():
    e = _engine()
    for bad in ("set_rms_mix_rate", "set_output_denoise", "set_gain",
                "set_volume_envelope", "set_eq", "set_limiter"):
        assert not hasattr(e, bad), f"output-shaping setter {bad!r} must not exist"
    # broader scan: no set_* method names an output-shaping concept
    forbidden = ("rms", "gain", "volume", "output", "eq", "limiter", "reverb", "compress")
    for name in dir(e):
        if name.startswith("set_"):
            assert not any(tok in name.lower() for tok in forbidden), \
                f"setter {name!r} looks like output shaping (contract violation)"
