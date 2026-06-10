"""Pure tests for the precise F0 mapping math (CDF / quantile histogram match).

No torch, no Qt — exercises meloie/engine/f0_map.py with synthetic F0-in-Hz arrays.
Pins: identity, octave shift, range compression + endpoint clamp, monotonicity,
unvoiced passthrough + shape/dtype, out-of-band rejection, min-frame guard, and a
flat-pitched source (tied quantiles) staying well-defined.
"""

from __future__ import annotations

import numpy as np
import pytest

from meloie.engine.f0_map import (
    MIN_VOICED_FRAMES,
    build_quantiles,
    make_remap,
)


def _rng():
    return np.random.default_rng(1234)


def _voiced(arr):
    a = np.asarray(arr, dtype=np.float64)
    return a[a > 0]


def test_identity_when_source_equals_target():
    f0 = _rng().uniform(110.0, 220.0, size=600).astype(np.float32)
    src_q, tgt_q = build_quantiles(f0, f0)
    remap = make_remap(src_q, tgt_q)
    probe = np.linspace(120.0, 210.0, 50).astype(np.float32)
    out = remap(probe)
    assert np.allclose(out, probe, rtol=0.02)        # ~identity (modulo the 1e-6 xp ramp)


def test_unvoiced_passthrough_and_shape_dtype():
    f0 = _rng().uniform(110.0, 220.0, size=600).astype(np.float32)
    remap = make_remap(*build_quantiles(f0, f0))
    x = np.array([0.0, 150.0, 0.0, 180.0], dtype=np.float32)
    out = remap(x)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert out[0] == 0.0 and out[2] == 0.0           # unvoiced stays 0


def test_octave_shift_maps_median_up_one_octave():
    voice = _rng().normal(120.0, 12.0, size=800).clip(80, 200).astype(np.float32)
    target = (voice * 2.0).astype(np.float32)        # +1 octave
    remap = make_remap(*build_quantiles(voice, target))
    out = remap(voice)
    # median of mapped voice ~ 2x the source median (i.e. +1.0 in log2)
    assert np.isclose(np.median(_voiced(out)) / np.median(_voiced(voice)), 2.0, rtol=0.05)


def test_range_compression_and_endpoint_clamp():
    wide = _rng().normal(150.0, 40.0, size=1500).clip(60, 400).astype(np.float32)
    narrow = _rng().normal(150.0, 8.0, size=1500).clip(120, 180).astype(np.float32)
    remap = make_remap(*build_quantiles(wide, narrow))
    out = remap(wide)
    assert np.std(_voiced(out)) < np.std(_voiced(wide))      # spread compressed toward target
    # an input far above the source range clamps to (around) the target's max, not beyond
    hi = remap(np.array([5000.0], dtype=np.float32))[0]
    assert hi <= np.max(narrow) * 1.05


def test_monotonic_map():
    voice = _rng().normal(130.0, 20.0, size=800).clip(70, 260).astype(np.float32)
    target = _rng().normal(200.0, 25.0, size=800).clip(120, 320).astype(np.float32)
    remap = make_remap(*build_quantiles(voice, target))
    probe = np.linspace(80.0, 250.0, 200).astype(np.float32)
    out = remap(probe)
    assert np.all(np.diff(out) >= -1e-6)             # non-decreasing


def test_out_of_band_frames_do_not_skew_quantiles():
    voice = _rng().normal(140.0, 15.0, size=800).clip(90, 220).astype(np.float32)
    target = _rng().normal(180.0, 15.0, size=800).clip(120, 260).astype(np.float32)
    clean = build_quantiles(voice, target)
    # inject octave-error spikes outside the [50,1100] band -> must be ignored
    voice_bad = np.concatenate([voice, np.full(50, 30.0, np.float32),
                                np.full(50, 2000.0, np.float32)])
    spiked = build_quantiles(voice_bad, target)
    assert np.allclose(clean[0], spiked[0]) and np.allclose(clean[1], spiked[1])


def test_too_few_voiced_frames_raises():
    short = _rng().uniform(110.0, 200.0, size=MIN_VOICED_FRAMES - 1).astype(np.float32)
    ok = _rng().uniform(110.0, 200.0, size=600).astype(np.float32)
    with pytest.raises(ValueError):
        build_quantiles(short, ok)
    with pytest.raises(ValueError):
        build_quantiles(ok, short)


def test_flat_source_is_well_defined():
    voice = np.full(600, 150.0, dtype=np.float32)    # constant pitch -> tied quantiles
    target = _rng().normal(200.0, 15.0, size=600).clip(150, 260).astype(np.float32)
    src_q, tgt_q = build_quantiles(voice, target)
    assert np.all(np.diff(src_q) > 0)                # strictly increasing xp (monotonicity guard)
    out = make_remap(src_q, tgt_q)(np.array([150.0, 151.0], dtype=np.float32))
    assert np.all(np.isfinite(out))
