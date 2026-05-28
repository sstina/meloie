"""Tests for safety helpers + metrics dataclass JSON-serialisability."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.safety.guard import (
    DBFS_SILENCE_FLOOR,
    dbfs_peak,
    dbfs_rms,
    scrub_nan_inf,
    simple_limiter,
)
from src.safety.metrics import AudioLevelMetrics, RuntimeMetrics


# ---------------------------------------------------------------------------
# dBFS helpers
# ---------------------------------------------------------------------------

def test_dbfs_silence_returns_floor_not_neg_inf():
    silence = np.zeros(1024, dtype=np.float32)
    assert dbfs_peak(silence) == DBFS_SILENCE_FLOOR
    assert dbfs_rms(silence) == DBFS_SILENCE_FLOOR
    assert math.isfinite(dbfs_peak(silence))
    assert math.isfinite(dbfs_rms(silence))


def test_dbfs_empty_array_returns_floor():
    empty = np.zeros(0, dtype=np.float32)
    assert dbfs_peak(empty) == DBFS_SILENCE_FLOOR
    assert dbfs_rms(empty) == DBFS_SILENCE_FLOOR


def test_dbfs_full_scale_is_zero():
    full = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    assert dbfs_peak(full) == pytest.approx(0.0, abs=1e-9)
    assert dbfs_rms(full) == pytest.approx(0.0, abs=1e-9)


def test_dbfs_half_scale_is_minus_six_ish():
    half = np.full(1024, 0.5, dtype=np.float32)
    assert dbfs_peak(half) == pytest.approx(-6.0206, abs=1e-3)
    assert dbfs_rms(half) == pytest.approx(-6.0206, abs=1e-3)


# ---------------------------------------------------------------------------
# NaN / Inf scrub
# ---------------------------------------------------------------------------

def test_scrub_replaces_nan_and_inf_with_zero_and_counts():
    bad = np.array([0.1, np.nan, 0.3, np.inf, -np.inf, 0.5], dtype=np.float32)
    result = scrub_nan_inf(bad)
    assert result.nan_count == 1
    assert result.inf_count == 2
    assert result.replaced_count == 3
    assert np.all(np.isfinite(result.audio))
    assert result.audio[1] == 0.0
    assert result.audio[3] == 0.0
    assert result.audio[4] == 0.0
    # untouched samples preserved
    assert result.audio[0] == pytest.approx(0.1)
    assert result.audio[5] == pytest.approx(0.5)


def test_scrub_clean_audio_is_noop():
    clean = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    result = scrub_nan_inf(clean)
    assert result.nan_count == 0
    assert result.inf_count == 0
    np.testing.assert_array_equal(result.audio, clean)


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------

def test_limiter_below_ceiling_does_not_clip():
    quiet = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    result = simple_limiter(quiet, ceiling_dbfs=-1.0)
    assert result.samples_clipped == 0
    assert not result.engaged
    np.testing.assert_array_equal(result.audio, quiet)


def test_limiter_clips_above_ceiling():
    loud = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    result = simple_limiter(loud, ceiling_dbfs=-6.0)
    # ceiling linear = 10**(-6/20) ~= 0.501
    assert result.samples_clipped >= 2
    assert result.engaged
    # ceiling linear at -6 dBFS is 10**(-6/20) ~= 0.50119
    assert float(np.max(np.abs(result.audio))) <= 0.5012 + 1e-3


def test_limiter_rejects_non_negative_ceiling():
    with pytest.raises(ValueError):
        simple_limiter(np.zeros(4, dtype=np.float32), ceiling_dbfs=0.0)


# ---------------------------------------------------------------------------
# RuntimeMetrics JSON-safety
# ---------------------------------------------------------------------------

def test_runtime_metrics_to_dict_is_json_serialisable():
    m = RuntimeMetrics(
        elapsed_seconds=12.5,
        input_frames=100,
        output_frames=100,
        input_queue_drops=1,
        output_underruns=2,
        fallback_count=0,
        notes=["startup_ok"],
    )
    payload = m.to_dict()
    # round-trip through json must succeed without TypeError
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["input_frames"] == 100
    assert decoded["output_underruns"] == 2
    assert decoded["notes"] == ["startup_ok"]


def test_runtime_metrics_to_json_round_trip():
    m = RuntimeMetrics()
    decoded = json.loads(m.to_json())
    assert decoded["input_frames"] == 0
    assert decoded["fallback_count"] == 0


def test_audio_level_metrics_default_to_silence_floor():
    al = AudioLevelMetrics()
    assert al.peak_dbfs == DBFS_SILENCE_FLOOR
    assert al.rms_dbfs == DBFS_SILENCE_FLOOR
    json.dumps(al.to_dict())  # must not raise
