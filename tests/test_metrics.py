"""Tests for safety helpers (dBFS, scrub) + the lean RuntimeMetrics."""

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
)
from src.safety.metrics import RuntimeMetrics


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


def test_dbfs_returns_native_python_float_on_real_signal():
    # np.log10 yields np.float64; the GUI crosses these into QML as a QVariantMap
    # where a numpy scalar is an un-assignable PyObjectWrapper. The non-silence
    # branch must hand back a *native* float. (regression: "Cannot assign
    # PySide::PyObjectWrapper to double")
    sig = np.full(512, 0.3, dtype=np.float32)
    assert type(dbfs_peak(sig)) is float
    assert type(dbfs_rms(sig)) is float


def test_to_dict_has_no_numpy_scalars():
    m = RuntimeMetrics()
    m.input_peak_dbfs = np.float64(-12.5)        # simulate a stray numpy value
    m.output_rms_dbfs = np.float32(-20.0)
    d = m.to_dict()
    assert type(d["input_peak_dbfs"]) is float and d["input_peak_dbfs"] == pytest.approx(-12.5)
    assert all(type(v).__module__ != "numpy" for v in d.values())


# ---------------------------------------------------------------------------
# NaN / Inf scrub (the only safety transform the runtime applies)
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
    assert result.audio[0] == pytest.approx(0.1)
    assert result.audio[5] == pytest.approx(0.5)


def test_scrub_clean_audio_is_noop():
    clean = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    result = scrub_nan_inf(clean)
    assert result.nan_count == 0
    assert result.inf_count == 0
    np.testing.assert_array_equal(result.audio, clean)


# ---------------------------------------------------------------------------
# RuntimeMetrics: JSON-safety + timing counters
# ---------------------------------------------------------------------------

def test_runtime_metrics_to_dict_is_json_serialisable():
    m = RuntimeMetrics(
        elapsed_seconds=12.5,
        input_frames=100,
        output_frames=100,
        input_queue_drops=1,
        output_underruns=2,
        notes=["startup_ok"],
    )
    decoded = json.loads(json.dumps(m.to_dict()))
    assert decoded["input_frames"] == 100
    assert decoded["output_underruns"] == 2
    assert decoded["notes"] == ["startup_ok"]


def test_runtime_metrics_to_json_round_trip():
    decoded = json.loads(RuntimeMetrics().to_json())
    assert decoded["input_frames"] == 0
    assert decoded["rvc_fallback_count"] == 0


def test_record_inference_ms_updates_mean_max_last():
    m = RuntimeMetrics()
    for ms in (100.0, 200.0, 300.0, 400.0, 500.0):
        m.record_inference_ms(ms, budget_ms=1000.0)
    assert m.rvc_inference_count == 5
    assert m.rvc_inference_last_ms == pytest.approx(500.0)
    assert m.rvc_inference_max_ms == pytest.approx(500.0)
    assert m.rvc_inference_mean_ms == pytest.approx(300.0)
