"""Tests for the pure RVC benchmark helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.benchmark import (
    fits_realtime,
    realtime_factor,
    slice_repeating,
    summarize_timings,
)


# ---------------------------------------------------------------------------
# slice_repeating
# ---------------------------------------------------------------------------

def test_slice_repeating_exact_length():
    audio = np.arange(10_000, dtype=np.float32)
    chunk = slice_repeating(audio, 5_000)
    assert chunk.shape == (5_000,)
    assert chunk.dtype == np.float32
    np.testing.assert_array_equal(chunk, audio[:5_000])
    # must be a copy, not a view
    chunk[0] = 999.0
    assert audio[0] == 0.0


def test_slice_repeating_tiles_short_input():
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    chunk = slice_repeating(audio, 8)
    np.testing.assert_array_equal(chunk, [1, 2, 3, 1, 2, 3, 1, 2])


def test_slice_repeating_rejects_empty_audio():
    with pytest.raises(ValueError):
        slice_repeating(np.zeros(0, dtype=np.float32), 100)


def test_slice_repeating_rejects_stereo():
    with pytest.raises(ValueError):
        slice_repeating(np.zeros((100, 2), dtype=np.float32), 50)


def test_slice_repeating_rejects_non_positive_chunk():
    with pytest.raises(ValueError):
        slice_repeating(np.zeros(100, dtype=np.float32), 0)


# ---------------------------------------------------------------------------
# summarize_timings
# ---------------------------------------------------------------------------

def test_summarize_timings_basic_stats():
    s = summarize_timings([10.0, 20.0, 30.0, 40.0, 50.0])
    assert s["count"] == 5
    assert s["min_ms"] == pytest.approx(10.0)
    assert s["max_ms"] == pytest.approx(50.0)
    assert s["mean_ms"] == pytest.approx(30.0)
    assert s["median_ms"] == pytest.approx(30.0)


def test_summarize_timings_p95_picks_high_tail():
    times = [10.0] * 19 + [1000.0]
    s = summarize_timings(times)
    # p95 should clearly exceed the median; max captures the tail.
    assert s["median_ms"] == pytest.approx(10.0)
    assert s["p95_ms"] > s["median_ms"]
    assert s["max_ms"] == 1000.0


def test_summarize_timings_empty_returns_zeros():
    s = summarize_timings([])
    assert s["count"] == 0
    assert s["mean_ms"] == 0.0
    assert s["max_ms"] == 0.0


# ---------------------------------------------------------------------------
# realtime_factor / fits_realtime
# ---------------------------------------------------------------------------

def test_realtime_factor_basic():
    # 90 ms inference for 180 ms chunk -> 0.5
    assert realtime_factor(90.0, 180.0) == pytest.approx(0.5)
    # 360 ms inference for 180 ms chunk -> 2.0 (twice realtime)
    assert realtime_factor(360.0, 180.0) == pytest.approx(2.0)


def test_realtime_factor_rejects_zero_chunk():
    with pytest.raises(ValueError):
        realtime_factor(50.0, 0.0)


def test_fits_realtime_default_headroom():
    # default headroom 0.7 — anything < 0.7 fits
    assert fits_realtime(0.5)
    assert fits_realtime(0.69)
    assert not fits_realtime(0.7)
    assert not fits_realtime(1.0)
    assert not fits_realtime(5.0)


def test_fits_realtime_custom_headroom():
    # tighter or looser headroom
    assert fits_realtime(0.85, headroom=0.9)
    assert not fits_realtime(0.85, headroom=0.5)
