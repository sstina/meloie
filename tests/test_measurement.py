"""Tests for the pure measurement helpers used by tools/verify_cable_route.py."""

from __future__ import annotations

import numpy as np
import pytest

from meloie.audio.measurement import (
    DEFAULT_NON_SILENCE_THRESHOLD_DBFS,
    generate_sine_tone,
    summarize_capture,
)


# ---------------------------------------------------------------------------
# generate_sine_tone
# ---------------------------------------------------------------------------

def test_sine_tone_has_expected_length_and_peak():
    sr = 48000
    tone = generate_sine_tone(sample_rate=sr, duration_seconds=0.25,
                              frequency_hz=440.0, amplitude=0.5)
    assert tone.size == int(round(0.25 * sr))
    assert float(np.max(np.abs(tone))) == pytest.approx(0.5, abs=1e-3)


def test_sine_tone_rejects_bad_args():
    with pytest.raises(ValueError):
        generate_sine_tone(sample_rate=48000, duration_seconds=0.0)
    with pytest.raises(ValueError):
        generate_sine_tone(sample_rate=48000, duration_seconds=0.1,
                           frequency_hz=0.0)


# ---------------------------------------------------------------------------
# summarize_capture
# ---------------------------------------------------------------------------

def test_summarize_capture_silence_is_safely_serialisable():
    silence = np.zeros(1024, dtype=np.float32)
    summary = summarize_capture(silence)
    assert summary["n_samples"] == 1024
    assert summary["non_silent"] is False
    # must round-trip through JSON
    import json
    decoded = json.loads(json.dumps(summary))
    assert decoded["non_silent"] is False
    assert decoded["threshold_dbfs"] == DEFAULT_NON_SILENCE_THRESHOLD_DBFS


def test_summarize_capture_signal_is_non_silent():
    tone = generate_sine_tone(sample_rate=48000, duration_seconds=0.1,
                              frequency_hz=440.0, amplitude=0.5)
    summary = summarize_capture(tone)
    assert summary["non_silent"] is True
    assert summary["peak_dbfs"] > -10.0
    assert summary["rms_dbfs"] < summary["peak_dbfs"]


def test_summarize_capture_threshold_is_honoured():
    very_quiet = np.full(1024, 1e-5, dtype=np.float32)  # -100 dBFS
    assert summarize_capture(very_quiet, threshold_dbfs=-60.0)["non_silent"] is False
    assert summarize_capture(very_quiet, threshold_dbfs=-120.0)["non_silent"] is True


def test_summarize_capture_rejects_non_array():
    with pytest.raises(TypeError):
        summarize_capture([0.0, 0.1])  # type: ignore[arg-type]
