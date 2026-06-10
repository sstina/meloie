"""Tests for pure measurement helpers used by the Stage 1 validation tools."""

from __future__ import annotations

import numpy as np
import pytest

from meloie.audio.measurement import (
    DEFAULT_NON_SILENCE_THRESHOLD_DBFS,
    estimate_latency_samples,
    generate_click_pulse,
    generate_sine_tone,
    is_non_silent,
    summarize_capture,
)


# ---------------------------------------------------------------------------
# generate_click_pulse
# ---------------------------------------------------------------------------

def test_click_pulse_has_expected_length_and_peak_position():
    sr = 48000
    pre = 0.1
    post = 0.2
    click = 32
    signal = generate_click_pulse(
        sample_rate=sr,
        pulse_amplitude=0.5,
        pre_silence_seconds=pre,
        post_silence_seconds=post,
        click_samples=click,
    )
    expected_len = int(round(pre * sr)) + click + int(round(post * sr))
    assert signal.size == expected_len
    assert signal.dtype == np.float32
    # peak should be located inside the click window
    pre_n = int(round(pre * sr))
    peak_idx = int(np.argmax(np.abs(signal)))
    assert pre_n <= peak_idx < pre_n + click
    # peak amplitude approximately matches request (within ramp resolution)
    assert float(np.max(np.abs(signal))) == pytest.approx(0.5, abs=1e-2)


def test_click_pulse_silence_outside_window_is_zero():
    sr = 8000
    signal = generate_click_pulse(
        sample_rate=sr,
        pulse_amplitude=0.7,
        pre_silence_seconds=0.05,
        post_silence_seconds=0.1,
        click_samples=8,
    )
    pre_n = int(round(0.05 * sr))
    assert float(np.max(np.abs(signal[:pre_n]))) == 0.0
    assert float(np.max(np.abs(signal[pre_n + 8:]))) == 0.0


def test_click_pulse_rejects_bad_args():
    with pytest.raises(ValueError):
        generate_click_pulse(sample_rate=0)
    with pytest.raises(ValueError):
        generate_click_pulse(sample_rate=48000, click_samples=0)
    with pytest.raises(ValueError):
        generate_click_pulse(sample_rate=48000, pulse_amplitude=0.0)
    with pytest.raises(ValueError):
        generate_click_pulse(sample_rate=48000, pulse_amplitude=1.5)


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
# estimate_latency_samples
# ---------------------------------------------------------------------------

def test_estimate_latency_zero_lag_perfect_match():
    sr = 48000
    ref = generate_click_pulse(sr, pre_silence_seconds=0.05,
                               post_silence_seconds=0.1, click_samples=16)
    lag, peak = estimate_latency_samples(ref, ref)
    assert lag == 0
    assert peak == pytest.approx(1.0, abs=1e-9)


def test_estimate_latency_recovers_known_positive_lag():
    sr = 48000
    ref = generate_click_pulse(sr, pre_silence_seconds=0.05,
                               post_silence_seconds=0.5, click_samples=16)
    # Captured = reference delayed by exactly 1234 samples.
    cap = np.zeros_like(ref)
    cap[1234:] = ref[: ref.size - 1234]
    lag, peak = estimate_latency_samples(ref, cap)
    assert lag == 1234
    assert peak > 0.9


def test_estimate_latency_respects_max_lag():
    sr = 48000
    ref = generate_click_pulse(sr, pre_silence_seconds=0.05,
                               post_silence_seconds=0.5, click_samples=16)
    cap = np.zeros_like(ref)
    cap[1234:] = ref[: ref.size - 1234]
    # Restrict to lags <= 100 — true lag (1234) is outside that range;
    # the search must return a lag <= 100 regardless.
    lag, _peak = estimate_latency_samples(ref, cap, max_lag_samples=100)
    assert 0 <= lag <= 100


def test_estimate_latency_rejects_empty_arrays():
    with pytest.raises(ValueError):
        estimate_latency_samples(np.zeros(0), np.zeros(10))
    with pytest.raises(ValueError):
        estimate_latency_samples(np.zeros(10), np.zeros(0))


def test_estimate_latency_on_silence_returns_zero_peak():
    sr = 48000
    ref = generate_click_pulse(sr, pre_silence_seconds=0.05,
                               post_silence_seconds=0.1, click_samples=16)
    silence = np.zeros_like(ref)
    lag, peak = estimate_latency_samples(ref, silence)
    # peak should be 0 because captured is all-zero; lag value is
    # undefined but must be a valid index.
    assert 0 <= lag < silence.size
    assert peak == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# is_non_silent + summarize_capture
# ---------------------------------------------------------------------------

def test_is_non_silent_with_silence_returns_false():
    silence = np.zeros(1024, dtype=np.float32)
    assert is_non_silent(silence) is False


def test_is_non_silent_with_tone_returns_true():
    tone = generate_sine_tone(sample_rate=48000, duration_seconds=0.1,
                              frequency_hz=440.0, amplitude=0.5)
    assert is_non_silent(tone) is True


def test_is_non_silent_threshold_is_honoured():
    very_quiet = np.full(1024, 1e-5, dtype=np.float32)  # -100 dBFS
    assert is_non_silent(very_quiet, threshold_dbfs=-60.0) is False
    assert is_non_silent(very_quiet, threshold_dbfs=-120.0) is True


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


def test_summarize_capture_rejects_non_array():
    with pytest.raises(TypeError):
        summarize_capture([0.0, 0.1])  # type: ignore[arg-type]
