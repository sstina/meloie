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


# ---------------------------------------------------------------------------
# Stage 2E: median / p95 helpers and resample mean
# ---------------------------------------------------------------------------

def test_inference_median_and_p95_use_recorded_times():
    m = RuntimeMetrics()
    for ms in (100.0, 200.0, 300.0, 400.0, 500.0):
        m.record_inference_ms(ms)
    assert m.rvc_inference_count == 5
    assert m.inference_median_ms() == pytest.approx(300.0)
    # p95 of 5 samples with default numpy linear interp lands near max
    assert m.inference_percentile_ms(95.0) > m.inference_median_ms()


def test_record_resample_ms_updates_mean():
    m = RuntimeMetrics()
    m.record_resample_ms(2.0)
    m.record_resample_ms(4.0)
    m.record_resample_ms(6.0)
    assert m.rvc_resample_count == 3
    assert m.rvc_resample_mean_ms == pytest.approx(4.0)
    assert m.rvc_resample_last_ms == pytest.approx(6.0)


def test_inference_times_list_is_capped():
    m = RuntimeMetrics()
    m.rvc_inference_times_cap = 5
    for _ in range(20):
        m.record_inference_ms(1.0)
    assert len(m.rvc_inference_times_ms) == 5
    # but the running counters still count all 20
    assert m.rvc_inference_count == 20


# ---------------------------------------------------------------------------
# Stage 4-C: inference-budget spike protection
# ---------------------------------------------------------------------------

def test_record_inference_ms_no_budget_skips_over_budget_tracking():
    """The legacy signature (no budget) must remain a no-op for spike
    counters so old callsites keep working."""
    m = RuntimeMetrics()
    m.record_inference_ms(99999.0)  # huge but no budget passed
    assert m.rvc_inference_over_budget_count == 0
    assert m.rvc_inference_over_budget_max_consecutive == 0
    assert m.rvc_inference_over_budget_total_ms == 0.0


def test_record_inference_ms_tracks_over_budget_count_and_debt():
    """Each ms > budget bumps the over-budget count and adds the excess
    to the total-debt counter; on-budget calls don't."""
    m = RuntimeMetrics()
    budget = 1000.0
    m.record_inference_ms(500.0, budget_ms=budget)     # on budget
    m.record_inference_ms(1500.0, budget_ms=budget)    # over by 500
    m.record_inference_ms(900.0, budget_ms=budget)     # on budget
    m.record_inference_ms(1200.0, budget_ms=budget)    # over by 200
    assert m.rvc_inference_over_budget_count == 2
    assert m.rvc_inference_over_budget_total_ms == pytest.approx(700.0)


def test_record_inference_ms_tracks_max_consecutive_over_budget():
    """A streak of over-budget calls pulls up max_consecutive; the
    streak resets when an on-budget call lands."""
    m = RuntimeMetrics()
    budget = 100.0
    # streak of 3
    m.record_inference_ms(200.0, budget_ms=budget)
    m.record_inference_ms(150.0, budget_ms=budget)
    m.record_inference_ms(300.0, budget_ms=budget)
    assert m.rvc_inference_over_budget_max_consecutive == 3
    assert m.rvc_inference_consecutive_over_budget_current == 3
    # break the streak
    m.record_inference_ms(50.0, budget_ms=budget)
    assert m.rvc_inference_consecutive_over_budget_current == 0
    # smaller streak of 2 -- max stays at 3
    m.record_inference_ms(200.0, budget_ms=budget)
    m.record_inference_ms(150.0, budget_ms=budget)
    assert m.rvc_inference_over_budget_max_consecutive == 3
    assert m.rvc_inference_consecutive_over_budget_current == 2


def test_record_inference_ms_budget_equal_is_on_budget():
    """ms == budget is exactly on budget, not over (strict > test)."""
    m = RuntimeMetrics()
    m.record_inference_ms(1000.0, budget_ms=1000.0)
    assert m.rvc_inference_over_budget_count == 0


# ---------------------------------------------------------------------------
# Stage 4-C: output queue health
# ---------------------------------------------------------------------------

def test_record_output_queue_depth_noop_before_steady_state():
    """Before first_real_output_seen=True the prebuffer is draining;
    we must not pollute the steady-state min with that."""
    m = RuntimeMetrics()
    m.record_output_queue_depth(5, near_empty_threshold_blocks=10)
    assert m.min_output_queue_depth_after_steady is None
    assert m.output_queue_near_empty_events == 0


def test_record_output_queue_depth_tracks_min_after_steady():
    m = RuntimeMetrics()
    m.first_real_output_seen = True
    m.record_output_queue_depth(50)
    m.record_output_queue_depth(30)
    m.record_output_queue_depth(40)
    m.record_output_queue_depth(20)
    assert m.min_output_queue_depth_after_steady == 20


def test_record_output_queue_depth_near_empty_edge_triggered():
    """One sustained drain should only tick the near-empty counter once,
    not on every sample. The counter advances each time the queue
    transitions from above-threshold to at-or-below-threshold."""
    m = RuntimeMetrics()
    m.first_real_output_seen = True
    threshold = 10
    m.record_output_queue_depth(50, threshold)   # above
    m.record_output_queue_depth(40, threshold)   # above
    m.record_output_queue_depth(5,  threshold)   # transition -> near (count++)
    m.record_output_queue_depth(3,  threshold)   # still near (no count)
    m.record_output_queue_depth(8,  threshold)   # still near (no count)
    m.record_output_queue_depth(50, threshold)   # above again
    m.record_output_queue_depth(2,  threshold)   # transition -> near (count++)
    assert m.output_queue_near_empty_events == 2
    assert m.output_queue_near_empty_threshold_blocks == threshold


def test_record_output_queue_depth_skipped_when_threshold_zero():
    """If threshold is 0 (caller doesn't want this), the min still
    updates but no near-empty events accumulate."""
    m = RuntimeMetrics()
    m.first_real_output_seen = True
    m.record_output_queue_depth(2, near_empty_threshold_blocks=0)
    m.record_output_queue_depth(1, near_empty_threshold_blocks=0)
    assert m.output_queue_near_empty_events == 0
    assert m.min_output_queue_depth_after_steady == 1


# ---------------------------------------------------------------------------
# Stage 4-C: cumulative frame delta
# ---------------------------------------------------------------------------

def test_cumulative_frame_delta_zero_at_start():
    m = RuntimeMetrics()
    assert m.cumulative_frame_delta == 0


def test_cumulative_frame_delta_positive_when_output_behind_input():
    m = RuntimeMetrics()
    m.input_frames = 1000
    m.output_frames = 950
    assert m.cumulative_frame_delta == 50  # output behind input


def test_cumulative_frame_delta_negative_when_output_ahead_of_input():
    m = RuntimeMetrics()
    m.input_frames = 100
    m.output_frames = 300  # eg. prebuffer of silence
    assert m.cumulative_frame_delta == -200


# ---------------------------------------------------------------------------
# Stage 4-E: timeline reconciliation recording
# ---------------------------------------------------------------------------

def test_record_timeline_reconcile_accumulates_totals():
    m = RuntimeMetrics()
    m.record_timeline_reconcile(expected_frames=48000, actual_frames=47040, reconciled_frames=48000)
    m.record_timeline_reconcile(expected_frames=48000, actual_frames=47100, reconciled_frames=48000)
    m.record_timeline_reconcile(expected_frames=48000, actual_frames=47200, reconciled_frames=48000)
    assert m.timeline_reconcile_count == 3
    assert m.timeline_expected_output_frames_total == 144000
    assert m.timeline_actual_output_frames_total == 47040 + 47100 + 47200
    assert m.timeline_reconciled_output_frames_total == 144000
    # Signed cumulative error (negative = model under-emitted).
    assert m.timeline_reconciliation_total_frame_error == (
        (47040 - 48000) + (47100 - 48000) + (47200 - 48000)
    )
    # Max abs per chunk = 960 (= 48000 - 47040).
    assert m.timeline_max_reconciliation_frames_per_chunk == 960
    # Mean ratio for kiki sits near 0.98.
    assert 0.97 < m.timeline_reconciliation_mean_ratio < 0.99


def test_timeline_reconciliation_mean_ratio_zero_when_no_data():
    m = RuntimeMetrics()
    assert m.timeline_reconciliation_mean_ratio == 0.0


def test_record_timeline_reconcile_handles_excess_actual():
    """Model occasionally emits MORE than expected (positive error)."""
    m = RuntimeMetrics()
    m.record_timeline_reconcile(expected_frames=48000, actual_frames=49000, reconciled_frames=48000)
    assert m.timeline_reconciliation_total_frame_error == 1000
    assert m.timeline_max_reconciliation_frames_per_chunk == 1000


def test_runtime_metrics_dict_includes_new_stage4e_timeline_fields():
    """Sidecar JSON consumers depend on the dataclass shape."""
    m = RuntimeMetrics()
    m.record_timeline_reconcile(expected_frames=48000, actual_frames=47040, reconciled_frames=48000)
    d = m.to_dict()
    for k in (
        "timeline_reconcile_enabled",
        "timeline_reconcile_method",
        "timeline_reconcile_count",
        "timeline_expected_output_frames_total",
        "timeline_actual_output_frames_total",
        "timeline_reconciled_output_frames_total",
        "timeline_max_reconciliation_frames_per_chunk",
        "timeline_reconciliation_total_frame_error",
    ):
        assert k in d, f"missing {k}"
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    assert decoded["timeline_reconcile_count"] == 1


# ---------------------------------------------------------------------------
# Stage 4-E2: input-side frame restoration recording
# ---------------------------------------------------------------------------

def test_record_frame_restore_accumulates_totals():
    m = RuntimeMetrics()
    # 3 chunks: emit target 4800, render 4896 (= chunk + tail 192 - deficit 96),
    # trim_start 0 (no context), trim_end 96 surplus, no shortfall.
    for _ in range(3):
        m.record_frame_restore(
            expected=4800, actual_before_trim=4896, emitted=4800,
            trim_start=0, trim_end=96, shortfall_frames=0,
        )
    assert m.frame_restoration_count == 3
    assert m.frame_restoration_expected_frames_total == 3 * 4800
    assert m.frame_restoration_actual_frames_total == 3 * 4896
    assert m.frame_restoration_emitted_frames_total == 3 * 4800
    assert m.frame_restoration_trim_start_frames_total == 0
    assert m.frame_restoration_trim_end_frames_total == 3 * 96
    assert m.frame_restoration_shortfall_count == 0
    # |actual - expected| per chunk = 96.
    assert m.frame_restoration_max_abs_per_chunk_frame_error == 96


def test_record_frame_restore_counts_shortfall():
    m = RuntimeMetrics()
    # Healthy chunk, then a chunk where the model render was too short and
    # the slice had to be zero-padded (shortfall > 0).
    m.record_frame_restore(4800, 4896, 4800, 0, 96, 0)
    m.record_frame_restore(4800, 4700, 4800, 0, 0, 100)
    assert m.frame_restoration_count == 2
    assert m.frame_restoration_shortfall_count == 1


def test_runtime_metrics_dict_includes_frame_restore_fields():
    """Sidecar JSON consumers depend on the dataclass shape."""
    m = RuntimeMetrics()
    m.record_frame_restore(4800, 4896, 4800, 0, 96, 0)
    d = m.to_dict()
    for k in (
        "frame_restore_method",
        "frame_restore_enabled",
        "input_tail_pad_ms",
        "input_tail_pad_frames",
        "frame_restoration_count",
        "frame_restoration_shortfall_count",
        "frame_restoration_expected_frames_total",
        "frame_restoration_actual_frames_total",
        "frame_restoration_emitted_frames_total",
        "frame_restoration_trim_start_frames_total",
        "frame_restoration_trim_end_frames_total",
        "frame_restoration_max_abs_per_chunk_frame_error",
        "output_stretch_used_count",
    ):
        assert k in d, f"missing {k}"
    decoded = json.loads(json.dumps(d))
    assert decoded["frame_restoration_count"] == 1


def test_runtime_metrics_dict_includes_new_stage4c_fields_and_is_json_safe():
    """Sidecar JSON consumers depend on the dataclass shape; make sure
    the new Stage 4-C fields are present, JSON-encodable (None ->
    null), and that no field accidentally ended up as a numpy type."""
    m = RuntimeMetrics()
    m.first_real_output_seen = True
    m.record_inference_ms(1500.0, budget_ms=1000.0)
    m.record_output_queue_depth(2, near_empty_threshold_blocks=5)
    d = m.to_dict()
    for k in (
        "rvc_chunk_ms_budget",
        "rvc_inference_over_budget_count",
        "rvc_inference_over_budget_total_ms",
        "rvc_inference_over_budget_max_consecutive",
        "min_output_queue_depth_after_steady",
        "output_queue_near_empty_threshold_blocks",
        "output_queue_near_empty_events",
    ):
        assert k in d, f"missing {k}"
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    assert decoded["rvc_inference_over_budget_count"] == 1
    assert decoded["min_output_queue_depth_after_steady"] == 2
