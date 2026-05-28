"""Runtime metrics dataclasses — JSON-serialisable.

Counter and level fields chosen to match the 22-metric dictionary in
the legacy dossier (see ``legacy.md`` §5). Field names are kept stable
so sidecar JSON shape is portable across stages.

Stage 2 adds RVC-specific fields (chunks processed, inference timing,
RVC fallback count) and session-info fields (model basename, params).
The identity path leaves these zero/empty.

Stage 4-C adds spike-protection and queue-health counters needed by
the headless quality-first runtime: how often per-chunk inference
exceeded its budget, how long the worst sustained over-budget run was,
how often the output queue grazed empty, and the running cumulative
input-vs-output frame delta. These surface "the chain is about to
glitch" trends before they become silence on `CABLE Output`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


# Mirrors safety/guard.DBFS_SILENCE_FLOOR. Duplicated here to keep
# this module import-light and dependency-free.
_DBFS_SILENCE_FLOOR = -200.0


@dataclass
class AudioLevelMetrics:
    """Peak and RMS levels for one side of the audio loop."""

    peak_dbfs: float = _DBFS_SILENCE_FLOOR
    rms_dbfs: float = _DBFS_SILENCE_FLOOR

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeMetrics:
    """Counters + levels for one realtime session.

    All fields are JSON-serialisable primitives. Identity-mode runs
    leave the ``rvc_*`` block at zero / empty string.
    """

    elapsed_seconds: float = 0.0

    # frame counters
    input_frames: int = 0
    output_frames: int = 0

    # queue health
    input_queue_drops: int = 0
    output_queue_drops: int = 0
    output_underruns: int = 0

    # levels
    input_peak_dbfs: float = _DBFS_SILENCE_FLOOR
    input_rms_dbfs: float = _DBFS_SILENCE_FLOOR
    output_peak_dbfs: float = _DBFS_SILENCE_FLOOR
    output_rms_dbfs: float = _DBFS_SILENCE_FLOOR

    # safety / fault counters
    fallback_count: int = 0
    nan_inf_scrub_count: int = 0
    clip_count: int = 0
    limiter_engagement_count: int = 0

    # device + callback health
    device_invalidation_count: int = 0
    input_status_flag_count: int = 0
    output_status_flag_count: int = 0

    # Stage 2: RVC-specific counters
    rvc_chunks_processed: int = 0
    rvc_inference_count: int = 0
    rvc_inference_mean_ms: float = 0.0
    rvc_inference_max_ms: float = 0.0
    rvc_inference_last_ms: float = 0.0
    rvc_fallback_count: int = 0

    # Stage 2E: live RVC diagnostics
    # Per-call inference timings, capped to avoid unbounded memory growth
    # during long sessions. The final-summary printer computes median /
    # p95 from this list.
    rvc_inference_times_ms: list = field(default_factory=list)
    rvc_inference_times_cap: int = 4096

    # Resample-step timing (worker-side 40k -> 48k linear resample).
    rvc_resample_count: int = 0
    rvc_resample_total_ms: float = 0.0
    rvc_resample_last_ms: float = 0.0

    # Output enqueue accounting (RVC mode only).
    rvc_output_blocks_enqueued: int = 0
    rvc_output_blocks_dropped: int = 0

    # Stale-input policy: chunks the worker deliberately discarded because
    # inference fell behind. Reduces latency drift at the cost of audio
    # continuity gaps.
    rvc_stale_chunk_drops: int = 0

    # Queue-depth high-water marks (sampled at metrics print + on push).
    max_input_queue_depth: int = 0
    max_output_queue_depth: int = 0

    # Stage 4-C: inference spike protection.
    # An inference is "over budget" when its wall-clock exceeds the
    # per-chunk audio budget (= chunk_ms). In steady state every
    # over-budget call consumes part of the output-queue safety margin,
    # so streaks of them are the most reliable predictor of underruns.
    rvc_chunk_ms_budget: float = 0.0
    rvc_inference_over_budget_count: int = 0
    rvc_inference_over_budget_total_ms: float = 0.0
    rvc_inference_over_budget_max_consecutive: int = 0
    # Transient state for max-consecutive tracking; reset to 0 each
    # time an on-budget inference lands. Kept on the dataclass (and
    # serialised) so a session's end-state is fully reproducible.
    rvc_inference_consecutive_over_budget_current: int = 0

    # Stage 4-C: output queue health (steady-state only).
    # ``min_output_queue_depth_after_steady`` is None until first real
    # audio is enqueued (so the prebuffer drain isn't mis-counted as a
    # near-empty). After that it tracks the minimum depth we ever saw.
    min_output_queue_depth_after_steady: Optional[int] = None
    output_queue_near_empty_threshold_blocks: int = 0
    output_queue_near_empty_events: int = 0
    # Edge-triggered: only counts the transition from above-threshold
    # to at-or-below-threshold, so one sustained drain doesn't tick
    # the counter on every poll.
    output_queue_above_near_empty_last: bool = True

    # Startup vs steady-state output underruns. Bin is decided in the
    # output callback based on whether the worker has emitted any real
    # audio yet (``first_real_output_seen``).
    startup_output_underruns: int = 0
    steady_state_output_underruns: int = 0
    first_real_output_seen: bool = False

    # Stage 2: session info (informational; populated at stream start)
    rvc_chunk_ms: float = 0.0
    rvc_crossfade_ms: float = 0.0
    rvc_model_basename: str = ""
    rvc_index_basename: str = ""
    rvc_f0_method: str = ""
    rvc_index_rate: float = 0.0
    rvc_protect: float = 0.0
    rvc_pitch_shift: int = 0

    # Free-form notes appended over the session.
    notes: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def record_inference_ms(self, ms: float, budget_ms: float = 0.0) -> None:
        """Update inference timing counters with one new sample.

        ``budget_ms`` is the per-chunk audio budget (typically the
        chunk_ms in input time). When > 0, three Stage 4-C counters
        are maintained:

        * ``rvc_inference_over_budget_count`` -- # calls with ms > budget
        * ``rvc_inference_over_budget_total_ms`` -- sum of (ms - budget)
          over those calls (= total wall-clock "debt" the worker has
          accumulated vs the audio stream)
        * ``rvc_inference_over_budget_max_consecutive`` -- longest
          run of consecutive over-budget calls observed

        Pass ``budget_ms=0`` (the default) to skip spike tracking --
        the legacy callsites that don't know the budget still work.
        """
        ms = float(ms)
        self.rvc_inference_count += 1
        self.rvc_inference_last_ms = ms
        if ms > self.rvc_inference_max_ms:
            self.rvc_inference_max_ms = ms
        # incremental mean
        n = self.rvc_inference_count
        self.rvc_inference_mean_ms = (
            ((n - 1) * self.rvc_inference_mean_ms + ms) / n
        )
        # Keep a bounded raw list for median / p95 computation at end.
        if len(self.rvc_inference_times_ms) < self.rvc_inference_times_cap:
            self.rvc_inference_times_ms.append(ms)
        # Stage 4-C: over-budget tracking.
        if budget_ms and budget_ms > 0.0:
            if ms > budget_ms:
                self.rvc_inference_over_budget_count += 1
                self.rvc_inference_over_budget_total_ms += (ms - budget_ms)
                self.rvc_inference_consecutive_over_budget_current += 1
                if (
                    self.rvc_inference_consecutive_over_budget_current
                    > self.rvc_inference_over_budget_max_consecutive
                ):
                    self.rvc_inference_over_budget_max_consecutive = (
                        self.rvc_inference_consecutive_over_budget_current
                    )
            else:
                self.rvc_inference_consecutive_over_budget_current = 0

    def record_output_queue_depth(
        self, qd: int, near_empty_threshold_blocks: int = 0
    ) -> None:
        """Sample the output queue depth for Stage 4-C health tracking.

        Skipped until ``first_real_output_seen`` is True so the prebuffer
        drain doesn't mis-count as a near-empty. ``min_output_queue_depth_
        after_steady`` is updated each sample (initialised lazily). When
        a near-empty threshold is configured (in blocks), the counter is
        edge-triggered -- one tick per transition from above-threshold to
        at-or-below-threshold -- so a single drain doesn't spam the count.
        """
        if not self.first_real_output_seen:
            return
        qd = int(qd)
        if (
            self.min_output_queue_depth_after_steady is None
            or qd < self.min_output_queue_depth_after_steady
        ):
            self.min_output_queue_depth_after_steady = qd
        if near_empty_threshold_blocks > 0:
            self.output_queue_near_empty_threshold_blocks = int(
                near_empty_threshold_blocks
            )
            now_near_empty = qd <= near_empty_threshold_blocks
            if now_near_empty and self.output_queue_above_near_empty_last:
                self.output_queue_near_empty_events += 1
            self.output_queue_above_near_empty_last = not now_near_empty

    @property
    def cumulative_frame_delta(self) -> int:
        """Net (input_frames - output_frames). Positive = output behind input.

        At session end this is the running audio-length deficit:
        if positive, the model + plumbing emitted that many fewer
        samples than the mic produced (the per-chunk framing loss
        documented in the README). If it grows monotonically over a
        long run, the prebuffer is being eaten faster than steady-state
        production can replenish -- expect eventual underruns.
        """
        return int(self.input_frames) - int(self.output_frames)

    def record_resample_ms(self, ms: float) -> None:
        """Update worker-side resample timing counters."""
        ms = float(ms)
        self.rvc_resample_count += 1
        self.rvc_resample_total_ms += ms
        self.rvc_resample_last_ms = ms

    def inference_percentile_ms(self, percentile: float) -> float:
        """Compute the requested percentile of the recorded inference times."""
        times = self.rvc_inference_times_ms
        if not times:
            return 0.0
        import numpy as _np  # local — keep module top dep-light
        arr = _np.asarray(times, dtype=_np.float64)
        return float(_np.percentile(arr, float(percentile)))

    def inference_median_ms(self) -> float:
        return self.inference_percentile_ms(50.0)

    @property
    def rvc_resample_mean_ms(self) -> float:
        if self.rvc_resample_count == 0:
            return 0.0
        return self.rvc_resample_total_ms / float(self.rvc_resample_count)
