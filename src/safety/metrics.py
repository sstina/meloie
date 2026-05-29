"""Runtime metrics for one realtime RVC session — JSON-serialisable.

Deliberately lean: only the counters that tell you whether the link is
healthy and stable (frames flowing, queues not overflowing/draining,
inference keeping up, fallbacks/NaN not firing). Anything that only
served a retired diagnostic stage has been removed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

# Mirrors safety/guard.DBFS_SILENCE_FLOOR; duplicated to keep this module
# dependency-free.
_DBFS_SILENCE_FLOOR = -200.0


@dataclass
class RuntimeMetrics:
    """Counters + levels for one realtime RVC session."""

    elapsed_seconds: float = 0.0

    # frame counters
    input_frames: int = 0
    output_frames: int = 0

    # queue health
    input_queue_drops: int = 0
    output_queue_drops: int = 0
    output_underruns: int = 0
    startup_output_underruns: int = 0
    steady_state_output_underruns: int = 0
    first_real_output_seen: bool = False
    max_input_queue_depth: int = 0
    max_output_queue_depth: int = 0

    # levels
    input_peak_dbfs: float = _DBFS_SILENCE_FLOOR
    input_rms_dbfs: float = _DBFS_SILENCE_FLOOR
    output_peak_dbfs: float = _DBFS_SILENCE_FLOOR
    output_rms_dbfs: float = _DBFS_SILENCE_FLOOR

    # safety / device-health counters
    nan_inf_scrub_count: int = 0
    input_status_flag_count: int = 0
    output_status_flag_count: int = 0

    # RVC counters
    rvc_chunks_processed: int = 0
    rvc_inference_count: int = 0
    rvc_inference_mean_ms: float = 0.0
    rvc_inference_max_ms: float = 0.0
    rvc_inference_last_ms: float = 0.0
    rvc_chunk_ms_budget: float = 0.0
    rvc_fallback_count: int = 0
    rvc_stale_chunk_drops: int = 0
    rvc_output_blocks_enqueued: int = 0
    rvc_output_blocks_dropped: int = 0
    frame_restoration_shortfall_count: int = 0
    # SilenceFront (w-okada borrow): chunks whose input RMS fell below the
    # silence threshold and were emitted as zeros without running inference.
    rvc_silence_skipped_count: int = 0

    # SOLA seam alignment (faithful: chooses the cut offset, no sample edit).
    rvc_sola_applied_count: int = 0
    rvc_sola_offset_last: int = 0

    # worker-side resample (model native SR -> stream SR)
    rvc_resample_count: int = 0
    rvc_resample_total_ms: float = 0.0
    rvc_resample_last_ms: float = 0.0

    # input-side frame restoration (look-ahead tail pad)
    input_tail_pad_ms: float = 0.0
    input_tail_pad_frames: int = 0

    # session info (informational; populated at stream start)
    rvc_chunk_ms: float = 0.0
    rvc_model_basename: str = ""
    rvc_index_basename: str = ""

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def record_inference_ms(self, ms: float, budget_ms: float = 0.0) -> None:
        """Update inference timing counters with one new sample.

        ``budget_ms`` is the per-chunk audio budget (= chunk_ms). It is
        accepted for callsite clarity; mean/max already surface spikes,
        so no separate over-budget bookkeeping is kept.
        """
        ms = float(ms)
        self.rvc_inference_count += 1
        self.rvc_inference_last_ms = ms
        if ms > self.rvc_inference_max_ms:
            self.rvc_inference_max_ms = ms
        n = self.rvc_inference_count
        self.rvc_inference_mean_ms = ((n - 1) * self.rvc_inference_mean_ms + ms) / n

    def record_resample_ms(self, ms: float) -> None:
        ms = float(ms)
        self.rvc_resample_count += 1
        self.rvc_resample_total_ms += ms
        self.rvc_resample_last_ms = ms

    @property
    def rvc_resample_mean_ms(self) -> float:
        if self.rvc_resample_count == 0:
            return 0.0
        return self.rvc_resample_total_ms / float(self.rvc_resample_count)
