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
    # Persistent-failure escalation (worker): after N consecutive fallbacks the
    # engine's streaming state is reset once (re-warms ~context_ms — cheap
    # insurance against a poisoned persistent state); after many more, the
    # unhealthy flag flips so the GUI/console can surface "engine down,
    # identity passthrough" instead of silently degrading forever.
    rvc_engine_resets: int = 0
    rvc_engine_unhealthy: bool = False
    rvc_stale_chunk_drops: int = 0
    rvc_output_blocks_enqueued: int = 0
    rvc_output_blocks_dropped: int = 0
    # SilenceFront (w-okada borrow): chunks whose input RMS fell below the
    # silence threshold and were emitted as zeros without running inference
    # (written by the worker from engine.last_silence_skipped).
    rvc_silence_skipped_count: int = 0
    # SOLA seam alignment (faithful: chooses the cut offset, no sample edit;
    # written by the worker from engine.last_sola_offset).
    rvc_sola_offset_last: int = 0

    # session info (informational; populated at stream start)
    rvc_chunk_ms: float = 0.0
    rvc_model_basename: str = ""
    rvc_index_basename: str = ""

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Coerce any numpy scalar (e.g. np.float64 from a dBFS calc) to a native
        # Python number: this dict becomes a QVariantMap in the GUI, and numpy
        # scalars arrive in QML as opaque PyObjectWrapper (un-assignable to a
        # `real`/`int` property). Numpy-agnostic — keys off the .item() method.
        for k, v in d.items():
            item = getattr(v, "item", None)
            if callable(item) and not isinstance(v, (str, bytes, list, tuple, dict)):
                try:
                    d[k] = v.item()
                except Exception:
                    pass
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def record_inference_ms(self, ms: float) -> None:
        """Update inference timing counters with one new sample. The per-chunk
        budget lives in ``rvc_chunk_ms_budget``; mean/max surface spikes."""
        ms = float(ms)
        self.rvc_inference_count += 1
        self.rvc_inference_last_ms = ms
        if ms > self.rvc_inference_max_ms:
            self.rvc_inference_max_ms = ms
        n = self.rvc_inference_count
        self.rvc_inference_mean_ms = ((n - 1) * self.rvc_inference_mean_ms + ms) / n
