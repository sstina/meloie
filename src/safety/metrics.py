"""Runtime metrics dataclasses — JSON-serialisable.

Counter and level fields chosen to match the 22-metric dictionary in
the legacy dossier (see ``legacy.md`` §5). Field names are kept stable
so sidecar JSON shape is portable across stages.

Stage 2 adds RVC-specific fields (chunks processed, inference timing,
RVC fallback count) and session-info fields (model basename, params).
The identity path leaves these zero/empty.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


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

    def record_inference_ms(self, ms: float) -> None:
        """Update inference timing counters with one new sample."""
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
