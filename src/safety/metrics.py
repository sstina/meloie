"""Runtime metrics dataclasses — JSON-serialisable.

Counter and level fields chosen to match the 22-metric dictionary
in the legacy dossier (see ``legacy.md`` §5). Field names are kept
stable so sidecar JSON shape is portable across stages.
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

    All fields are JSON-serialisable primitives or nested dataclasses
    that themselves serialise cleanly via ``asdict``.
    """

    elapsed_seconds: float = 0.0

    # frame counters
    input_frames: int = 0
    output_frames: int = 0

    # queue health
    input_queue_drops: int = 0
    output_underruns: int = 0

    # levels (filled in periodically from safety.guard helpers)
    input_peak_dbfs: float = _DBFS_SILENCE_FLOOR
    input_rms_dbfs: float = _DBFS_SILENCE_FLOOR
    output_peak_dbfs: float = _DBFS_SILENCE_FLOOR
    output_rms_dbfs: float = _DBFS_SILENCE_FLOOR

    # safety / fault counters
    fallback_count: int = 0
    nan_inf_scrub_count: int = 0
    clip_count: int = 0
    limiter_engagement_count: int = 0

    # device health
    device_invalidation_count: int = 0

    # free-form notes appended over the session (kept short / non-PII)
    notes: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
