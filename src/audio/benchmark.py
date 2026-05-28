"""Pure helpers for the RVC inference microbenchmark.

These helpers are deliberately free of any sounddevice / hardware /
torch dependency so they can be unit tested without the RVC stack
installed.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def slice_repeating(audio: np.ndarray, chunk_size: int) -> np.ndarray:
    """Return exactly ``chunk_size`` mono float32 samples.

    If ``audio`` is shorter than ``chunk_size`` it is tiled to fill.
    Always returns a fresh array (callers may mutate it without affecting
    the source).
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a numpy array, got {type(audio).__name__}"
        )
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D mono, got shape {audio.shape}")
    if audio.size == 0:
        raise ValueError("audio must be non-empty")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be > 0; got {chunk_size}")

    if audio.size >= chunk_size:
        return audio[:chunk_size].astype(np.float32, copy=True)
    reps = (chunk_size + audio.size - 1) // audio.size
    return np.tile(audio, reps)[:chunk_size].astype(np.float32, copy=True)


def summarize_timings(times_ms: List[float]) -> Dict[str, float]:
    """Summarise a list of per-call inference times in milliseconds."""
    if not times_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "min_ms": 0.0,
        }
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
        "min_ms": float(np.min(arr)),
    }


def realtime_factor(mean_inference_ms: float, chunk_ms: float) -> float:
    """Ratio of inference time to chunk audio duration.

    < 1.0 = faster than realtime (chunk-bound latency)
    1.0   = exactly realtime — no headroom
    > 1.0 = slower than realtime — queue will grow without bound
    """
    if float(chunk_ms) <= 0:
        raise ValueError(f"chunk_ms must be > 0; got {chunk_ms}")
    return float(mean_inference_ms) / float(chunk_ms)


def fits_realtime(rt_factor: float, headroom: float = 0.7) -> bool:
    """True iff there is enough margin to sustain realtime safely.

    ``headroom`` of 0.7 means inference must consistently use less than
    70% of the chunk duration. Lower than that gives queue + jitter
    breathing room.
    """
    return float(rt_factor) < float(headroom)
