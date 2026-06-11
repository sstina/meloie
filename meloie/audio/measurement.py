"""Pure measurement helpers for the cable-route verification tool.

Used by ``tools/verify_cable_route.py`` (renders a tone, checks the capture is
non-silent). Hardware-free — unit testable with synthetic numpy arrays. (The
historical click-pulse + cross-correlation latency helpers were removed: no
in-repo tool consumed them.)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..safety.guard import dbfs_peak, dbfs_rms


DEFAULT_NON_SILENCE_THRESHOLD_DBFS = -60.0


def generate_sine_tone(
    sample_rate: int,
    duration_seconds: float,
    frequency_hz: float = 440.0,
    amplitude: float = 0.25,
) -> np.ndarray:
    """Generate a mono sine tone for cable-route verification."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0")
    if not (0.0 < amplitude <= 1.0):
        raise ValueError("amplitude must be in (0, 1]")

    n = int(round(duration_seconds * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    signal = (amplitude * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)
    return signal


def summarize_capture(
    audio: np.ndarray,
    threshold_dbfs: float = DEFAULT_NON_SILENCE_THRESHOLD_DBFS,
) -> Dict[str, Any]:
    """Return a JSON-serialisable summary of a captured buffer. The threshold
    (-60 dBFS default) sits well above a healthy path's noise floor but below
    any deliberately rendered signal."""
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a numpy array, got {type(audio).__name__}"
        )
    peak = float(dbfs_peak(audio))
    rms = float(dbfs_rms(audio))
    return {
        "n_samples": int(audio.size),
        "peak_dbfs": peak,
        "rms_dbfs": rms,
        "non_silent": bool(peak > float(threshold_dbfs)),
        "threshold_dbfs": float(threshold_dbfs),
    }
