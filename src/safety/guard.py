"""Pure safety helpers: dBFS measurement and NaN/Inf scrub.

No audio hardware dependency, no sounddevice import. All helpers are
numpy-based and handle silence (all zeros) without producing NaN or
crashing. The runtime does NO gain shaping / limiting (model-faithful);
the only safety transform here is replacing non-finite samples with 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Floor for dBFS readings so silence returns a finite (very negative)
# value instead of -inf. -200 dBFS is well below the noise floor of any
# real signal and avoids special-casing -inf at every callsite.
DBFS_SILENCE_FLOOR = -200.0


def _as_float(audio: np.ndarray) -> np.ndarray:
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a numpy array, got {type(audio).__name__}"
        )
    return audio.astype(np.float64, copy=False)


def dbfs_peak(audio: np.ndarray) -> float:
    """Peak level in dBFS (0 dBFS = full scale ±1.0).

    Silence returns ``DBFS_SILENCE_FLOOR`` rather than -inf.
    """
    a = _as_float(audio)
    if a.size == 0:
        return DBFS_SILENCE_FLOOR
    peak = float(np.max(np.abs(a)))
    if peak <= 0.0 or not np.isfinite(peak):
        return DBFS_SILENCE_FLOOR
    return float(20.0 * np.log10(peak))   # native float: np.log10 returns np.float64


def dbfs_rms(audio: np.ndarray) -> float:
    """RMS level in dBFS. Silence returns ``DBFS_SILENCE_FLOOR``."""
    a = _as_float(audio)
    if a.size == 0:
        return DBFS_SILENCE_FLOOR
    mean_sq = float(np.mean(a * a))
    if mean_sq <= 0.0 or not np.isfinite(mean_sq):
        return DBFS_SILENCE_FLOOR
    rms = np.sqrt(mean_sq)
    return float(20.0 * np.log10(rms))    # native float: np.log10 returns np.float64


@dataclass(frozen=True)
class ScrubResult:
    """Outcome of a NaN/Inf scrub pass."""

    audio: np.ndarray
    nan_count: int
    inf_count: int

    @property
    def replaced_count(self) -> int:
        return self.nan_count + self.inf_count


def scrub_nan_inf(audio: np.ndarray) -> ScrubResult:
    """Replace NaN / +-Inf samples with 0.0. Returns scrubbed copy + counts."""
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a numpy array, got {type(audio).__name__}"
        )
    out = audio.astype(audio.dtype, copy=True)
    nan_mask = np.isnan(out)
    inf_mask = np.isinf(out)
    nan_count = int(nan_mask.sum())
    inf_count = int(inf_mask.sum())
    if nan_count or inf_count:
        out[nan_mask | inf_mask] = 0.0
    return ScrubResult(audio=out, nan_count=nan_count, inf_count=inf_count)
