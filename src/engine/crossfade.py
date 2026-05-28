"""Pure crossfade helpers for chunk-boundary smoothing.

Used in Stage 3 to remove the click at the seam between two adjacent
RVC chunks. Implemented here in Stage 1 because it is pure numpy with
no hardware dependency and is convenient to unit test in isolation.
"""

from __future__ import annotations

import numpy as np


def _validate_mono(a: np.ndarray, name: str) -> None:
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(a).__name__}")
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D mono, got shape {a.shape}")


def linear_crossfade(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Linear crossfade between ``tail`` (fading out) and ``head`` (fading in).

    Both inputs must be mono and the same length. Returns a mono array
    of the same length.
    """
    _validate_mono(tail, "tail")
    _validate_mono(head, "head")
    if tail.shape != head.shape:
        raise ValueError(
            f"tail and head must have the same shape, "
            f"got {tail.shape} and {head.shape}"
        )

    n = tail.shape[0]
    if n == 0:
        return np.zeros(0, dtype=tail.dtype)

    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float64)
    fade_in = 1.0 - fade_out
    out = tail.astype(np.float64) * fade_out + head.astype(np.float64) * fade_in
    return out.astype(tail.dtype, copy=False)


def equal_power_crossfade(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Equal-power (sin / cos) crossfade between ``tail`` and ``head``.

    Preserves perceived loudness across the fade region better than
    linear for correlated material. Inputs must be 1-D and same length.
    """
    _validate_mono(tail, "tail")
    _validate_mono(head, "head")
    if tail.shape != head.shape:
        raise ValueError(
            f"tail and head must have the same shape, "
            f"got {tail.shape} and {head.shape}"
        )

    n = tail.shape[0]
    if n == 0:
        return np.zeros(0, dtype=tail.dtype)

    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    fade_out = np.cos(t * np.pi / 2.0)
    fade_in = np.sin(t * np.pi / 2.0)
    out = tail.astype(np.float64) * fade_out + head.astype(np.float64) * fade_in
    return out.astype(tail.dtype, copy=False)
