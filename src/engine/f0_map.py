"""Precise F0 mapping — CDF / quantile histogram matching (pure numpy).

INPUT-side carrier conditioning ONLY. Builds a monotonic map from a *source*
speaker's F0 distribution (the live user's own voice) onto a *target* voice's F0
distribution (the model's native voice), in the log2(Hz) domain, so the carrier's
ESTIMATED F0 can be remapped before it drives the model's pitch input. The model's
OUTPUT samples are never touched — this is the same kind of input-side knob as the
``f0_up_key`` transpose / autotune / auto-center, so it stays within the
faithful-carrier contract.

Why CDF: a single scalar transpose only aligns the *median*; quantile matching
aligns the whole distribution (median + spread + shape) — a user F0 at their Nth
percentile maps to the target's Nth-percentile F0. We have both real voices, so
the empirical CDFs are trustworthy.

No torch, no Qt — just numpy, so the math is unit-testable in isolation. The
realtime engine does the wav-load + F0 estimation and hands raw F0-in-Hz arrays to
:func:`build_quantiles`, then attaches :func:`make_remap`'s closure to the pipeline.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

# F0 estimators occasionally emit sub-/double-octave errors; clip the analysis to a
# sane vocal band so those outliers don't skew the quantile anchors.
F0_MIN_HZ = 50.0
F0_MAX_HZ = 1100.0
# Quantile knots: trim the bottom/top 1% (octave-error tails) via the endpoints. 48
# knots give a smooth map without adjacent-knot collisions on a few-second clip.
QUANTILE_COUNT = 48
# ~2 s of voiced audio at a 10 ms hop — below this the tail quantiles are unstable.
MIN_VOICED_FRAMES = 200

_P = np.linspace(0.01, 0.99, QUANTILE_COUNT)


def _voiced_log2(f0_hz) -> np.ndarray:
    """The log2 of the in-band voiced frames of a raw F0-in-Hz array (0 = unvoiced)."""
    f0 = np.asarray(f0_hz, dtype=np.float64).reshape(-1)
    voiced = f0[(f0 >= F0_MIN_HZ) & (f0 <= F0_MAX_HZ)]
    return np.log2(voiced) if voiced.size else voiced


def build_quantiles(voice_f0_hz, target_f0_hz) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(src_q, tgt_q)`` log2-Hz quantile anchors from two raw F0-in-Hz arrays.

    ``voice_f0_hz`` is the live user's own voice (source); ``target_f0_hz`` is the
    model's native voice (target). Raises ``ValueError`` if either has fewer than
    :data:`MIN_VOICED_FRAMES` in-band voiced frames. ``src_q`` is nudged strictly
    increasing so it is a valid ``xp`` for :func:`numpy.interp` even when the source
    is flat-pitched (which would otherwise yield tied quantiles).
    """
    lv = _voiced_log2(voice_f0_hz)
    lt = _voiced_log2(target_f0_hz)
    if lv.size < MIN_VOICED_FRAMES or lt.size < MIN_VOICED_FRAMES:
        raise ValueError(
            "精确映射需要更长的有声样本（每段约≥2秒有声）："
            f"你的声音 {lv.size} 帧 / 模型原声 {lt.size} 帧（各需 ≥ {MIN_VOICED_FRAMES}）"
        )
    src_q = np.quantile(lv, _P).astype(np.float64)
    tgt_q = np.quantile(lt, _P).astype(np.float64)
    # strictly-increasing xp for np.interp (tgt_q, the fp/output, may legitimately be flat)
    src_q = src_q + np.arange(src_q.size, dtype=np.float64) * 1e-6
    return src_q, tgt_q


def make_remap(src_q, tgt_q) -> Callable[[np.ndarray], np.ndarray]:
    """Return an ``f0_remap(f0_hz) -> f0_hz`` closure (shape- and dtype-preserving).

    Voiced frames (``f0 > 0``) are quantile-mapped in the log2 domain; unvoiced
    frames stay 0. ``numpy.interp`` is monotonic and clamps beyond the anchors, so an
    unusually high/low live F0 saturates to the target's extreme instead of
    extrapolating wildly. The returned array is C-contiguous and keeps the input
    dtype, because the vendored pipeline then does ``f0 *= ...`` and
    ``torch.from_numpy(f0)`` on it.
    """
    src_q = np.asarray(src_q, dtype=np.float64)
    tgt_q = np.asarray(tgt_q, dtype=np.float64)

    def f0_remap(f0: np.ndarray) -> np.ndarray:
        f0 = np.asarray(f0)
        out = f0.astype(np.float64, copy=True)
        voiced = f0 > 0
        if np.any(voiced):
            out[voiced] = np.exp2(np.interp(np.log2(out[voiced]), src_q, tgt_q))
        return np.ascontiguousarray(out, dtype=f0.dtype)

    return f0_remap
