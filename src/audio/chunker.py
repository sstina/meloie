"""Pure block -> chunk accumulator helpers + resampling helpers.

RVC inference operates on chunks of ~150-200 ms (Stage 1) up to
~1000 ms (Stage 2D first-usable realtime config), but ``sounddevice``
callbacks deliver small blocks (~10 ms). The accumulator here collects
small mono float32 blocks until a fixed chunk size is reached, then
emits the chunk.

Resampling is needed because some RVC models return audio at a
different sample rate than the realtime stream uses (the kiki model
returns 40 kHz natively; our stream is 48 kHz). Two implementations
live here:

* :func:`linear_resample` — pure ``np.interp``. Stable, no edge
  transients, but linear interpolation imprints a low-pass roll-off
  that is not anti-aliased to the destination Nyquist. Kept because
  the test suite pins exact values for the constant-signal case.
* :func:`resample_audio` — preferred. Uses
  ``scipy.signal.resample_poly`` (sinc-windowed polyphase) when scipy
  is importable, falling back to :func:`linear_resample` otherwise.
  The realtime worker uses this one; the audit (tools/pseudo_stream)
  measured a ~+11 dB output-vs-reference SNR upgrade vs linear interp
  on the kiki 40 kHz -> 48 kHz path at negligible CPU cost.

This module is intentionally free of realtime / threading complexity.
It is just numpy buffers + interp, so it can be unit tested without
any audio hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class ChunkerConfig:
    """Configuration for the block accumulator."""

    chunk_size: int      # samples per emitted chunk
    dtype: str = "float32"

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")


def _validate_mono_block(block: np.ndarray) -> None:
    if not isinstance(block, np.ndarray):
        raise TypeError(f"block must be a numpy array, got {type(block).__name__}")
    if block.ndim != 1:
        raise ValueError(f"block must be 1-D mono, got shape {block.shape}")


class BlockAccumulator:
    """Append small mono blocks; emit fixed-size chunks when full.

    Usage::

        acc = BlockAccumulator(ChunkerConfig(chunk_size=9600))
        chunks = acc.feed(small_block_480_samples)
        # chunks is a list of np.ndarray, each exactly chunk_size long
    """

    def __init__(self, config: ChunkerConfig) -> None:
        config.validate()
        self._config = config
        self._buffer: np.ndarray = np.zeros(0, dtype=config.dtype)

    @property
    def pending_samples(self) -> int:
        return int(self._buffer.size)

    @property
    def chunk_size(self) -> int:
        return self._config.chunk_size

    def feed(self, block: np.ndarray) -> List[np.ndarray]:
        """Append one block; return zero or more full chunks."""
        _validate_mono_block(block)
        self._buffer = np.concatenate(
            [self._buffer, block.astype(self._config.dtype, copy=False)]
        )

        chunks: List[np.ndarray] = []
        cs = self._config.chunk_size
        while self._buffer.size >= cs:
            chunks.append(self._buffer[:cs].copy())
            self._buffer = self._buffer[cs:]
        return chunks

    def flush_pending(self) -> np.ndarray:
        """Return whatever partial samples remain (may be empty) and clear."""
        out = self._buffer.copy()
        self._buffer = np.zeros(0, dtype=self._config.dtype)
        return out

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=self._config.dtype)


def linear_resample(
    audio: np.ndarray, from_sr: int, to_sr: int
) -> np.ndarray:
    """Resample a 1-D mono float32 array from ``from_sr`` to ``to_sr``.

    Uses ``np.interp``. Cheap (< 1 ms for one second of audio at 48 kHz)
    and good enough for "first usable realtime"; for actual production
    quality, prefer ``torchaudio.functional.resample`` (sinc-based,
    GPU-aware) or ``scipy.signal.resample_poly``.

    No-ops when ``from_sr == to_sr`` (returns a copy to keep callers
    safe from accidental mutation of an upstream buffer).
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a numpy array, got {type(audio).__name__}"
        )
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D mono, got shape {audio.shape}")
    if int(from_sr) <= 0 or int(to_sr) <= 0:
        raise ValueError(
            f"sample rates must be > 0; got from_sr={from_sr} to_sr={to_sr}"
        )

    if int(from_sr) == int(to_sr) or audio.size == 0:
        return audio.astype(np.float32, copy=True)

    new_len = int(round(audio.size * float(to_sr) / float(from_sr)))
    if new_len <= 0:
        return np.zeros(0, dtype=np.float32)

    x_old = np.arange(audio.size, dtype=np.float64)
    x_new = np.linspace(0.0, audio.size - 1, new_len, dtype=np.float64)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def resample_audio(
    audio: np.ndarray, from_sr: int, to_sr: int
) -> np.ndarray:
    """Resample 1-D mono float32 audio with the best available kernel.

    Prefers ``scipy.signal.resample_poly`` (sinc-windowed polyphase)
    when scipy is importable, otherwise delegates to
    :func:`linear_resample`. The function signature and dtype contract
    mirror ``linear_resample`` exactly so the worker / pseudo-stream
    can call this without conditionals.

    Edge transients: polyphase resampling introduces a small ringing
    transient at the start/end of each *invocation* (a few taps wide).
    For chunked realtime, this means every chunk boundary acquires a
    short ringing region. Empirically (audit run) the cumulative
    effect is still ~+11 dB cleaner than linear interpolation on
    speech material. If a future revision adds input-overlap
    crossfade, those transients fall inside the overlap region and
    become inaudible.
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a numpy array, got {type(audio).__name__}"
        )
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D mono, got shape {audio.shape}")
    if int(from_sr) <= 0 or int(to_sr) <= 0:
        raise ValueError(
            f"sample rates must be > 0; got from_sr={from_sr} to_sr={to_sr}"
        )
    if int(from_sr) == int(to_sr) or audio.size == 0:
        return audio.astype(np.float32, copy=True)

    try:
        from math import gcd
        from scipy.signal import resample_poly  # type: ignore  # noqa: WPS433
    except ImportError:
        return linear_resample(audio, int(from_sr), int(to_sr))

    g = gcd(int(from_sr), int(to_sr))
    up = int(to_sr) // g
    down = int(from_sr) // g
    out = resample_poly(audio.astype(np.float64, copy=False), up, down)
    return out.astype(np.float32, copy=False)
