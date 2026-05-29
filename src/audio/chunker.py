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


def find_sola_offset(haystack: np.ndarray, needle: np.ndarray) -> int:
    """Return the index in ``haystack`` where ``needle`` best aligns.

    Normalised cross-correlation (the alignment half of SOLA). Used to
    phase-match a new chunk's seam region against the previously emitted
    tail, so consecutive per-chunk renders join at a phase-aligned point
    instead of an arbitrary one — which is what removes the comb-filter
    "电音" at chunk boundaries.

    This is the FAITHFUL half of SOLA: it only chooses *where* to cut.
    It returns an index; it never modifies, blends, stretches, or
    pitch-shifts any sample. The caller emits the model's own untouched
    samples starting at the returned offset.

    ``haystack`` must be at least as long as ``needle``. Returns an index
    in ``[0, haystack.size - needle.size]`` (0 if inputs are degenerate).
    """
    if not isinstance(haystack, np.ndarray) or not isinstance(needle, np.ndarray):
        raise TypeError("haystack and needle must be numpy arrays")
    n = int(needle.size)
    if n == 0 or haystack.size < n:
        return 0
    hay = haystack.astype(np.float64, copy=False)
    ndl = needle.astype(np.float64, copy=False)
    # Numerator: sliding dot product of needle over haystack.
    nom = np.correlate(hay, ndl, mode="valid")
    # Denominator: per-window energy of haystack times needle energy, so
    # the score is a normalised correlation (insensitive to loudness).
    energy = np.convolve(hay * hay, np.ones(n, dtype=np.float64), mode="valid")
    den = np.sqrt(energy * float(np.dot(ndl, ndl))) + 1e-8
    return int(np.argmax(nom / den))


def trim_to_region(
    audio: np.ndarray, trim_start: int, target_length: int
) -> tuple:
    """Return exactly ``target_length`` samples starting at ``trim_start``.

    Stage 4-E2 input-side frame restoration. The realtime worker feeds the
    backend ``[left_context][chunk][tail_pad]`` so the structural ~20 ms
    tail-frame loss falls inside the tail pad, then resamples the whole
    model output to the stream SR. This function extracts the chunk's own
    region from that resampled output as a **sample-accurate slice** —
    no stretch, no pitch shift, no proportional rounding.

    * ``trim_start`` = the left-context region to drop (= context_size at
      the stream SR). The probe confirmed the model's front render maps
      cleanly (no start deficit), so the chunk begins exactly here.
    * ``target_length`` = chunk_size at the stream SR (what we emit).

    Returns ``(out, shortfall_frames)`` where ``out.size == target_length``
    always. ``shortfall_frames`` is how many trailing samples had to be
    zero-padded because ``audio`` was shorter than ``trim_start +
    target_length`` (0 in healthy operation, since the tail pad covers the
    deficit). A non-zero shortfall is the signal that the tail pad was not
    large enough for that chunk.
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a numpy array, got {type(audio).__name__}"
        )
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D mono, got shape {audio.shape}")
    if int(trim_start) < 0:
        raise ValueError(f"trim_start must be >= 0; got {trim_start}")
    if int(target_length) < 0:
        raise ValueError(f"target_length must be >= 0; got {target_length}")

    trim_start = int(trim_start)
    target_length = int(target_length)
    if target_length == 0:
        return np.zeros(0, dtype=np.float32), 0

    region = audio[trim_start: trim_start + target_length].astype(
        np.float32, copy=True
    )
    shortfall = target_length - region.size
    if shortfall > 0:
        region = np.concatenate(
            [region, np.zeros(shortfall, dtype=np.float32)]
        ).astype(np.float32, copy=False)
    return region, int(shortfall)
