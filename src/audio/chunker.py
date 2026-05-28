"""Pure block -> chunk accumulator helpers.

RVC inference operates on chunks of ~150-200 ms, but ``sounddevice``
callbacks deliver small blocks (~10 ms). The accumulator here collects
small mono float32 blocks until a fixed chunk size is reached, then
emits the chunk.

This module is intentionally free of realtime / threading complexity.
It is just a numpy buffer with a flush rule, so it can be unit tested
without any audio hardware.
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
