"""Audio worker scaffold.

The realtime architecture is:

    input callback -> in_queue -> WORKER -> out_queue -> output callback

The audio callbacks never block; all work happens here on a worker
thread. In Stage 1 the worker is pure identity (returns the input
unchanged). In Stage 2 the worker body will call into ``rvc_engine``
and fall back to identity on any exception.

This module does NOT start threads on import. The loop function is
provided as a placeholder; callers wire it up explicitly once the
realtime stream layer is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class WorkerMode(str, Enum):
    """Selectable worker behaviour."""

    IDENTITY = "identity"
    RVC_NOT_IMPLEMENTED = "rvc_not_implemented"


@dataclass(frozen=True)
class WorkerConfig:
    """Worker configuration. Kept minimal during Stage 1."""

    mode: WorkerMode = WorkerMode.IDENTITY
    fallback_to_identity_on_error: bool = True


def process_identity(block: np.ndarray) -> np.ndarray:
    """Stage 1 worker body. Identity passthrough.

    Returns the input array unchanged in value. Returns a new array
    (not the same object) so callers can never accidentally mutate the
    audio buffer that came from the input callback.
    """
    if not isinstance(block, np.ndarray):
        raise TypeError(f"block must be a numpy array, got {type(block).__name__}")
    return block.copy()


def process_rvc(block: np.ndarray) -> np.ndarray:
    """Stage 2 worker body — not implemented in this skeleton."""
    raise NotImplementedError(
        "RVC worker mode is Stage 2 and is not implemented in this skeleton."
    )


def worker_loop(config: WorkerConfig, in_queue, out_queue) -> None:
    """Realtime worker loop scaffold.

    Pseudocode::

        while True:
            block = in_queue.get()
            if block is None:        # sentinel for shutdown
                break
            try:
                if mode == IDENTITY:
                    processed = process_identity(block)
                elif mode == RVC_NOT_IMPLEMENTED:
                    processed = process_rvc(block)
            except Exception:
                processed = process_identity(block)   # safety fallback
                metrics.fallback_count += 1
            try:
                out_queue.put_nowait(processed)
            except queue.Full:
                metrics.drop_count += 1

    Wiring this loop into actual threads / queues is deliberately
    deferred until the streams layer lands. Calling it now raises so
    nothing here can silently start a thread on import.
    """
    raise NotImplementedError(
        "worker_loop is a Stage 1 scaffold. The realtime worker thread "
        "will be wired up in the next commit, alongside the streams "
        "layer. The shape of the loop is documented in this function's "
        "docstring."
    )
