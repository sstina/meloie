"""Audio worker thread.

Realtime architecture:

    input callback -> in_queue -> WORKER -> out_queue -> output callback

The audio callbacks never block; all per-block work happens here on a
dedicated worker thread. In Stage 1 the worker is pure identity. In
Stage 2 the worker body will call into ``rvc_engine`` and fall back to
identity on any exception (RVC OOM / timeout / NaN are the new failure
modes).

This module does NOT start threads on import. The streams layer starts
the thread explicitly and supplies the queues, metrics, stop event,
and shutdown sentinel.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..safety.guard import scrub_nan_inf


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

    Returns a *copy* of the input so callers can never accidentally
    mutate the audio buffer that came from the input callback.
    """
    if not isinstance(block, np.ndarray):
        raise TypeError(f"block must be a numpy array, got {type(block).__name__}")
    return block.copy()


def process_rvc(block: np.ndarray) -> np.ndarray:
    """Stage 2 worker body — not implemented in this skeleton."""
    raise NotImplementedError(
        "RVC worker mode is Stage 2 and is not implemented in this skeleton."
    )


def _process(block: np.ndarray, mode: WorkerMode) -> np.ndarray:
    if mode == WorkerMode.IDENTITY:
        return process_identity(block)
    if mode == WorkerMode.RVC_NOT_IMPLEMENTED:
        return process_rvc(block)
    raise ValueError(f"unknown worker mode: {mode!r}")


def worker_loop(
    config: "WorkerConfig",
    in_queue: "queue.Queue",
    out_queue: "queue.Queue",
    metrics,
    stop_event,
    shutdown_sentinel: object,
    poll_timeout_seconds: float = 0.1,
) -> None:
    """Run the worker until ``stop_event`` is set or the sentinel arrives.

    Per-block flow::

        block = in_queue.get(timeout=...)
        if block is sentinel or stop_event.is_set():
            break
        try:
            processed = _process(block, mode)
            scrub NaN/Inf (cheap; tracked in metrics)
        except Exception:
            if fallback_to_identity_on_error:
                processed = process_identity(block)
                metrics.fallback_count += 1
            else:
                raise
        try:
            out_queue.put_nowait(processed)
        except queue.Full:
            metrics.output_queue_drops += 1
    """
    while not stop_event.is_set():
        try:
            block = in_queue.get(timeout=poll_timeout_seconds)
        except queue.Empty:
            continue

        if block is shutdown_sentinel:
            break

        try:
            processed = _process(block, config.mode)
            scrub = scrub_nan_inf(processed)
            if scrub.replaced_count:
                metrics.nan_inf_scrub_count += int(scrub.replaced_count)
                processed = scrub.audio
        except Exception:
            if not config.fallback_to_identity_on_error:
                raise
            processed = process_identity(block)
            metrics.fallback_count += 1

        try:
            out_queue.put_nowait(processed)
        except queue.Full:
            metrics.output_queue_drops += 1
