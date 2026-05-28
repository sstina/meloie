"""Audio worker thread.

Two worker loops live here:

* ``worker_loop``    — Stage 1, identity / per-block. Used by the
                       identity stream and by tests of the fallback
                       machinery.
* ``rvc_worker_loop`` — Stage 2, chunk-accumulating, RVC inference.
                       Used by the realtime ``--mode rvc`` stream.

Both loops do NOT start threads on import. The streams layer starts
them explicitly and supplies the queues, metrics, stop event, and
shutdown sentinel.

Realtime invariants (do not violate):

* The audio callbacks never block. Heavy work (RVC inference, NaN
  scrubbing, crossfade) happens here on the worker thread.
* RVC mode falls back to identity on any inference exception. The
  audio link stays alive; the user hears their own voice rather than
  silence or a crash.
* Output is split into ``output_block_size``-sized blocks so the
  output callback never has to truncate or zero-pad an oversized
  RVC chunk.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from ..audio.chunker import BlockAccumulator, ChunkerConfig
from ..audio.devices import iter_device_infos  # noqa: F401  (re-export friendliness)
from ..engine.crossfade import linear_crossfade
from ..safety.guard import scrub_nan_inf


class WorkerMode(str, Enum):
    """Selectable identity-worker behaviour.

    The RVC realtime path uses :func:`rvc_worker_loop`, not
    ``worker_loop`` — these values are for the identity loop only.
    ``RVC_NOT_IMPLEMENTED`` exists to test the identity fallback
    machinery (``process_rvc`` raises ``NotImplementedError`` so the
    fallback path is exercised).
    """

    IDENTITY = "identity"
    RVC_NOT_IMPLEMENTED = "rvc_not_implemented"


@dataclass(frozen=True)
class WorkerConfig:
    """Identity-worker configuration."""

    mode: WorkerMode = WorkerMode.IDENTITY
    fallback_to_identity_on_error: bool = True


def process_identity(block: np.ndarray) -> np.ndarray:
    """Stage 1 worker body. Identity passthrough returning a copy."""
    if not isinstance(block, np.ndarray):
        raise TypeError(f"block must be a numpy array, got {type(block).__name__}")
    return block.copy()


def process_rvc(block: np.ndarray) -> np.ndarray:
    """Placeholder for the identity-worker's RVC-mode test path.

    The realtime RVC path uses ``rvc_worker_loop`` and ``RvcEngine``;
    this function is intentionally still NotImplementedError to keep
    :func:`worker_loop`'s fallback test meaningful.
    """
    raise NotImplementedError(
        "RVC worker mode is Stage 2 and is not implemented in this skeleton."
    )


def _process_identity_mode(block: np.ndarray, mode: WorkerMode) -> np.ndarray:
    if mode == WorkerMode.IDENTITY:
        return process_identity(block)
    if mode == WorkerMode.RVC_NOT_IMPLEMENTED:
        return process_rvc(block)
    raise ValueError(f"unknown worker mode: {mode!r}")


# ---------------------------------------------------------------------------
# Stage 1: identity worker loop
# ---------------------------------------------------------------------------

def worker_loop(
    config: "WorkerConfig",
    in_queue: "queue.Queue",
    out_queue: "queue.Queue",
    metrics,
    stop_event,
    shutdown_sentinel: object,
    poll_timeout_seconds: float = 0.1,
) -> None:
    """Run the identity worker until shutdown."""
    while not stop_event.is_set():
        try:
            block = in_queue.get(timeout=poll_timeout_seconds)
        except queue.Empty:
            continue

        if block is shutdown_sentinel:
            break

        try:
            processed = _process_identity_mode(block, config.mode)
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


# ---------------------------------------------------------------------------
# Stage 2: RVC chunk worker loop
# ---------------------------------------------------------------------------

def rvc_worker_loop(
    engine,
    in_queue: "queue.Queue",
    out_queue: "queue.Queue",
    metrics,
    stop_event,
    shutdown_sentinel: object,
    *,
    sample_rate: int,
    chunk_size: int,
    output_block_size: int,
    crossfade_size: int = 0,
    fallback_to_identity_on_error: bool = True,
    poll_timeout_seconds: float = 0.1,
) -> None:
    """Chunked RVC worker.

    Per-block flow::

        block = in_queue.get(timeout=...)
        append to input_acc (BlockAccumulator at chunk_size)
        for each full chunk:
            t0 = perf_counter()
            try:
                processed, _sr = engine.infer_array(chunk, sample_rate)
            except Exception:
                processed = identity(chunk)   # fallback, link stays alive
                metrics.rvc_fallback_count += 1
            scrub NaN/Inf
            crossfade with previous chunk's tail (if crossfade_size > 0)
            feed processed into output_acc (BlockAccumulator at block_size)
            push each emitted block to out_queue (put_nowait; full -> drop++)

    On shutdown, flush any pending tail and partial blocks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if output_block_size <= 0:
        raise ValueError("output_block_size must be > 0")
    if crossfade_size < 0:
        raise ValueError("crossfade_size must be >= 0")
    if engine is None:
        raise ValueError("engine must not be None")

    input_acc = BlockAccumulator(ChunkerConfig(chunk_size=int(chunk_size)))
    output_acc = BlockAccumulator(ChunkerConfig(chunk_size=int(output_block_size)))
    pending_tail: Optional[np.ndarray] = None

    def _emit(buf: np.ndarray) -> None:
        if buf.size == 0:
            return
        for sub in output_acc.feed(buf):
            try:
                out_queue.put_nowait(sub)
            except queue.Full:
                metrics.output_queue_drops += 1

    while not stop_event.is_set():
        try:
            block = in_queue.get(timeout=poll_timeout_seconds)
        except queue.Empty:
            continue
        if block is shutdown_sentinel:
            break

        try:
            chunks = input_acc.feed(block)
        except Exception:
            # Bad block shape — count as fallback and drop.
            metrics.rvc_fallback_count += 1
            continue

        for chunk in chunks:
            t0 = time.perf_counter()
            try:
                processed, _result_sr = engine.infer_array(chunk, int(sample_rate))
                # Defensive: shape match for downstream splitter.
                processed = np.asarray(processed, dtype=np.float32).reshape(-1)
            except Exception:
                if not fallback_to_identity_on_error:
                    raise
                processed = chunk.astype(np.float32, copy=True)
                metrics.rvc_fallback_count += 1
            metrics.record_inference_ms((time.perf_counter() - t0) * 1000.0)

            # NaN/Inf scrub — RVC can produce these on edge cases.
            scrub = scrub_nan_inf(processed)
            if scrub.replaced_count:
                metrics.nan_inf_scrub_count += int(scrub.replaced_count)
                processed = scrub.audio

            metrics.rvc_chunks_processed += 1

            # Chunk-boundary crossfade. Hold back the last ``crossfade_size``
            # samples of each chunk; when the next chunk arrives, blend its
            # head with the held tail before emitting it.
            if crossfade_size == 0 or processed.size < (2 * crossfade_size):
                # Too short to crossfade safely — emit as-is, no tail.
                if pending_tail is not None:
                    _emit(pending_tail)
                    pending_tail = None
                _emit(processed)
            elif pending_tail is None:
                # First crossfade-eligible chunk: emit body, hold tail.
                _emit(processed[:-crossfade_size])
                pending_tail = processed[-crossfade_size:].astype(np.float32, copy=True)
            else:
                head = processed[:crossfade_size]
                blended = linear_crossfade(pending_tail, head)
                _emit(blended)
                _emit(processed[crossfade_size:-crossfade_size])
                pending_tail = processed[-crossfade_size:].astype(np.float32, copy=True)

    # Shutdown: emit any held tail and partial block leftover.
    if pending_tail is not None:
        _emit(pending_tail)
    leftover = output_acc.flush_pending()
    if leftover.size > 0:
        try:
            out_queue.put_nowait(leftover)
        except queue.Full:
            metrics.output_queue_drops += 1
