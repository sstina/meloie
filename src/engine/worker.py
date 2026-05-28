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

from ..audio.chunker import BlockAccumulator, ChunkerConfig, resample_audio
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
    drop_stale_input: bool = True,
    context_size: int = 0,
) -> None:
    """Chunked RVC worker.

    Per-block flow::

        block = in_queue.get(timeout=...)
        append to input_acc (BlockAccumulator at chunk_size)
        for each full chunk:
            if context_size > 0:
                input_to_model = concat(context_buffer, chunk)
            else:
                input_to_model = chunk
            t0 = perf_counter()
            try:
                processed, _sr = engine.infer_array(input_to_model, sample_rate)
            except Exception:
                processed = identity(chunk)   # fallback, link stays alive
                metrics.rvc_fallback_count += 1
            if context_size > 0:
                # Trim leading region proportionally — preserves the chunk's
                # output duration exactly while letting the model see real
                # previous audio as warmup context.
                trim = round(context_size * processed.size / input_to_model.size)
                processed = processed[trim:]
                context_buffer = chunk[-context_size:]
            scrub NaN/Inf
            crossfade with previous chunk's tail (if crossfade_size > 0)
            feed processed into output_acc (BlockAccumulator at block_size)
            push each emitted block to out_queue (put_nowait; full -> drop++)

    Stage 3 — input-left-context (``context_size > 0``):

    Per-chunk inference inherits no past audio across chunk boundaries,
    so HuBERT / F0 / index re-initialise on every call. The first
    ~tens of ms of every chunk's output exhibits cold-start instability
    (boundary clicks, F0 wobble, sustained-vowel flutter). The
    model-faithful fix is to feed the model some real previous input
    audio as left-context, then discard the proportional output region.

    * Each inference receives ``[context_buffer, chunk]`` (``context_size
      + chunk_size`` samples). On the very first chunk the buffer is
      zeros, which mirrors a true start-of-signal.
    * The output is trimmed by ``round(context_size * out_len / in_len)``
      samples from the front. This preserves the chunk's nominal
      output duration exactly (no timeline drift).
    * ``context_buffer`` is then refreshed to ``chunk[-context_size:]``
      so the next call sees the correct preceding audio.
    * On ``drop_stale_input`` engagement, intermediate chunks' tails
      are not propagated; the buffer carries the LAST processed
      chunk's tail. This is a known small discontinuity that only
      fires when inference falls behind; identity fallback semantics
      are unchanged.

    On shutdown, flush any pending crossfade tail and partial block.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if output_block_size <= 0:
        raise ValueError("output_block_size must be > 0")
    if crossfade_size < 0:
        raise ValueError("crossfade_size must be >= 0")
    if context_size < 0:
        raise ValueError("context_size must be >= 0")
    if engine is None:
        raise ValueError("engine must not be None")

    input_acc = BlockAccumulator(ChunkerConfig(chunk_size=int(chunk_size)))
    output_acc = BlockAccumulator(ChunkerConfig(chunk_size=int(output_block_size)))
    pending_tail: Optional[np.ndarray] = None
    saw_shutdown = False
    # Stage 3 input-left-context buffer. Zero-initialised so the very
    # first chunk sees a silent past (matches true start-of-signal).
    context_buffer: Optional[np.ndarray] = (
        np.zeros(int(context_size), dtype=np.float32) if context_size > 0 else None
    )

    # Stage 4-C: per-chunk wall-clock budget for the inference call.
    # If an inference exceeds this it consumes part of the output-queue
    # safety margin. ``record_inference_ms(..., budget_ms=)`` tracks how
    # often this happens, the longest consecutive streak, and the total
    # over-budget debt in ms.
    chunk_ms_budget: float = float(chunk_size) * 1000.0 / float(sample_rate)
    metrics.rvc_chunk_ms_budget = chunk_ms_budget

    def _emit(buf: np.ndarray) -> None:
        if buf.size == 0:
            return
        for sub in output_acc.feed(buf):
            try:
                out_queue.put_nowait(sub)
                metrics.rvc_output_blocks_enqueued += 1
                if not metrics.first_real_output_seen:
                    metrics.first_real_output_seen = True
                qd = out_queue.qsize()
                if qd > metrics.max_output_queue_depth:
                    metrics.max_output_queue_depth = qd
            except queue.Full:
                metrics.output_queue_drops += 1
                metrics.rvc_output_blocks_dropped += 1

    while not stop_event.is_set():
        # Wait for at least one input block. ``poll_timeout_seconds`` keeps
        # the loop responsive to ``stop_event`` even when the mic is silent.
        try:
            first_block = in_queue.get(timeout=poll_timeout_seconds)
        except queue.Empty:
            continue
        if first_block is shutdown_sentinel:
            break

        # Drain everything that arrived while we were busy with the
        # previous chunk's inference. This is the fix for the live-vs-
        # benchmark gap: without this drain, we processed 640 ms-stale
        # audio on every cycle.
        pulled_blocks = [first_block]
        while True:
            try:
                b = in_queue.get_nowait()
                if b is shutdown_sentinel:
                    saw_shutdown = True
                    break
                pulled_blocks.append(b)
            except queue.Empty:
                break
        # Sample max input queue depth right after we just drained — qsize
        # is sticky-low at that point, so we count what we just removed.
        observed_depth = len(pulled_blocks) + in_queue.qsize()
        if observed_depth > metrics.max_input_queue_depth:
            metrics.max_input_queue_depth = observed_depth

        # Feed all pulled blocks into the chunk accumulator.
        chunks: list = []
        try:
            for b in pulled_blocks:
                chunks.extend(input_acc.feed(b))
        except Exception:
            metrics.rvc_fallback_count += 1
            if saw_shutdown:
                break
            continue

        # If multiple chunks are ready and stale-drop is on, keep only the
        # latest — the older chunks would otherwise add latency without
        # adding listenable content (we'd never catch up).
        if drop_stale_input and len(chunks) > 1:
            metrics.rvc_stale_chunk_drops += len(chunks) - 1
            chunks = chunks[-1:]

        for chunk in chunks:
            # Stage 3: prepend left-context before sending to the model.
            # ``input_to_model`` is what the engine sees; ``chunk`` is the
            # caller's "new audio" that should map to chunk_size samples
            # of OUTPUT after trimming.
            if context_buffer is not None and context_size > 0:
                input_to_model = np.concatenate([context_buffer, chunk])
            else:
                input_to_model = chunk
            t0 = time.perf_counter()
            result_sr = int(sample_rate)
            used_fallback = False
            try:
                processed, result_sr_raw = engine.infer_array(
                    input_to_model, int(sample_rate)
                )
                processed = np.asarray(processed, dtype=np.float32).reshape(-1)
                result_sr = int(result_sr_raw)
            except Exception:
                if not fallback_to_identity_on_error:
                    raise
                processed = chunk.astype(np.float32, copy=True)
                metrics.rvc_fallback_count += 1
                used_fallback = True
            metrics.record_inference_ms(
                (time.perf_counter() - t0) * 1000.0,
                budget_ms=chunk_ms_budget,
            )

            # Trim the leading region that corresponds to the context
            # input. Skipped when context is off, when the model failed
            # (the fallback is already the chunk's own audio), or when
            # the backend returned a degenerate output (the SR-handling
            # block below will treat that as fallback anyway).
            if (
                context_size > 0
                and not used_fallback
                and processed.size > 0
                and input_to_model.size > 0
            ):
                trim = int(round(
                    int(context_size) * processed.size / input_to_model.size
                ))
                if trim < 0:
                    trim = 0
                if trim >= processed.size:
                    # Defensive: would leave nothing; treat as fallback so the
                    # chain stays alive. Should never happen in practice.
                    processed = chunk.astype(np.float32, copy=True)
                    metrics.rvc_fallback_count += 1
                else:
                    processed = processed[trim:]

            # Refresh context buffer for the next inference. We refresh
            # from the chunk's actual input regardless of whether the
            # model succeeded — the next chunk's correct left-context is
            # always "this chunk's tail" in input time.
            if context_buffer is not None and context_size > 0:
                if chunk.size >= context_size:
                    context_buffer = chunk[-context_size:].astype(np.float32, copy=True)
                else:
                    # Partial chunk (shorter than context). Pad-shift.
                    new_ctx = np.zeros(context_size, dtype=np.float32)
                    new_ctx[-chunk.size:] = chunk
                    context_buffer = new_ctx

            # SR handling. The kiki model returns 40 kHz natively while the
            # stream is 48 kHz; asking infer_rvc_python to resample
            # internally (resample_sr=stream_sr) is too slow for realtime
            # (Stage 2D benchmark: +230 ms / chunk). The worker resamples
            # post-model instead, using ``resample_audio`` which prefers
            # scipy polyphase (sinc-windowed) and falls back to np.interp
            # if scipy is unavailable. Audit measurement (tools/pseudo_
            # stream) showed polyphase yields ~+11 dB output-vs-reference
            # SNR upgrade vs np.interp at negligible CPU cost.
            if processed.size == 0 or result_sr <= 0:
                # Genuine garbage from backend -> identity fallback.
                processed = chunk.astype(np.float32, copy=True)
                metrics.rvc_fallback_count += 1
            elif result_sr != int(sample_rate):
                t_rs = time.perf_counter()
                processed = resample_audio(processed, result_sr, int(sample_rate))
                metrics.record_resample_ms((time.perf_counter() - t_rs) * 1000.0)

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

        if saw_shutdown:
            break

    # Shutdown: emit any held tail and partial block leftover.
    if pending_tail is not None:
        _emit(pending_tail)
    leftover = output_acc.flush_pending()
    if leftover.size > 0:
        try:
            out_queue.put_nowait(leftover)
            metrics.rvc_output_blocks_enqueued += 1
        except queue.Full:
            metrics.output_queue_drops += 1
            metrics.rvc_output_blocks_dropped += 1
