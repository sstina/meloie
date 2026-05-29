"""Audio worker thread.

Two worker loops live here:

* ``worker_loop``    — Stage 1, identity / per-block. Used by the
                       identity stream and by tests of the fallback
                       machinery.
* ``rvc_worker_loop`` — Stage 2+, chunk-accumulating, RVC inference.
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

from ..audio.chunker import (
    BlockAccumulator,
    ChunkerConfig,
    reconcile_to_length,
    resample_audio,
    trim_to_region,
)
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
# Stage 2+: RVC chunk worker loop
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
    frame_restore_method: str = "lookahead",
    tail_pad_size: int = 0,
) -> None:
    """Chunked RVC worker with input-side frame restoration (Stage 4-E2).

    The model structurally emits exactly one 50 Hz HuBERT frame (~20 ms)
    less audio than the input demands, and the loss sits entirely at the
    **tail** (confirmed deterministic across input lengths and content —
    see ``tools/probe_frame_deficit.py``). Left uncorrected the realtime
    output queue drains at ~17 ms / s and eventually underruns.

    Production correction (``frame_restore_method`` in
    ``{"lookahead", "silence"}``) is **input-side**: feed the backend
    ``[left_context][chunk][tail_pad]`` so the dropped frame falls inside
    the tail pad, resample the whole model render to the stream SR, then
    take a **sample-accurate slice** ``[context_size : context_size +
    chunk_size]``. The emit is exactly ``chunk_size`` samples, generated
    by the model, **not time-stretched and not pitch-shifted**.

    * ``lookahead`` (default, most faithful): ``tail_pad`` is the next
      chunk's first ``tail_pad_size`` real input samples, so the chunk's
      last frame is rendered with real continuation and chunk boundaries
      stay seamless. Costs ``tail_pad_size`` of look-ahead latency: a
      chunk is processed only once ``chunk_size + tail_pad_size`` input
      samples are available; only ``chunk_size`` are consumed (the tail
      becomes the next chunk's head).
    * ``silence``: ``tail_pad`` is zeros — the dropped frame still lands
      in the pad, but the chunk's last frame is vocoded with silence as
      right-context. Zero added latency.

    Diagnostic / fallback methods preserve earlier-stage behaviour:

    * ``stretch`` — Stage 4-E output-side polyphase stretch of the model
      render up to ``chunk_size`` (~34 cents flat for kiki). No tail pad.
    * ``off`` — Stage 4-D: emit the model render verbatim (drain-prone).
      No tail pad.

    Stage 3 input-left-context (``context_size > 0``) composes with all
    methods: the engine sees real previous input as warm-up; for the
    input-side methods the exact slice drops it, for the legacy methods
    the proportional native trim does. On the very first chunk the context
    is zeros (a true start-of-signal).

    Identity fallback on any backend error emits the chunk's own audio
    (already ``chunk_size`` at the stream SR) and skips restoration. On
    shutdown a final full chunk still buffered is flushed with a silence
    tail so the last ~chunk_ms is not lost.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if output_block_size <= 0:
        raise ValueError("output_block_size must be > 0")
    if crossfade_size < 0:
        raise ValueError("crossfade_size must be >= 0")
    if context_size < 0:
        raise ValueError("context_size must be >= 0")
    if tail_pad_size < 0:
        raise ValueError("tail_pad_size must be >= 0")
    if frame_restore_method not in ("lookahead", "silence", "stretch", "off"):
        raise ValueError(
            f"frame_restore_method must be one of 'lookahead', 'silence', "
            f"'stretch', 'off'; got {frame_restore_method!r}"
        )
    if engine is None:
        raise ValueError("engine must not be None")

    chunk_size = int(chunk_size)
    context_size = int(context_size)
    tail_pad_size = int(tail_pad_size)
    sample_rate = int(sample_rate)

    output_acc = BlockAccumulator(ChunkerConfig(chunk_size=int(output_block_size)))
    pending_tail: Optional[np.ndarray] = None
    saw_shutdown = False
    # Stage 3 input-left-context buffer. Zero-initialised so the very
    # first chunk sees a silent past (matches true start-of-signal).
    context_buffer: Optional[np.ndarray] = (
        np.zeros(context_size, dtype=np.float32) if context_size > 0 else None
    )

    # Stage 4-C: per-chunk wall-clock budget for the inference call.
    chunk_ms_budget: float = float(chunk_size) * 1000.0 / float(sample_rate)
    metrics.rvc_chunk_ms_budget = chunk_ms_budget

    # Stage 4-E2: frame-restoration mode.
    input_side = frame_restore_method in ("lookahead", "silence")
    eff_tail_pad = tail_pad_size if input_side else 0
    metrics.frame_restore_method = str(frame_restore_method)
    metrics.frame_restore_enabled = bool(input_side)
    metrics.input_tail_pad_frames = int(eff_tail_pad)
    metrics.input_tail_pad_ms = float(eff_tail_pad) * 1000.0 / float(sample_rate)
    # Stage 4-E compat flags (only the 'stretch' diagnostic touches these).
    metrics.timeline_reconcile_enabled = (frame_restore_method == "stretch")
    metrics.timeline_reconcile_method = (
        "polyphase" if frame_restore_method == "stretch" else ""
    )
    reconcile_target_samples = chunk_size

    # Look-ahead needs real future samples in the buffer before a chunk can
    # be processed; silence synthesises the tail so it needs only the chunk.
    lookahead_size = eff_tail_pad if frame_restore_method == "lookahead" else 0
    required = chunk_size + lookahead_size

    stream_buf = np.zeros(0, dtype=np.float32)

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

    def _emit_with_crossfade(buf: np.ndarray) -> None:
        nonlocal pending_tail
        if crossfade_size == 0 or buf.size < (2 * crossfade_size):
            if pending_tail is not None:
                _emit(pending_tail)
                pending_tail = None
            _emit(buf)
        elif pending_tail is None:
            _emit(buf[:-crossfade_size])
            pending_tail = buf[-crossfade_size:].astype(np.float32, copy=True)
        else:
            head = buf[:crossfade_size]
            blended = linear_crossfade(pending_tail, head)
            _emit(blended)
            _emit(buf[crossfade_size:-crossfade_size])
            pending_tail = buf[-crossfade_size:].astype(np.float32, copy=True)

    def _refresh_context(consumed_chunk: np.ndarray) -> None:
        nonlocal context_buffer
        if context_buffer is None or context_size <= 0:
            return
        if consumed_chunk.size >= context_size:
            context_buffer = consumed_chunk[-context_size:].astype(
                np.float32, copy=True
            )
        else:
            new_ctx = np.zeros(context_size, dtype=np.float32)
            new_ctx[-consumed_chunk.size:] = consumed_chunk
            context_buffer = new_ctx

    def _scrub(buf: np.ndarray) -> np.ndarray:
        scrub = scrub_nan_inf(buf)
        if scrub.replaced_count:
            metrics.nan_inf_scrub_count += int(scrub.replaced_count)
            return scrub.audio
        return buf

    def _handle_chunk(chunk: np.ndarray, tail: np.ndarray) -> None:
        # Build the model input: [context?][chunk][tail?].
        parts = []
        if context_buffer is not None and context_size > 0:
            parts.append(context_buffer)
        parts.append(chunk)
        if tail.size > 0:
            parts.append(tail)
        input_to_model = parts[0] if len(parts) == 1 else np.concatenate(parts)

        t0 = time.perf_counter()
        result_sr = sample_rate
        used_fallback = False
        try:
            processed, result_sr_raw = engine.infer_array(input_to_model, sample_rate)
            processed = np.asarray(processed, dtype=np.float32).reshape(-1)
            result_sr = int(result_sr_raw)
        except Exception:
            if not fallback_to_identity_on_error:
                raise
            processed = chunk.astype(np.float32, copy=True)
            metrics.rvc_fallback_count += 1
            used_fallback = True
        metrics.record_inference_ms(
            (time.perf_counter() - t0) * 1000.0, budget_ms=chunk_ms_budget
        )

        # Identity fallback (engine raised): chunk is already chunk_size @
        # stream SR. Skip restoration.
        if used_fallback:
            processed = _scrub(processed)
            metrics.rvc_chunks_processed += 1
            _emit_with_crossfade(processed)
            return

        # Degenerate backend output -> identity fallback.
        if processed.size == 0 or result_sr <= 0:
            processed = _scrub(chunk.astype(np.float32, copy=True))
            metrics.rvc_fallback_count += 1
            metrics.rvc_chunks_processed += 1
            _emit_with_crossfade(processed)
            return

        if input_side:
            # Resample the WHOLE render to the stream SR, then take the
            # chunk's own region as an exact slice. The slice replaces both
            # the Stage-3 proportional context trim and the Stage-4-E stretch.
            if result_sr != sample_rate:
                t_rs = time.perf_counter()
                processed = resample_audio(processed, result_sr, sample_rate)
                metrics.record_resample_ms((time.perf_counter() - t_rs) * 1000.0)
            processed = _scrub(processed)
            actual_before = int(processed.size)
            trim_start = context_size
            emit, shortfall = trim_to_region(processed, trim_start, chunk_size)
            trim_end = max(0, actual_before - trim_start - chunk_size)
            metrics.record_frame_restore(
                expected=chunk_size,
                actual_before_trim=actual_before,
                emitted=int(emit.size),
                trim_start=trim_start,
                trim_end=trim_end,
                shortfall_frames=int(shortfall),
            )
            processed = emit
            metrics.rvc_chunks_processed += 1
            _emit_with_crossfade(processed)
            return

        # ---- Legacy diagnostic path: 'stretch' / 'off' ----
        # Proportional native context trim (Stage 3), in input-time terms.
        if context_size > 0 and processed.size > 0 and input_to_model.size > 0:
            trim = int(round(context_size * processed.size / input_to_model.size))
            if trim < 0:
                trim = 0
            if trim >= processed.size:
                # Defensive: would leave nothing -> identity fallback.
                processed = _scrub(chunk.astype(np.float32, copy=True))
                metrics.rvc_fallback_count += 1
                metrics.rvc_chunks_processed += 1
                _emit_with_crossfade(processed)
                return
            processed = processed[trim:]

        if result_sr != sample_rate:
            t_rs = time.perf_counter()
            processed = resample_audio(processed, result_sr, sample_rate)
            metrics.record_resample_ms((time.perf_counter() - t_rs) * 1000.0)
        processed = _scrub(processed)

        if frame_restore_method == "stretch" and processed.size > 0:
            actual = int(processed.size)
            target = reconcile_target_samples
            if actual != target:
                processed = reconcile_to_length(processed, target, method="polyphase")
                metrics.output_stretch_used_count += 1
            metrics.record_timeline_reconcile(
                expected_frames=target,
                actual_frames=actual,
                reconciled_frames=int(processed.size),
            )
        # 'off': emit the model render verbatim (drain-prone, diagnostic).

        metrics.rvc_chunks_processed += 1
        _emit_with_crossfade(processed)

    while not stop_event.is_set():
        # Wait for at least one input block. ``poll_timeout_seconds`` keeps
        # the loop responsive to ``stop_event`` even when the mic is silent.
        try:
            first_block = in_queue.get(timeout=poll_timeout_seconds)
        except queue.Empty:
            continue
        if first_block is shutdown_sentinel:
            break

        # Drain everything that arrived while we were busy with the previous
        # chunk's inference (avoids processing stale audio every cycle).
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
        observed_depth = len(pulled_blocks) + in_queue.qsize()
        if observed_depth > metrics.max_input_queue_depth:
            metrics.max_input_queue_depth = observed_depth

        # Append pulled blocks to the rolling raw input buffer.
        try:
            stream_buf = np.concatenate(
                [stream_buf]
                + [np.asarray(b, dtype=np.float32).reshape(-1) for b in pulled_blocks]
            )
        except Exception:
            metrics.rvc_fallback_count += 1
            if saw_shutdown:
                break
            continue

        # Drop-stale: if more than one full chunk is queued beyond what we
        # need for the next processable window, drop whole oldest chunks.
        # Dropped chunks do NOT update the context buffer — the documented
        # Stage-3 behaviour is that context carries forward only the last
        # *processed* chunk's tail (the skipped chunks are a known gap).
        if drop_stale_input:
            while stream_buf.size >= required + chunk_size:
                stream_buf = stream_buf[chunk_size:]
                metrics.rvc_stale_chunk_drops += 1

        # Process every fully-available window. Consume only chunk_size per
        # iteration; the look-ahead tail (if any) stays as the next head.
        while stream_buf.size >= required:
            chunk = stream_buf[:chunk_size].astype(np.float32, copy=True)
            if frame_restore_method == "lookahead":
                tail = stream_buf[chunk_size:chunk_size + tail_pad_size].astype(
                    np.float32, copy=True
                )
            elif frame_restore_method == "silence":
                tail = np.zeros(eff_tail_pad, dtype=np.float32)
            else:
                tail = np.zeros(0, dtype=np.float32)
            stream_buf = stream_buf[chunk_size:]
            _handle_chunk(chunk, tail)
            _refresh_context(chunk)

        if saw_shutdown:
            break

    # Shutdown: flush a final full chunk (with a silence tail — no future
    # input remains) so the last ~chunk_ms is not lost, then any held
    # crossfade tail and partial output block.
    if stream_buf.size >= chunk_size:
        chunk = stream_buf[:chunk_size].astype(np.float32, copy=True)
        tail = (
            np.zeros(eff_tail_pad, dtype=np.float32)
            if input_side and eff_tail_pad > 0
            else np.zeros(0, dtype=np.float32)
        )
        _handle_chunk(chunk, tail)
        _refresh_context(chunk)

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
