"""Direct-mode realtime worker for the stateful ``StreamingRvcEngine`` (Path A).

The **engine owns streaming continuity** (persistent 16 kHz buffer, F0 cache,
SOLA + crossfade), so this worker is deliberately thin: carry fixed-size raw
blocks in, emit the engine's output blocks out. It keeps the non-negotiable
safety net:

* On ANY engine error -> emit the user's own audio (identity passthrough) for
  that block, scrubbed for NaN/Inf. The link never dies; ``rvc_fallback_count``
  makes it observable. Persistent failure escalates: after 3 consecutive
  fallbacks the engine's streaming state is reset once (recovers a poisoned
  state at the cost of a ~context_ms re-warm), and after ~10 s of continuous
  fallback ``rvc_engine_unhealthy`` flips so the UI can say "engine down,
  passthrough" instead of degrading silently forever.
* Drop-stale fail-safe: if inference falls behind the mic, drop whole oldest
  blocks so latency stays bounded (a dropped block leaves a gap in the engine's
  continuity — acceptable only as an overload backstop).

Faithful-carrier: this worker reshapes nothing. The engine defines the voice
(its model + the sanctioned seam crossfade); the worker only moves samples.
"""

from __future__ import annotations

import queue
import time
from typing import Optional

import numpy as np

from ..audio.chunker import BlockAccumulator, ChunkerConfig
from ..safety.guard import scrub_nan_inf


def rvc_direct_worker_loop(
    engine,
    in_queue: "queue.Queue",
    out_queue: "queue.Queue",
    metrics,
    stop_event,
    shutdown_sentinel: object,
    *,
    stream_sr: int,
    block_frame: int,
    output_block_size: int,
    drop_stale_input: bool = True,
    fallback_to_identity_on_error: bool = True,
    poll_timeout_seconds: float = 0.1,
) -> None:
    """Run the direct streaming worker until shutdown. See module docstring."""
    if engine is None:
        raise ValueError("engine must not be None")
    if block_frame <= 0:
        raise ValueError("block_frame must be > 0")
    if output_block_size <= 0:
        raise ValueError("output_block_size must be > 0")

    block_frame = int(block_frame)
    stream_sr = int(stream_sr)
    output_acc = BlockAccumulator(ChunkerConfig(chunk_size=int(output_block_size)))
    stream_buf = np.zeros(0, dtype=np.float32)
    saw_shutdown = False

    block_ms_budget = float(block_frame) * 1000.0 / float(stream_sr)
    metrics.rvc_chunk_ms_budget = block_ms_budget

    # Persistent-failure escalation thresholds (consecutive fallbacks).
    reset_after = 3
    unhealthy_after = max(reset_after + 1, int(round(10_000.0 / max(block_ms_budget, 1.0))))
    consecutive_fallbacks = 0
    reset_fired = False    # one reset per failure streak

    def _emit(buf: np.ndarray) -> None:
        if buf.size == 0:
            return
        for sub in output_acc.feed(buf):
            try:
                out_queue.put_nowait(sub)
                metrics.rvc_output_blocks_enqueued += 1
                # "Real output" = past the engine's warm-up zeros; warm-up
                # underruns must keep classifying as startup, not steady-state.
                if (
                    not metrics.first_real_output_seen
                    and getattr(engine, "warmup_blocks_left", 0) <= 0
                ):
                    metrics.first_real_output_seen = True
            except queue.Full:
                metrics.output_queue_drops += 1
                metrics.rvc_output_blocks_dropped += 1

    def _scrub(buf: np.ndarray) -> np.ndarray:
        scrub = scrub_nan_inf(buf)
        if scrub.replaced_count:
            metrics.nan_inf_scrub_count += int(scrub.replaced_count)
            return scrub.audio
        return buf

    def _fallback(block: np.ndarray) -> None:
        nonlocal consecutive_fallbacks, reset_fired
        metrics.rvc_fallback_count += 1
        metrics.rvc_chunks_processed += 1
        _emit(_scrub(block.astype(np.float32, copy=True)))
        consecutive_fallbacks += 1
        if consecutive_fallbacks >= reset_after and not reset_fired:
            # A short failure streak suggests poisoned streaming state (e.g. a
            # CUDA hiccup mid-block): reset once. Costs a ~context_ms re-warm
            # (zeros) on recovery — better than identity passthrough forever.
            reset_fired = True
            try:
                engine.reset()
                metrics.rvc_engine_resets += 1
            except Exception:
                pass
        if consecutive_fallbacks >= unhealthy_after:
            metrics.rvc_engine_unhealthy = True

    while not stop_event.is_set():
        try:
            first_block = in_queue.get(timeout=poll_timeout_seconds)
        except queue.Empty:
            continue
        if first_block is shutdown_sentinel:
            break

        pulled = [first_block]
        while True:
            try:
                b = in_queue.get_nowait()
                if b is shutdown_sentinel:
                    saw_shutdown = True
                    break
                pulled.append(b)
            except queue.Empty:
                break
        depth = len(pulled) + in_queue.qsize()
        if depth > metrics.max_input_queue_depth:
            metrics.max_input_queue_depth = depth

        try:
            stream_buf = np.concatenate(
                [stream_buf]
                + [np.asarray(b, dtype=np.float32).reshape(-1) for b in pulled]
            )
        except Exception:
            metrics.rvc_fallback_count += 1
            if saw_shutdown:
                break
            continue

        # Drop-stale: keep latency bounded under overload. A dropped block leaves
        # a one-block gap in the engine's continuity (acceptable backstop only).
        if drop_stale_input:
            while stream_buf.size >= 2 * block_frame:
                stream_buf = stream_buf[block_frame:]
                metrics.rvc_stale_chunk_drops += 1

        while stream_buf.size >= block_frame:
            block = stream_buf[:block_frame].astype(np.float32, copy=True)
            stream_buf = stream_buf[block_frame:]
            t0 = time.perf_counter()
            try:
                out = engine.process_block(block, stream_sr)
                out = np.asarray(out, dtype=np.float32).reshape(-1)
            except Exception:
                if not fallback_to_identity_on_error:
                    raise
                metrics.record_inference_ms((time.perf_counter() - t0) * 1000.0)
                _fallback(block)
                continue
            metrics.record_inference_ms((time.perf_counter() - t0) * 1000.0)
            if out.size == 0:
                _fallback(block)
                continue
            # A successful block ends any failure streak.
            consecutive_fallbacks = 0
            reset_fired = False
            metrics.rvc_engine_unhealthy = False
            metrics.rvc_chunks_processed += 1
            # Surface the engine's last-block seam/gate state (engine stays
            # metrics-agnostic; the worker is the single reader/writer here).
            metrics.rvc_sola_offset_last = int(getattr(engine, "last_sola_offset", 0))
            if getattr(engine, "last_silence_skipped", False):
                metrics.rvc_silence_skipped_count += 1
            _emit(_scrub(out))

        if saw_shutdown:
            break

    leftover = output_acc.flush_pending()
    if leftover.size > 0:
        try:
            out_queue.put_nowait(leftover)
            metrics.rvc_output_blocks_enqueued += 1
        except queue.Full:
            metrics.output_queue_drops += 1
            metrics.rvc_output_blocks_dropped += 1
