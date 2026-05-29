"""Realtime RVC worker thread — the one and only worker loop.

The worker owns all heavy work so the audio callbacks never block:

    in_queue (mic blocks) -> accumulate -> RVC inference -> resample
                          -> exact slice -> out_queue (output blocks)

Faithful-carrier contract (do not violate):

* The model defines the voice. Between ``engine.infer_array`` and the
  output queue the worker does ONLY what is structurally required to
  carry the model's samples to the stream:
    - resample the model's native SR to the stream SR (sinc-polyphase),
    - a sample-accurate slice that drops the input-side warm-up context
      and the look-ahead tail pad,
    - NaN/Inf scrub (safety, not shaping).
  No pitch shift, no time-stretch, no crossfade, no EQ, no gain shaping.

* Continuity is achieved input-side, not by reshaping the output:
    - LEFT context (``context_size``): the model is fed
      ``[prev_input_tail, chunk]`` as warm-up; the context region of the
      render is sliced away. This is real past audio, faithfully sliced.
    - look-ahead TAIL pad (``tail_pad_size``): the chunked pipeline emits
      exactly one ~20 ms HuBERT frame less than the input demands, and the
      loss sits at the tail. Feeding ``[context][chunk][tail_pad]`` so the
      lost frame lands in the pad — then slicing exactly ``chunk_size`` —
      keeps the emitted timeline drift-free with no stretch and no pitch
      change. The tail is the next chunk's real audio, so every emitted
      sample is rendered with real neighbours.

* Safety net: on ANY backend error (CUDA OOM, NaN, model fault) the
  worker emits the chunk's own audio (identity passthrough) for that one
  chunk. The link stays alive — the user hears themselves, never silence
  or a crash. ``rvc_fallback_count`` makes it observable.
"""

from __future__ import annotations

import queue
import time
from typing import Optional

import numpy as np

from ..audio.chunker import (
    BlockAccumulator,
    ChunkerConfig,
    find_sola_offset,
    resample_audio,
    trim_to_region,
)
from ..safety.guard import scrub_nan_inf


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
    context_size: int = 0,
    tail_pad_size: int = 0,
    sola_search_size: int = 0,
    sola_link_size: int = 0,
    drop_stale_input: bool = True,
    fallback_to_identity_on_error: bool = True,
    silence_rms_threshold: float = 0.0,
    silence_hangover_chunks: int = 0,
    poll_timeout_seconds: float = 0.1,
) -> None:
    """Run the realtime RVC worker until shutdown. See module docstring.

    SilenceFront (borrowed from w-okada): when ``silence_rms_threshold > 0`` and
    a chunk's linear RMS falls below it, the inference pipeline is skipped and
    that chunk is emitted as ``chunk_size`` zeros — silence in, silence out, a
    faithful no-op (no voiced sample is reshaped). ``silence_hangover_chunks``
    keeps processing for that many chunks after the last voiced chunk so soft /
    trailing syllables are never clipped. Default (threshold 0.0) = disabled.
    """
    if engine is None:
        raise ValueError("engine must not be None")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if output_block_size <= 0:
        raise ValueError("output_block_size must be > 0")
    if context_size < 0:
        raise ValueError("context_size must be >= 0")
    if tail_pad_size < 0:
        raise ValueError("tail_pad_size must be >= 0")
    if sola_search_size < 0 or sola_link_size < 0:
        raise ValueError("sola_search_size and sola_link_size must be >= 0")
    if silence_rms_threshold < 0:
        raise ValueError("silence_rms_threshold must be >= 0 (0 disables)")
    if silence_hangover_chunks < 0:
        raise ValueError("silence_hangover_chunks must be >= 0")

    silence_rms_threshold = float(silence_rms_threshold)
    silence_hangover_chunks = int(silence_hangover_chunks)
    chunk_size = int(chunk_size)
    context_size = int(context_size)
    tail_pad_size = int(tail_pad_size)
    sample_rate = int(sample_rate)

    # SOLA seam alignment needs room: left context to search backwards into,
    # and a tail pad to search forwards into. Clamp so the search window can
    # never read outside the render (and disable if there is no left context).
    sola_search_size = int(sola_search_size)
    sola_link_size = int(sola_link_size)
    if context_size < sola_search_size + sola_link_size:
        sola_search_size = 0
    if tail_pad_size < sola_search_size:
        sola_search_size = min(sola_search_size, tail_pad_size)
    sola_enabled = sola_search_size > 0 and sola_link_size > 0

    output_acc = BlockAccumulator(ChunkerConfig(chunk_size=int(output_block_size)))
    # Zero-initialised so the very first chunk sees a silent past (a true
    # start-of-signal), then carries forward the last processed chunk's tail.
    context_buffer: Optional[np.ndarray] = (
        np.zeros(context_size, dtype=np.float32) if context_size > 0 else None
    )
    # The last samples we actually emitted, kept as a phase signature for
    # SOLA alignment of the next chunk's seam. None = align at the anchor
    # (first chunk, or after an identity fallback emitted non-model audio).
    sola_link: Optional[np.ndarray] = None
    stream_buf = np.zeros(0, dtype=np.float32)
    saw_shutdown = False
    # SilenceFront hangover: chunks still to process after the last voiced one.
    voiced_hangover = 0
    silence_enabled = silence_rms_threshold > 0.0

    chunk_ms_budget = float(chunk_size) * 1000.0 / float(sample_rate)
    metrics.rvc_chunk_ms_budget = chunk_ms_budget
    metrics.input_tail_pad_frames = tail_pad_size
    metrics.input_tail_pad_ms = float(tail_pad_size) * 1000.0 / float(sample_rate)

    # A chunk can be processed only once chunk + look-ahead samples exist.
    required = chunk_size + tail_pad_size

    def _emit(buf: np.ndarray) -> None:
        if buf.size == 0:
            return
        for sub in output_acc.feed(buf):
            try:
                out_queue.put_nowait(sub)
                metrics.rvc_output_blocks_enqueued += 1
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

    def _refresh_context(consumed_chunk: np.ndarray) -> None:
        nonlocal context_buffer
        if context_buffer is None or context_size <= 0:
            return
        if consumed_chunk.size >= context_size:
            context_buffer = consumed_chunk[-context_size:].astype(np.float32, copy=True)
        else:
            new_ctx = np.zeros(context_size, dtype=np.float32)
            new_ctx[-consumed_chunk.size:] = consumed_chunk
            context_buffer = new_ctx

    def _fallback_emit(chunk: np.ndarray) -> None:
        nonlocal sola_link
        # The user's own voice is not a model render, so it is not a valid
        # phase reference for the next chunk's seam — reset the link.
        sola_link = None
        metrics.rvc_fallback_count += 1
        metrics.rvc_chunks_processed += 1
        _emit(_scrub(chunk.astype(np.float32, copy=True)))

    def _silence_skip(chunk: np.ndarray) -> None:
        # SilenceFront: input below the silence floor -> emit chunk_size zeros
        # (silence in, silence out: faithful, no voiced sample touched), skip
        # the whole inference pipeline. Refresh context from this real (silent)
        # chunk so the next voiced chunk warms up on its true left neighbour
        # (fixes w-okada's stale-context-on-resume weakness), and reset the SOLA
        # link since emitted zeros are not a model render.
        nonlocal sola_link
        sola_link = None
        metrics.rvc_silence_skipped_count += 1
        _refresh_context(chunk)
        _emit(np.zeros(chunk_size, dtype=np.float32))

    def _seam_aligned_start(processed: np.ndarray) -> int:
        """Pick the slice start that phase-aligns this render's seam with the
        previously emitted tail (SOLA alignment — chooses where to cut, never
        modifies samples). Falls back to the plain context anchor."""
        if not sola_enabled or sola_link is None:
            return context_size
        lo = context_size - sola_search_size - sola_link_size
        hi = context_size + sola_search_size
        if lo < 0 or hi > processed.size:
            return context_size
        offset = find_sola_offset(processed[lo:hi], sola_link)
        start = context_size - sola_search_size + offset
        metrics.rvc_sola_applied_count += 1
        metrics.rvc_sola_offset_last = start - context_size
        return start

    def _handle_chunk(chunk: np.ndarray, tail: np.ndarray) -> None:
        nonlocal sola_link
        # Build model input: [context?][chunk][tail?].
        parts = []
        if context_buffer is not None and context_size > 0:
            parts.append(context_buffer)
        parts.append(chunk)
        if tail.size > 0:
            parts.append(tail)
        model_in = parts[0] if len(parts) == 1 else np.concatenate(parts)

        t0 = time.perf_counter()
        try:
            processed, result_sr = engine.infer_array(model_in, sample_rate)
            processed = np.asarray(processed, dtype=np.float32).reshape(-1)
            result_sr = int(result_sr)
        except Exception:
            if not fallback_to_identity_on_error:
                raise
            # Safety net: emit the user's own voice for this chunk.
            metrics.record_inference_ms(
                (time.perf_counter() - t0) * 1000.0, budget_ms=chunk_ms_budget
            )
            _fallback_emit(chunk)
            return
        metrics.record_inference_ms(
            (time.perf_counter() - t0) * 1000.0, budget_ms=chunk_ms_budget
        )

        # Degenerate backend output -> identity fallback for this chunk.
        if processed.size == 0 or result_sr <= 0:
            _fallback_emit(chunk)
            return

        # Carry the model's samples to the stream SR (required adaptation).
        if result_sr != sample_rate:
            t_rs = time.perf_counter()
            processed = resample_audio(processed, result_sr, sample_rate)
            metrics.record_resample_ms((time.perf_counter() - t_rs) * 1000.0)
        processed = _scrub(processed)

        # Sample-accurate slice of exactly chunk_size model samples. The start
        # is the context anchor, nudged by SOLA so the seam phase-matches the
        # previous chunk's emitted tail. No stretch, no pitch, no blend.
        emit_start = _seam_aligned_start(processed)
        emit, shortfall = trim_to_region(processed, emit_start, chunk_size)
        if shortfall > 0:
            metrics.frame_restoration_shortfall_count += 1
        if sola_link_size > 0 and emit.size >= sola_link_size:
            sola_link = emit[-sola_link_size:].astype(np.float32, copy=True)
        metrics.rvc_chunks_processed += 1
        _emit(emit)

    while not stop_event.is_set():
        # Block briefly for the first arriving block; the timeout keeps the
        # loop responsive to stop_event when the mic is silent.
        try:
            first_block = in_queue.get(timeout=poll_timeout_seconds)
        except queue.Empty:
            continue
        if first_block is shutdown_sentinel:
            break

        # Drain everything queued while the previous inference was running.
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

        # Stability fail-safe: if inference falls behind the mic, drop whole
        # oldest chunks so latency stays bounded instead of drifting upward.
        # Refresh the context buffer from each dropped chunk so the surviving
        # chunk still warms up on its TRUE left neighbour (the audio that
        # immediately precedes it), not on a stale, non-adjacent chunk.
        if drop_stale_input:
            while stream_buf.size >= required + chunk_size:
                _refresh_context(stream_buf[:chunk_size])
                stream_buf = stream_buf[chunk_size:]
                metrics.rvc_stale_chunk_drops += 1

        # Process every fully-available window. Consume chunk_size per
        # iteration; the look-ahead tail stays as the next chunk's head.
        while stream_buf.size >= required:
            chunk = stream_buf[:chunk_size].astype(np.float32, copy=True)
            tail = stream_buf[chunk_size:chunk_size + tail_pad_size].astype(
                np.float32, copy=True
            )
            stream_buf = stream_buf[chunk_size:]

            # SilenceFront gate: skip inference on sub-threshold chunks, but a
            # hangover keeps processing for a short tail after voiced audio so
            # soft / trailing syllables are never clipped.
            if silence_enabled:
                rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
                if rms >= silence_rms_threshold:
                    voiced_hangover = silence_hangover_chunks
                elif voiced_hangover > 0:
                    voiced_hangover -= 1
                else:
                    _silence_skip(chunk)
                    continue

            _handle_chunk(chunk, tail)
            _refresh_context(chunk)

        if saw_shutdown:
            break

    # Shutdown: flush a final full chunk (silence tail — no future input
    # remains) so the last ~chunk_ms is not lost, then any partial block.
    if stream_buf.size >= chunk_size:
        chunk = stream_buf[:chunk_size].astype(np.float32, copy=True)
        tail = np.zeros(tail_pad_size, dtype=np.float32)
        _handle_chunk(chunk, tail)
        _refresh_context(chunk)

    leftover = output_acc.flush_pending()
    if leftover.size > 0:
        try:
            out_queue.put_nowait(leftover)
            metrics.rvc_output_blocks_enqueued += 1
        except queue.Full:
            metrics.output_queue_drops += 1
            metrics.rvc_output_blocks_dropped += 1
