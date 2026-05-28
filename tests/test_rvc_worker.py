"""Tests for the Stage 2 RVC worker loop, using a fake engine.

The worker is exercised through real ``queue.Queue`` instances and a
real ``threading.Event``, but the RVC engine is faked so no torch /
infer_rvc_python install is needed. The fake engine deterministically
scales the audio by 0.5 so we can verify chunk wiring end-to-end.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

from src.engine.worker import rvc_worker_loop
from src.safety.metrics import RuntimeMetrics


SENTINEL = object()


class _FakeEngine:
    """Duck-typed engine: just needs ``infer_array(audio, sr)``."""

    def __init__(self, gain: float = 0.5, raise_on_call: bool = False) -> None:
        self.gain = float(gain)
        self.raise_on_call = raise_on_call
        self.call_count = 0
        self.last_chunk_size = None

    def infer_array(self, audio, sample_rate):
        self.call_count += 1
        self.last_chunk_size = int(audio.size)
        if self.raise_on_call:
            raise RuntimeError("simulated RVC failure")
        return audio.astype(np.float32, copy=False) * np.float32(self.gain), int(sample_rate)


def _drain(q: "queue.Queue"):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# ---------------------------------------------------------------------------
# Happy path: accumulate into one chunk, infer, split into output blocks
# ---------------------------------------------------------------------------

def test_rvc_worker_processes_one_chunk_and_emits_blocks():
    chunk_size = 2400      # 50 ms at 48 kHz, exact multiple of block_size
    block_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _FakeEngine(gain=0.5)

    # 5 input blocks of 480 == 2400 == one chunk
    for i in range(5):
        in_q.put(np.full(block_size, 0.4, dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
    )

    out_blocks = _drain(out_q)
    assert engine.call_count == 1
    assert engine.last_chunk_size == chunk_size
    assert metrics.rvc_chunks_processed == 1
    assert metrics.rvc_inference_count == 1
    assert metrics.rvc_inference_mean_ms >= 0.0
    assert metrics.rvc_fallback_count == 0
    # Should have emitted exactly chunk_size / block_size = 5 blocks.
    assert len(out_blocks) == 5
    concat = np.concatenate(out_blocks)
    assert concat.shape == (chunk_size,)
    np.testing.assert_allclose(concat, 0.4 * 0.5 * np.ones(chunk_size, dtype=np.float32),
                               atol=1e-6)


# ---------------------------------------------------------------------------
# Fallback path: engine raises -> identity output, link stays alive
# ---------------------------------------------------------------------------

def test_rvc_worker_falls_back_to_identity_on_engine_error():
    chunk_size = 1920
    block_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _FakeEngine(raise_on_call=True)

    expected = np.full(chunk_size, 0.25, dtype=np.float32)
    for i in range(4):
        in_q.put(expected[i * block_size:(i + 1) * block_size])
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
    )

    assert metrics.rvc_fallback_count == 1
    assert metrics.rvc_chunks_processed == 1  # we did emit a chunk via identity
    out_blocks = _drain(out_q)
    concat = np.concatenate(out_blocks)
    np.testing.assert_allclose(concat, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Scrub: engine returns NaN/Inf -> scrubbed and counted
# ---------------------------------------------------------------------------

class _NaNyEngine:
    """Engine that injects NaN/Inf into the output."""

    def infer_array(self, audio, sample_rate):
        out = audio.astype(np.float32, copy=True)
        out[0] = np.nan
        out[1] = np.inf
        out[2] = -np.inf
        return out, int(sample_rate)


def test_rvc_worker_scrubs_nan_inf_from_engine_output():
    chunk_size = 480
    block_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=8)
    out_q: "queue.Queue" = queue.Queue(maxsize=8)
    metrics = RuntimeMetrics()
    stop = threading.Event()

    in_q.put(np.full(chunk_size, 0.1, dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        _NaNyEngine(), in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
    )

    assert metrics.nan_inf_scrub_count == 3
    out_blocks = _drain(out_q)
    assert len(out_blocks) == 1
    assert np.all(np.isfinite(out_blocks[0]))


# ---------------------------------------------------------------------------
# Stop event
# ---------------------------------------------------------------------------

def test_rvc_worker_stop_event_exits_cleanly():
    metrics = RuntimeMetrics()
    stop = threading.Event()
    stop.set()
    in_q: "queue.Queue" = queue.Queue(maxsize=4)
    out_q: "queue.Queue" = queue.Queue(maxsize=4)
    rvc_worker_loop(
        _FakeEngine(), in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=480,
        output_block_size=480,
        crossfade_size=0,
        poll_timeout_seconds=0.01,
    )
    assert metrics.rvc_chunks_processed == 0
    assert metrics.rvc_fallback_count == 0


# ---------------------------------------------------------------------------
# Crossfade smoke test
# ---------------------------------------------------------------------------

def test_rvc_worker_crossfade_preserves_output_length():
    """With crossfade enabled, the worker holds back the last K samples
    of each chunk for blending. After N chunks the total emitted should
    be (N * chunk_size - K). When shutdown flushes the pending tail
    there should be (N * chunk_size) emitted total."""
    chunk_size = 1920
    block_size = 480
    crossfade = 240

    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _FakeEngine(gain=1.0)  # identity gain for easier accounting

    # Two full chunks of input
    for i in range(2 * (chunk_size // block_size)):
        in_q.put(np.full(block_size, 0.2, dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=crossfade,
        poll_timeout_seconds=0.05,
    )

    assert engine.call_count == 2
    assert metrics.rvc_chunks_processed == 2
    out_blocks = _drain(out_q)
    total = sum(b.size for b in out_blocks)
    # With a single hold-tail crossfade, each chunk after the first
    # contributes (chunk_size - crossfade) net samples, plus the very
    # first chunk contributes (chunk_size - crossfade), plus the final
    # pending tail (crossfade) is emitted on flush.
    # = (chunk_size - crossfade) + (chunk_size - crossfade) + crossfade
    # = 2 * chunk_size - crossfade.
    assert total == 2 * chunk_size - crossfade


# ---------------------------------------------------------------------------
# Inference timing metrics
# ---------------------------------------------------------------------------

def test_record_inference_ms_updates_mean_and_max():
    metrics = RuntimeMetrics()
    metrics.record_inference_ms(10.0)
    metrics.record_inference_ms(20.0)
    metrics.record_inference_ms(30.0)
    assert metrics.rvc_inference_count == 3
    assert metrics.rvc_inference_last_ms == pytest.approx(30.0)
    assert metrics.rvc_inference_max_ms == pytest.approx(30.0)
    assert metrics.rvc_inference_mean_ms == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Stage 2C: sample-rate mismatch safety
# ---------------------------------------------------------------------------

class _WrongSrEngine:
    """Returns audio at a sample rate different from what the worker asked for.

    This simulates a backend that ignores ``resample_sr`` or returns its
    own native rate (e.g. the 40 kHz kiki model into a 48 kHz stream).
    """

    def __init__(self, returned_sr: int) -> None:
        self.returned_sr = int(returned_sr)
        self.call_count = 0

    def infer_array(self, audio, sample_rate):
        self.call_count += 1
        return audio.astype(np.float32, copy=True), self.returned_sr


def test_rvc_worker_resamples_when_returned_sr_mismatches_stream_sr():
    """Stage 2D: if the backend returns valid audio at a different SR
    than the stream uses, the worker now resamples to match (cheap
    linear interp) instead of falling back to identity. This unlocks
    the kiki@native_sr=40k path which is the only realtime-fits config."""
    chunk_size = 4800        # 100 ms @ 48 kHz
    block_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _WrongSrEngine(returned_sr=40000)  # model native SR

    input_audio = np.full(chunk_size, 0.3, dtype=np.float32)
    for i in range(chunk_size // block_size):
        in_q.put(input_audio[i * block_size:(i + 1) * block_size])
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
    )

    assert engine.call_count == 1
    # No fallback — the resample path took over.
    assert metrics.rvc_fallback_count == 0
    # The fake engine returns the input audio at 40 kHz; the worker
    # resamples back to 48 kHz, so the output length should match the
    # original input chunk length when rounded.
    out_blocks = _drain(out_q)
    total = sum(b.size for b in out_blocks)
    # 4800 samples at 40 kHz -> resampled to 48 kHz -> 5760 samples,
    # then chunked into 480-sample blocks; output_block_size divides it.
    assert total == 5760


def test_rvc_worker_falls_back_when_backend_returns_invalid_sr():
    """A backend returning result_sr <= 0 is a hard error -> identity."""
    chunk_size = 4800
    block_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _WrongSrEngine(returned_sr=0)  # invalid

    input_audio = np.full(chunk_size, 0.3, dtype=np.float32)
    for i in range(chunk_size // block_size):
        in_q.put(input_audio[i * block_size:(i + 1) * block_size])
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
    )

    assert metrics.rvc_fallback_count == 1
    out_blocks = _drain(out_q)
    concat = np.concatenate(out_blocks)
    np.testing.assert_allclose(concat, input_audio, atol=1e-6)


def test_rvc_worker_accepts_matching_sr():
    """If the backend returns the same SR the worker asked for, the
    safety net must NOT fire."""
    in_q: "queue.Queue" = queue.Queue(maxsize=8)
    out_q: "queue.Queue" = queue.Queue(maxsize=8)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _WrongSrEngine(returned_sr=48000)

    in_q.put(np.full(480, 0.2, dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=480,
        output_block_size=480,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
    )

    assert metrics.rvc_fallback_count == 0
    assert metrics.rvc_chunks_processed == 1
