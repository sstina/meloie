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
        drop_stale_input=False,
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
    the kiki@native_sr=40k path which is the only realtime-fits config.

    Stage 4-E note: this test asserts the *post-SR-resample* emit size
    (5760 samples). The new timeline-reconciliation step would otherwise
    stretch that to chunk_size (4800). We pin reconcile to ``off`` here
    so the test continues to exercise only the SR-mismatch path.
    """
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
        reconcile_timeline_method="off",
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


# ---------------------------------------------------------------------------
# Stage 2E: stale-input drop, output enqueue counting, resample timing,
# first_real_output_seen flag
# ---------------------------------------------------------------------------

def test_rvc_worker_drops_stale_chunks_when_multiple_ready():
    """When drained input produces more than one chunk in a single
    worker cycle, only the latest is processed; the older ones are
    counted as stale drops."""
    chunk_size = 480     # 10 ms @ 48 kHz, tiny on purpose
    block_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _FakeEngine(gain=1.0)

    # Pre-load 3 chunks worth of input before the worker starts so the
    # initial drain will produce 3 chunks at once.
    for i in range(3):
        in_q.put(np.full(block_size, 0.1 * (i + 1), dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
        drop_stale_input=True,
    )

    # Engine should have been called only for the latest chunk.
    assert engine.call_count == 1
    assert metrics.rvc_stale_chunk_drops == 2
    assert metrics.rvc_chunks_processed == 1


def test_rvc_worker_does_not_drop_when_drop_stale_off():
    chunk_size = 480
    block_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _FakeEngine(gain=1.0)

    for i in range(3):
        in_q.put(np.full(block_size, 0.1 * (i + 1), dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
        drop_stale_input=False,
    )

    assert engine.call_count == 3
    assert metrics.rvc_stale_chunk_drops == 0


def test_rvc_worker_counts_output_blocks_enqueued():
    chunk_size = 2400      # 5 blocks at 480
    block_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _FakeEngine(gain=1.0)

    for _ in range(5):
        in_q.put(np.full(block_size, 0.2, dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
    )

    # One chunk produced 5 output blocks.
    assert metrics.rvc_output_blocks_enqueued == 5
    assert metrics.rvc_output_blocks_dropped == 0
    assert metrics.first_real_output_seen is True
    assert metrics.max_output_queue_depth >= 1


def test_rvc_worker_records_resample_timing_on_sr_mismatch():
    chunk_size = 4800
    block_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _WrongSrEngine(returned_sr=40000)

    for _ in range(chunk_size // block_size):
        in_q.put(np.full(block_size, 0.3, dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
    )

    assert metrics.rvc_resample_count == 1
    assert metrics.rvc_resample_total_ms >= 0.0
    assert metrics.rvc_resample_mean_ms >= 0.0


# ---------------------------------------------------------------------------
# Stage 3: input-side left-context with proportional output trim
# ---------------------------------------------------------------------------

class _RecordingEngine:
    """Engine that records every input shape it was called with and
    returns its input unchanged (identity gain, same SR).

    This makes it trivial to assert what the worker actually sent to
    ``infer_array`` — specifically that the left-context was prepended.
    """

    def __init__(self) -> None:
        self.calls: list = []   # list of (np.ndarray copy, sr)

    def infer_array(self, audio, sample_rate):
        self.calls.append((audio.astype(np.float32, copy=True), int(sample_rate)))
        return audio.astype(np.float32, copy=False), int(sample_rate)


def test_rvc_worker_prepends_left_context_to_engine_input():
    """With context_size > 0 the engine sees chunk_size + context_size
    samples per call; the first call's context is zeros (true start-of-
    signal); subsequent calls' context is the previous chunk's tail."""
    chunk_size = 2400
    block_size = 480
    context_size = 480     # 10 ms @ 48 kHz; convenient block multiple

    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _RecordingEngine()

    chunk_1 = np.linspace(0.1, 0.5, chunk_size, dtype=np.float32)
    chunk_2 = np.linspace(0.5, 0.9, chunk_size, dtype=np.float32)
    for i in range(chunk_size // block_size):
        in_q.put(chunk_1[i * block_size:(i + 1) * block_size].copy())
    for i in range(chunk_size // block_size):
        in_q.put(chunk_2[i * block_size:(i + 1) * block_size].copy())
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        context_size=context_size,
        poll_timeout_seconds=0.05,
        drop_stale_input=False,
    )

    assert len(engine.calls) == 2
    a0, _ = engine.calls[0]
    a1, _ = engine.calls[1]
    # Each engine input is context_size + chunk_size long.
    assert a0.size == context_size + chunk_size
    assert a1.size == context_size + chunk_size
    # First call's context is zeros.
    np.testing.assert_array_equal(a0[:context_size], np.zeros(context_size, dtype=np.float32))
    # First call's new region matches chunk_1 exactly.
    np.testing.assert_allclose(a0[context_size:], chunk_1, atol=1e-6)
    # Second call's context is chunk_1's tail.
    np.testing.assert_allclose(a1[:context_size], chunk_1[-context_size:], atol=1e-6)
    np.testing.assert_allclose(a1[context_size:], chunk_2, atol=1e-6)


def test_rvc_worker_trims_output_proportionally_to_preserve_chunk_duration():
    """With an identity-gain engine, the EMITTED output per chunk must
    equal the chunk_size in input samples — no timeline drift."""
    chunk_size = 2400
    block_size = 480
    context_size = 480

    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _RecordingEngine()

    for _ in range(3 * (chunk_size // block_size)):
        in_q.put(np.full(block_size, 0.25, dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        context_size=context_size,
        poll_timeout_seconds=0.05,
        drop_stale_input=False,
    )

    assert len(engine.calls) == 3
    out_blocks = _drain(out_q)
    total = sum(b.size for b in out_blocks)
    # 3 chunks * 2400 samples each = exactly 7200 emitted samples.
    # (Identity engine -> 1:1 input/output ratio -> trim removes exactly
    # context_size samples per chunk.)
    assert total == 3 * chunk_size


def test_rvc_worker_zero_context_is_identical_to_legacy_behaviour():
    """context_size=0 must reproduce the no-context legacy path: the
    engine sees the chunk only, and no trim is applied."""
    chunk_size = 1920
    block_size = 480

    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _RecordingEngine()

    payload = np.linspace(-0.5, 0.5, chunk_size, dtype=np.float32)
    for i in range(chunk_size // block_size):
        in_q.put(payload[i * block_size:(i + 1) * block_size].copy())
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        context_size=0,
        poll_timeout_seconds=0.05,
    )

    assert len(engine.calls) == 1
    a0, _ = engine.calls[0]
    assert a0.size == chunk_size
    np.testing.assert_allclose(a0, payload, atol=1e-6)
    out_blocks = _drain(out_q)
    total = sum(b.size for b in out_blocks)
    assert total == chunk_size


def test_rvc_worker_fallback_path_does_not_apply_trim():
    """When the engine raises, the worker emits the chunk's own audio
    via identity fallback. The trim path (which assumes the model ran
    on chunk+context) must NOT engage in that case, otherwise the
    listener loses the first ``context_size`` samples of fallback
    audio."""
    chunk_size = 2400
    block_size = 480
    context_size = 480

    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _FakeEngine(raise_on_call=True)

    payload = np.full(chunk_size, 0.42, dtype=np.float32)
    for i in range(chunk_size // block_size):
        in_q.put(payload[i * block_size:(i + 1) * block_size].copy())
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        context_size=context_size,
        poll_timeout_seconds=0.05,
    )

    assert metrics.rvc_fallback_count == 1
    out_blocks = _drain(out_q)
    total = sum(b.size for b in out_blocks)
    assert total == chunk_size
    concat = np.concatenate(out_blocks)
    np.testing.assert_allclose(concat, payload, atol=1e-6)


def test_rvc_worker_rejects_negative_context_size():
    in_q: "queue.Queue" = queue.Queue()
    out_q: "queue.Queue" = queue.Queue()
    with pytest.raises(ValueError):
        rvc_worker_loop(
            _FakeEngine(), in_q, out_q, RuntimeMetrics(), threading.Event(), SENTINEL,
            sample_rate=48000, chunk_size=480, output_block_size=480,
            context_size=-1,
            poll_timeout_seconds=0.01,
        )


class _SlowEngine:
    """Engine that sleeps for a configurable wall-clock duration on each
    inference call so we can test Stage 4-C over-budget tracking without
    actually loading a model."""

    def __init__(self, sleep_seconds: float) -> None:
        self.sleep_seconds = float(sleep_seconds)

    def infer_array(self, audio, sample_rate):
        import time as _time
        _time.sleep(self.sleep_seconds)
        return audio.astype(np.float32, copy=False), int(sample_rate)


def test_rvc_worker_passes_chunk_ms_budget_into_record_inference_ms():
    """Stage 4-C: the worker derives the per-chunk budget from
    chunk_size / sample_rate and passes it into record_inference_ms,
    so an over-budget inference shows up in the spike counters.

    Uses a small chunk and a 50 ms sleep so the inference reliably
    exceeds the ~10 ms budget."""
    chunk_size = 480       # 10 ms @ 48 kHz -> budget = 10 ms
    block_size = 480

    in_q: "queue.Queue" = queue.Queue(maxsize=8)
    out_q: "queue.Queue" = queue.Queue(maxsize=8)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _SlowEngine(sleep_seconds=0.05)   # 50 ms >> 10 ms budget

    in_q.put(np.full(chunk_size, 0.1, dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.01,
    )

    assert metrics.rvc_chunk_ms_budget == pytest.approx(10.0)
    assert metrics.rvc_inference_count == 1
    assert metrics.rvc_inference_over_budget_count == 1
    assert metrics.rvc_inference_over_budget_max_consecutive == 1
    # The over-budget debt should be roughly (50 - 10) = 40 ms
    # but with timing slop, just check it's positive and < the
    # actual measured last-inference time.
    assert 30.0 < metrics.rvc_inference_over_budget_total_ms <= metrics.rvc_inference_last_ms


class _DeficitEngine:
    """Engine that drops a fixed tail of N samples from the output.

    Simulates the kiki backend's structural 20 ms framing loss
    (`infer_rvc_python` returns ~960 fewer samples per call than the
    input chunk demands, regardless of input length). Useful for end-
    to-end Stage 4-E reconciliation testing without loading a model.
    """

    def __init__(self, deficit_samples: int = 960) -> None:
        self.deficit_samples = int(deficit_samples)
        self.call_count = 0

    def infer_array(self, audio, sample_rate):
        self.call_count += 1
        n = max(0, audio.size - self.deficit_samples)
        return audio[:n].astype(np.float32, copy=True), int(sample_rate)


def test_rvc_worker_reconciles_each_chunk_to_chunk_size_by_default():
    """Stage 4-E default (polyphase): every chunk's emit must equal
    chunk_size in samples, regardless of the model's structural deficit."""
    chunk_size = 4800     # 100 ms @ 48 kHz
    block_size = 480
    deficit = 96          # 2 ms loss @ 48 kHz, ~analogous to kiki's 20 ms

    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _DeficitEngine(deficit_samples=deficit)

    # 3 full chunks of input.
    for _ in range(3 * (chunk_size // block_size)):
        in_q.put(np.full(block_size, 0.2, dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
        drop_stale_input=False,
        # reconcile_timeline_method defaults to "polyphase"
    )

    out_blocks = _drain(out_q)
    total = sum(b.size for b in out_blocks)
    # 3 chunks * chunk_size samples each = no drain, no per-chunk deficit.
    assert total == 3 * chunk_size
    assert metrics.timeline_reconcile_enabled is True
    assert metrics.timeline_reconcile_method == "polyphase"
    assert metrics.timeline_reconcile_count == 3
    assert metrics.timeline_expected_output_frames_total == 3 * chunk_size
    # actual = chunk_size - deficit, three chunks
    assert metrics.timeline_actual_output_frames_total == 3 * (chunk_size - deficit)
    assert metrics.timeline_reconciled_output_frames_total == 3 * chunk_size
    # signed cumulative error = -3 * deficit
    assert metrics.timeline_reconciliation_total_frame_error == -3 * deficit
    assert metrics.timeline_max_reconciliation_frames_per_chunk == deficit


def test_rvc_worker_off_mode_preserves_legacy_drain_behavior():
    """``--reconcile-timeline-method off`` must reproduce the Stage 4-D
    drain: emit size matches what the model returned, NOT chunk_size."""
    chunk_size = 4800
    block_size = 480
    deficit = 96

    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _DeficitEngine(deficit_samples=deficit)

    for _ in range(3 * (chunk_size // block_size)):
        in_q.put(np.full(block_size, 0.2, dtype=np.float32))
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        poll_timeout_seconds=0.05,
        drop_stale_input=False,
        reconcile_timeline_method="off",
    )

    out_blocks = _drain(out_q)
    total = sum(b.size for b in out_blocks)
    # Without reconciliation, the worker emits chunk_size - deficit per
    # chunk; over 3 chunks that's 3 * (chunk_size - deficit). The final
    # block_size leftover bookkeeping rounds the count, so verify the
    # drain is at least ``3 * deficit`` samples short of full timeline.
    assert total <= 3 * chunk_size - (3 * deficit) + block_size
    assert total >= 3 * (chunk_size - deficit) - block_size
    assert metrics.timeline_reconcile_enabled is False
    assert metrics.timeline_reconcile_count == 0


def test_rvc_worker_reconciliation_skipped_on_identity_fallback():
    """Identity fallback path emits the chunk's own audio (already at
    chunk_size). The reconciliation must NOT count that, otherwise the
    error totals would falsely include fallback frames."""
    chunk_size = 2400
    block_size = 480

    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _FakeEngine(raise_on_call=True)

    for _ in range(chunk_size // block_size):
        in_q.put(np.full(block_size, 0.25, dtype=np.float32))
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
    total = sum(b.size for b in out_blocks)
    # Fallback emits chunk_size samples directly; reconciliation does NOT
    # apply (the chain already matches the timeline by virtue of being
    # the original chunk).
    assert total == chunk_size
    assert metrics.timeline_reconcile_count == 0


def test_rvc_worker_context_preserved_across_stale_drop():
    """When drop_stale_input drops intermediate chunks, the context
    buffer must carry forward from the LAST processed chunk (we never
    saw the dropped ones individually). This is documented behaviour."""
    chunk_size = 480
    block_size = 480
    context_size = 240

    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    stop = threading.Event()
    engine = _RecordingEngine()

    # 3 chunks of distinct content. The worker should pull all 3 in one
    # drain cycle and process only the latest under drop_stale_input.
    c1 = np.full(chunk_size, 0.1, dtype=np.float32)
    c2 = np.full(chunk_size, 0.2, dtype=np.float32)
    c3 = np.full(chunk_size, 0.3, dtype=np.float32)
    for blk in (c1, c2, c3):
        in_q.put(blk.copy())
    in_q.put(SENTINEL)

    rvc_worker_loop(
        engine, in_q, out_q, metrics, stop, SENTINEL,
        sample_rate=48000,
        chunk_size=chunk_size,
        output_block_size=block_size,
        crossfade_size=0,
        context_size=context_size,
        poll_timeout_seconds=0.05,
        drop_stale_input=True,
    )

    assert metrics.rvc_stale_chunk_drops == 2
    assert len(engine.calls) == 1
    a0, _ = engine.calls[0]
    # The single engine call's context is zeros (no prior chunk had been
    # processed yet); its new region is c3.
    np.testing.assert_array_equal(a0[:context_size], np.zeros(context_size, dtype=np.float32))
    np.testing.assert_allclose(a0[context_size:], c3, atol=1e-6)
