"""Tests for the realtime RVC worker loop, using fake engines.

The worker runs through real ``queue.Queue`` instances and a real
``threading.Event``; the RVC engine is faked so no torch /
infer_rvc_python install is needed. These tests pin the faithful-carrier
contract: every processed chunk emits exactly ``chunk_size`` samples (a
sample-accurate slice — no stretch, no pitch change), the SR-adaptation
resample runs when needed, and a backend error falls back to the user's
own voice rather than killing the link.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

from src.audio.chunker import find_sola_offset, resample_audio
from src.engine.worker import rvc_worker_loop
from src.safety.metrics import RuntimeMetrics


SENTINEL = object()


class _FakeEngine:
    """Duck-typed engine: just needs ``infer_array(audio, sr)``."""

    def __init__(self, gain: float = 0.5, raise_on_call: bool = False) -> None:
        self.gain = float(gain)
        self.raise_on_call = raise_on_call
        self.call_count = 0
        self.last_input_size = None

    def infer_array(self, audio, sample_rate):
        self.call_count += 1
        self.last_input_size = int(audio.size)
        if self.raise_on_call:
            raise RuntimeError("simulated RVC failure")
        return audio.astype(np.float32, copy=False) * np.float32(self.gain), int(sample_rate)


def _drain(q: "queue.Queue"):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _run(engine, in_q, out_q, metrics, **kw):
    kw.setdefault("sample_rate", 48000)
    kw.setdefault("output_block_size", 480)
    kw.setdefault("poll_timeout_seconds", 0.05)
    rvc_worker_loop(engine, in_q, out_q, metrics, threading.Event(), SENTINEL, **kw)


# ---------------------------------------------------------------------------
# Happy path: one chunk -> infer -> emit chunk_size as output blocks
# ---------------------------------------------------------------------------

def test_processes_one_chunk_and_emits_chunk_size():
    chunk_size = 2400      # 50 ms @ 48 kHz, exact multiple of block_size
    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    engine = _FakeEngine(gain=0.5)

    for _ in range(5):  # 5 * 480 == 2400 == one chunk
        in_q.put(np.full(480, 0.4, dtype=np.float32))
    in_q.put(SENTINEL)

    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size)

    out_blocks = _drain(out_q)
    assert engine.call_count == 1
    assert engine.last_input_size == chunk_size      # no context, no tail
    assert metrics.rvc_chunks_processed == 1
    assert metrics.rvc_inference_count == 1
    assert metrics.rvc_fallback_count == 0
    concat = np.concatenate(out_blocks)
    assert concat.shape == (chunk_size,)
    np.testing.assert_allclose(concat, 0.2 * np.ones(chunk_size, np.float32), atol=1e-6)


# ---------------------------------------------------------------------------
# Safety net: engine raises -> identity output (own voice), link alive
# ---------------------------------------------------------------------------

def test_falls_back_to_identity_on_engine_error():
    chunk_size = 1920
    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    engine = _FakeEngine(raise_on_call=True)

    expected = np.full(chunk_size, 0.25, dtype=np.float32)
    for i in range(4):
        in_q.put(expected[i * 480:(i + 1) * 480])
    in_q.put(SENTINEL)

    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size)

    assert metrics.rvc_fallback_count == 1
    assert metrics.rvc_chunks_processed == 1
    concat = np.concatenate(_drain(out_q))
    np.testing.assert_allclose(concat, expected, atol=1e-6)


def test_falls_back_when_backend_returns_invalid_sr():
    chunk_size = 4800

    class _ZeroSrEngine:
        def infer_array(self, audio, sample_rate):
            return audio.astype(np.float32, copy=True), 0  # invalid SR

    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    payload = np.full(chunk_size, 0.3, dtype=np.float32)
    for i in range(chunk_size // 480):
        in_q.put(payload[i * 480:(i + 1) * 480])
    in_q.put(SENTINEL)

    _run(_ZeroSrEngine(), in_q, out_q, metrics, chunk_size=chunk_size)

    assert metrics.rvc_fallback_count == 1
    np.testing.assert_allclose(np.concatenate(_drain(out_q)), payload, atol=1e-6)


# ---------------------------------------------------------------------------
# NaN/Inf scrub
# ---------------------------------------------------------------------------

def test_scrubs_nan_inf_from_engine_output():
    class _NaNyEngine:
        def infer_array(self, audio, sample_rate):
            out = audio.astype(np.float32, copy=True)
            out[0], out[1], out[2] = np.nan, np.inf, -np.inf
            return out, int(sample_rate)

    in_q: "queue.Queue" = queue.Queue(maxsize=8)
    out_q: "queue.Queue" = queue.Queue(maxsize=8)
    metrics = RuntimeMetrics()
    in_q.put(np.full(480, 0.1, dtype=np.float32))
    in_q.put(SENTINEL)

    _run(_NaNyEngine(), in_q, out_q, metrics, chunk_size=480)

    assert metrics.nan_inf_scrub_count == 3
    out_blocks = _drain(out_q)
    assert len(out_blocks) == 1
    assert np.all(np.isfinite(out_blocks[0]))


# ---------------------------------------------------------------------------
# Stop event
# ---------------------------------------------------------------------------

def test_stop_event_exits_cleanly():
    metrics = RuntimeMetrics()
    stop = threading.Event()
    stop.set()
    rvc_worker_loop(
        _FakeEngine(), queue.Queue(), queue.Queue(), metrics, stop, SENTINEL,
        sample_rate=48000, chunk_size=480, output_block_size=480,
        poll_timeout_seconds=0.01,
    )
    assert metrics.rvc_chunks_processed == 0
    assert metrics.rvc_fallback_count == 0


# ---------------------------------------------------------------------------
# SR adaptation: backend returns its native SR -> worker resamples, then
# still emits exactly chunk_size via the slice.
# ---------------------------------------------------------------------------

def test_resamples_native_sr_and_emits_chunk_size():
    class _NativeSrEngine:
        """Returns the audio resampled to 40 kHz (duration preserved),
        like the kiki model's 40 kHz native output."""

        def __init__(self):
            self.call_count = 0

        def infer_array(self, audio, sample_rate):
            self.call_count += 1
            return resample_audio(audio, int(sample_rate), 40000), 40000

    chunk_size = 4800     # 100 ms @ 48 kHz
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    engine = _NativeSrEngine()
    for _ in range(chunk_size // 480):
        in_q.put(np.full(480, 0.3, dtype=np.float32))
    in_q.put(SENTINEL)

    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size)

    assert engine.call_count == 1
    assert metrics.rvc_fallback_count == 0
    assert metrics.rvc_resample_count == 1
    # The slice guarantees exactly chunk_size emitted regardless of native SR.
    assert sum(b.size for b in _drain(out_q)) == chunk_size


def test_no_resample_when_sr_matches():
    in_q: "queue.Queue" = queue.Queue(maxsize=8)
    out_q: "queue.Queue" = queue.Queue(maxsize=8)
    metrics = RuntimeMetrics()
    in_q.put(np.full(480, 0.2, dtype=np.float32))
    in_q.put(SENTINEL)
    _run(_FakeEngine(gain=1.0), in_q, out_q, metrics, chunk_size=480)
    assert metrics.rvc_resample_count == 0
    assert metrics.rvc_chunks_processed == 1


# ---------------------------------------------------------------------------
# Input-side left-context warm-up (faithful: sliced away)
# ---------------------------------------------------------------------------

class _RecordingEngine:
    """Records every input it was called with; returns it unchanged."""

    def __init__(self) -> None:
        self.calls: list = []

    def infer_array(self, audio, sample_rate):
        self.calls.append(audio.astype(np.float32, copy=True))
        return audio.astype(np.float32, copy=False), int(sample_rate)


def test_prepends_left_context_and_emits_chunk_size():
    chunk_size = 2400
    context_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    engine = _RecordingEngine()

    c1 = np.linspace(0.1, 0.5, chunk_size, dtype=np.float32)
    c2 = np.linspace(0.5, 0.9, chunk_size, dtype=np.float32)
    for i in range(chunk_size // 480):
        in_q.put(c1[i * 480:(i + 1) * 480].copy())
    for i in range(chunk_size // 480):
        in_q.put(c2[i * 480:(i + 1) * 480].copy())
    in_q.put(SENTINEL)

    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size,
         context_size=context_size, drop_stale_input=False)

    assert len(engine.calls) == 2
    a0, a1 = engine.calls
    assert a0.size == context_size + chunk_size
    # First call's context is zeros (true start-of-signal).
    np.testing.assert_array_equal(a0[:context_size], np.zeros(context_size, np.float32))
    np.testing.assert_allclose(a0[context_size:], c1, atol=1e-6)
    # Second call's context is chunk_1's tail.
    np.testing.assert_allclose(a1[:context_size], c1[-context_size:], atol=1e-6)
    # Emit duration is exactly chunk_size per chunk (no drift).
    assert sum(b.size for b in _drain(out_q)) == 2 * chunk_size


def test_zero_context_feeds_chunk_only():
    chunk_size = 1920
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    engine = _RecordingEngine()
    payload = np.linspace(-0.5, 0.5, chunk_size, dtype=np.float32)
    for i in range(chunk_size // 480):
        in_q.put(payload[i * 480:(i + 1) * 480].copy())
    in_q.put(SENTINEL)
    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size, context_size=0)
    assert len(engine.calls) == 1
    assert engine.calls[0].size == chunk_size
    assert sum(b.size for b in _drain(out_q)) == chunk_size


# ---------------------------------------------------------------------------
# Look-ahead tail pad restores the model's structural ~20 ms tail deficit
# without stretch/pitch: emit stays exactly chunk_size, no queue drain.
# ---------------------------------------------------------------------------

class _DeficitEngine:
    """Drops a fixed tail of N samples (simulates kiki's framing loss)."""

    def __init__(self, deficit_samples: int = 96) -> None:
        self.deficit_samples = int(deficit_samples)
        self.call_count = 0

    def infer_array(self, audio, sample_rate):
        self.call_count += 1
        n = max(0, audio.size - self.deficit_samples)
        return audio[:n].astype(np.float32, copy=True), int(sample_rate)


def test_lookahead_tail_pad_keeps_emit_at_chunk_size():
    chunk_size = 4800
    deficit = 96
    tail_pad = 192        # > deficit so the lost frame lands in the pad
    in_q: "queue.Queue" = queue.Queue(maxsize=128)
    out_q: "queue.Queue" = queue.Queue(maxsize=128)
    metrics = RuntimeMetrics()
    engine = _DeficitEngine(deficit_samples=deficit)

    # 32 blocks = 15360 samples: 3 chunks each with a 192-sample look-ahead
    # tail, leaving < chunk_size so the shutdown flush does not add a 4th.
    for _ in range(32):
        in_q.put(np.full(480, 0.2, dtype=np.float32))
    in_q.put(SENTINEL)

    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size,
         drop_stale_input=False, tail_pad_size=tail_pad)

    assert engine.call_count == 3
    assert sum(b.size for b in _drain(out_q)) == 3 * chunk_size
    assert metrics.frame_restoration_shortfall_count == 0
    assert metrics.input_tail_pad_frames == tail_pad


# ---------------------------------------------------------------------------
# Stability fail-safe: drop-stale keeps latency bounded
# ---------------------------------------------------------------------------

def test_drop_stale_drops_oldest_when_behind():
    chunk_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    engine = _FakeEngine(gain=1.0)
    for i in range(3):  # 3 chunks pre-loaded, drained at once
        in_q.put(np.full(480, 0.1 * (i + 1), dtype=np.float32))
    in_q.put(SENTINEL)
    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size, drop_stale_input=True)
    assert engine.call_count == 1
    assert metrics.rvc_stale_chunk_drops == 2
    assert metrics.rvc_chunks_processed == 1


def test_drop_stale_refreshes_context_to_true_left_neighbour():
    """When drop-stale discards chunks, the surviving chunk must warm up on
    its TRUE left neighbour (the last dropped chunk's tail), not on stale,
    non-adjacent audio. Regression guard for the context-on-drop fix."""
    chunk_size = 480
    context_size = 240
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    engine = _RecordingEngine()
    c1 = np.full(chunk_size, 0.1, dtype=np.float32)
    c2 = np.full(chunk_size, 0.2, dtype=np.float32)  # immediate left neighbour of c3
    c3 = np.full(chunk_size, 0.3, dtype=np.float32)
    for blk in (c1, c2, c3):
        in_q.put(blk.copy())
    in_q.put(SENTINEL)

    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size,
         context_size=context_size, drop_stale_input=True)

    assert metrics.rvc_stale_chunk_drops == 2
    assert len(engine.calls) == 1
    a0 = engine.calls[0]
    # Context is c2's tail (the true left neighbour), NOT zeros / c-something-stale.
    np.testing.assert_allclose(a0[:context_size], c2[-context_size:], atol=1e-6)
    np.testing.assert_allclose(a0[context_size:], c3, atol=1e-6)


def test_no_drop_when_drop_stale_off():
    chunk_size = 480
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    engine = _FakeEngine(gain=1.0)
    for i in range(3):
        in_q.put(np.full(480, 0.1 * (i + 1), dtype=np.float32))
    in_q.put(SENTINEL)
    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size, drop_stale_input=False)
    assert engine.call_count == 3
    assert metrics.rvc_stale_chunk_drops == 0


# ---------------------------------------------------------------------------
# Output enqueue accounting
# ---------------------------------------------------------------------------

def test_counts_output_blocks_enqueued():
    chunk_size = 2400      # 5 blocks at 480
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    for _ in range(5):
        in_q.put(np.full(480, 0.2, dtype=np.float32))
    in_q.put(SENTINEL)
    _run(_FakeEngine(gain=1.0), in_q, out_q, metrics, chunk_size=chunk_size)
    assert metrics.rvc_output_blocks_enqueued == 5
    assert metrics.rvc_output_blocks_dropped == 0
    assert metrics.first_real_output_seen is True


# ---------------------------------------------------------------------------
# Inference timing + validation
# ---------------------------------------------------------------------------

def test_record_inference_ms_updates_mean_and_max():
    metrics = RuntimeMetrics()
    for ms in (10.0, 20.0, 30.0):
        metrics.record_inference_ms(ms)
    assert metrics.rvc_inference_count == 3
    assert metrics.rvc_inference_last_ms == pytest.approx(30.0)
    assert metrics.rvc_inference_max_ms == pytest.approx(30.0)
    assert metrics.rvc_inference_mean_ms == pytest.approx(20.0)


def test_rejects_negative_context_size():
    with pytest.raises(ValueError):
        rvc_worker_loop(
            _FakeEngine(), queue.Queue(), queue.Queue(), RuntimeMetrics(),
            threading.Event(), SENTINEL,
            sample_rate=48000, chunk_size=480, output_block_size=480,
            context_size=-1, poll_timeout_seconds=0.01,
        )


def test_rejects_zero_chunk_size():
    with pytest.raises(ValueError):
        rvc_worker_loop(
            _FakeEngine(), queue.Queue(), queue.Queue(), RuntimeMetrics(),
            threading.Event(), SENTINEL,
            sample_rate=48000, chunk_size=0, output_block_size=480,
            poll_timeout_seconds=0.01,
        )


# ---------------------------------------------------------------------------
# SOLA seam alignment (faithful: chooses the cut offset, never edits samples)
# ---------------------------------------------------------------------------

def test_find_sola_offset_locates_known_alignment():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(3000).astype(np.float32)
    needle = sig[1000:1120]            # 120-sample phase signature
    haystack = sig[940:1180]           # needle sits at index 60
    assert find_sola_offset(haystack, needle) == 60


def test_find_sola_offset_degenerate_returns_zero():
    assert find_sola_offset(np.zeros(0, np.float32), np.ones(4, np.float32)) == 0
    assert find_sola_offset(np.ones(3, np.float32), np.ones(5, np.float32)) == 0


def _feed_signal(in_q, signal, block=480):
    n = signal.size - (signal.size % block)
    for i in range(0, n, block):
        in_q.put(signal[i:i + block].copy())


def test_sola_identity_aligns_at_anchor_with_no_drift():
    """With an identity engine, consecutive renders are byte-identical, so
    SOLA must find the seam already aligned (offset 0) and emit exactly
    chunk_size per chunk — no timeline drift."""
    chunk_size, ctx, tail = 4800, 2400, 480
    search, link = 240, 480
    in_q: "queue.Queue" = queue.Queue(maxsize=256)
    out_q: "queue.Queue" = queue.Queue(maxsize=256)
    metrics = RuntimeMetrics()
    rng = np.random.default_rng(7)
    sig = (0.2 * rng.standard_normal(chunk_size * 3 + tail + 960)).astype(np.float32)
    _feed_signal(in_q, sig)
    in_q.put(SENTINEL)

    _run(_FakeEngine(gain=1.0), in_q, out_q, metrics,
         chunk_size=chunk_size, context_size=ctx, tail_pad_size=tail,
         sola_search_size=search, sola_link_size=link, drop_stale_input=False)

    total = sum(b.size for b in _drain(out_q))
    assert metrics.rvc_chunks_processed >= 2
    # Exactly chunk_size emitted per chunk — SOLA introduced no drift.
    assert total == metrics.rvc_chunks_processed * chunk_size
    # Every chunk after the first ran SOLA, and identical renders align at 0.
    assert metrics.rvc_sola_applied_count == metrics.rvc_chunks_processed - 1
    assert metrics.rvc_sola_offset_last == 0


def test_sola_disabled_emits_at_context_anchor():
    chunk_size, ctx = 4800, 2400
    in_q: "queue.Queue" = queue.Queue(maxsize=256)
    out_q: "queue.Queue" = queue.Queue(maxsize=256)
    metrics = RuntimeMetrics()
    rng = np.random.default_rng(3)
    sig = (0.2 * rng.standard_normal(chunk_size * 2 + 960)).astype(np.float32)
    _feed_signal(in_q, sig)
    in_q.put(SENTINEL)

    _run(_FakeEngine(gain=1.0), in_q, out_q, metrics,
         chunk_size=chunk_size, context_size=ctx, tail_pad_size=480,
         sola_search_size=0, sola_link_size=0, drop_stale_input=False)

    assert metrics.rvc_sola_applied_count == 0
    total = sum(b.size for b in _drain(out_q))
    assert total == metrics.rvc_chunks_processed * chunk_size


def test_sola_disabled_when_context_too_small():
    """If left context can't fit search+link, SOLA self-disables (no crash,
    no drift) rather than reading outside the render."""
    chunk_size = 4800
    in_q: "queue.Queue" = queue.Queue(maxsize=256)
    out_q: "queue.Queue" = queue.Queue(maxsize=256)
    metrics = RuntimeMetrics()
    rng = np.random.default_rng(5)
    sig = (0.2 * rng.standard_normal(chunk_size * 2 + 960)).astype(np.float32)
    _feed_signal(in_q, sig)
    in_q.put(SENTINEL)

    _run(_FakeEngine(gain=1.0), in_q, out_q, metrics,
         chunk_size=chunk_size, context_size=240,   # < search+link
         tail_pad_size=480, sola_search_size=240, sola_link_size=480,
         drop_stale_input=False)

    assert metrics.rvc_sola_applied_count == 0
    total = sum(b.size for b in _drain(out_q))
    assert total == metrics.rvc_chunks_processed * chunk_size


# ---------------------------------------------------------------------------
# SilenceFront (w-okada borrow): RMS-gated silence skip, faithful + hangover.
# ---------------------------------------------------------------------------

def test_silence_skips_below_threshold_emitting_zeros():
    chunk_size = 2400  # 5 * 480
    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    engine = _FakeEngine(gain=0.5)

    for _ in range(5):
        in_q.put(np.zeros(480, dtype=np.float32))
    in_q.put(SENTINEL)

    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size,
         silence_rms_threshold=0.01, silence_hangover_chunks=0)

    assert engine.call_count == 0                 # inference skipped entirely
    assert metrics.rvc_silence_skipped_count == 1
    assert metrics.rvc_chunks_processed == 0
    concat = np.concatenate(_drain(out_q))
    assert concat.shape == (chunk_size,)          # still emits chunk_size...
    np.testing.assert_array_equal(concat, np.zeros(chunk_size, np.float32))  # ...of zeros


def test_silence_processes_above_threshold():
    chunk_size = 2400
    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    engine = _FakeEngine(gain=0.5)

    for _ in range(5):
        in_q.put(np.full(480, 0.4, dtype=np.float32))   # RMS 0.4 >> 0.01
    in_q.put(SENTINEL)

    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size,
         silence_rms_threshold=0.01, silence_hangover_chunks=0)

    assert engine.call_count == 1
    assert metrics.rvc_silence_skipped_count == 0
    assert metrics.rvc_chunks_processed == 1


def test_silence_disabled_by_default_processes_silent_chunk():
    """The safe default (threshold 0.0) must NEVER skip — a silent chunk is
    still run through inference, so soft speech can never be gated out."""
    chunk_size = 2400
    in_q: "queue.Queue" = queue.Queue(maxsize=32)
    out_q: "queue.Queue" = queue.Queue(maxsize=32)
    metrics = RuntimeMetrics()
    engine = _FakeEngine(gain=0.5)

    for _ in range(5):
        in_q.put(np.zeros(480, dtype=np.float32))
    in_q.put(SENTINEL)

    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size)  # no silence args

    assert engine.call_count == 1
    assert metrics.rvc_silence_skipped_count == 0
    assert metrics.rvc_chunks_processed == 1


def test_silence_hangover_keeps_quiet_chunk_then_skips():
    """One loud chunk then quiet: with hangover=1, the first quiet chunk is
    still processed (tail protection); the next quiet chunk is skipped."""
    chunk_size = 2400
    in_q: "queue.Queue" = queue.Queue(maxsize=64)
    out_q: "queue.Queue" = queue.Queue(maxsize=64)
    metrics = RuntimeMetrics()
    engine = _FakeEngine(gain=0.5)

    for _ in range(5):                              # chunk 1: loud
        in_q.put(np.full(480, 0.4, dtype=np.float32))
    for _ in range(10):                             # chunks 2 & 3: quiet
        in_q.put(np.zeros(480, dtype=np.float32))
    in_q.put(SENTINEL)

    _run(engine, in_q, out_q, metrics, chunk_size=chunk_size,
         silence_rms_threshold=0.01, silence_hangover_chunks=1,
         drop_stale_input=False)                    # process all 3 chunks in order

    assert engine.call_count == 2                   # loud + 1 hangover chunk
    assert metrics.rvc_chunks_processed == 2
    assert metrics.rvc_silence_skipped_count == 1   # the 3rd (post-hangover)
