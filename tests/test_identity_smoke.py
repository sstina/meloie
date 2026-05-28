"""Stage 1 identity smoke test + import-safety guard.

This test protects three invariants that the rest of the project
relies on:

1. The Stage 1 identity worker is a true passthrough.
2. The worker loop drains an input queue, processes via identity, and
   emits to an output queue without opening audio devices.
3. Importing the top-level ``src.main`` CLI does not open any audio
   devices and does not import ``sounddevice``.
"""

from __future__ import annotations

import importlib
import queue
import sys
import threading

import numpy as np
import pytest

from src.audio.chunker import BlockAccumulator, ChunkerConfig
from src.engine.crossfade import linear_crossfade
from src.engine.rvc_engine import RvcEngine
from src.engine.worker import (
    WorkerConfig,
    WorkerMode,
    process_identity,
    process_rvc,
    worker_loop,
)
from src.safety.metrics import RuntimeMetrics


# ---------------------------------------------------------------------------
# Identity worker (pure function)
# ---------------------------------------------------------------------------

def test_identity_processing_returns_equivalent_audio():
    rng = np.random.default_rng(0)
    block = rng.standard_normal(480).astype(np.float32)
    out = process_identity(block)
    np.testing.assert_array_equal(out, block)
    # Must NOT be the same object: callers can never mutate the input
    # buffer that came from the sounddevice callback.
    assert out is not block


def test_identity_processing_preserves_silence():
    silence = np.zeros(480, dtype=np.float32)
    out = process_identity(silence)
    np.testing.assert_array_equal(out, silence)


def test_identity_processing_rejects_non_array():
    with pytest.raises(TypeError):
        process_identity([0.0, 0.1, 0.2])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# worker_loop end-to-end (with fake queues, no audio hardware)
# ---------------------------------------------------------------------------

SENTINEL = object()


def _drain(q: "queue.Queue"):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_worker_loop_identity_passthrough_via_queues():
    in_q: "queue.Queue" = queue.Queue(maxsize=8)
    out_q: "queue.Queue" = queue.Queue(maxsize=8)
    metrics = RuntimeMetrics()
    stop_event = threading.Event()

    rng = np.random.default_rng(1)
    block_a = rng.standard_normal(480).astype(np.float32)
    block_b = rng.standard_normal(480).astype(np.float32)
    in_q.put(block_a)
    in_q.put(block_b)
    in_q.put(SENTINEL)  # tells the worker to exit

    worker_loop(
        WorkerConfig(mode=WorkerMode.IDENTITY),
        in_q,
        out_q,
        metrics,
        stop_event,
        SENTINEL,
        poll_timeout_seconds=0.05,
    )

    produced = _drain(out_q)
    assert len(produced) == 2
    np.testing.assert_array_equal(produced[0], block_a)
    np.testing.assert_array_equal(produced[1], block_b)
    assert metrics.fallback_count == 0
    assert metrics.output_queue_drops == 0
    assert metrics.nan_inf_scrub_count == 0


def test_worker_loop_scrubs_nan_inf_and_counts():
    in_q: "queue.Queue" = queue.Queue(maxsize=4)
    out_q: "queue.Queue" = queue.Queue(maxsize=4)
    metrics = RuntimeMetrics()
    stop_event = threading.Event()

    dirty = np.array([0.1, np.nan, np.inf, -np.inf, 0.5], dtype=np.float32)
    in_q.put(dirty)
    in_q.put(SENTINEL)

    worker_loop(
        WorkerConfig(mode=WorkerMode.IDENTITY),
        in_q, out_q, metrics, stop_event, SENTINEL,
        poll_timeout_seconds=0.05,
    )

    produced = _drain(out_q)
    assert len(produced) == 1
    assert np.all(np.isfinite(produced[0]))
    assert metrics.nan_inf_scrub_count == 3


def test_worker_loop_rvc_mode_falls_back_to_identity():
    in_q: "queue.Queue" = queue.Queue(maxsize=4)
    out_q: "queue.Queue" = queue.Queue(maxsize=4)
    metrics = RuntimeMetrics()
    stop_event = threading.Event()

    block = np.full(480, 0.25, dtype=np.float32)
    in_q.put(block)
    in_q.put(SENTINEL)

    worker_loop(
        WorkerConfig(
            mode=WorkerMode.RVC_NOT_IMPLEMENTED,
            fallback_to_identity_on_error=True,
        ),
        in_q, out_q, metrics, stop_event, SENTINEL,
        poll_timeout_seconds=0.05,
    )

    produced = _drain(out_q)
    assert len(produced) == 1
    np.testing.assert_array_equal(produced[0], block)
    assert metrics.fallback_count == 1


def test_worker_loop_stop_event_exits_cleanly():
    in_q: "queue.Queue" = queue.Queue(maxsize=4)
    out_q: "queue.Queue" = queue.Queue(maxsize=4)
    metrics = RuntimeMetrics()
    stop_event = threading.Event()
    stop_event.set()  # already signalled — loop should exit immediately

    worker_loop(
        WorkerConfig(mode=WorkerMode.IDENTITY),
        in_q, out_q, metrics, stop_event, SENTINEL,
        poll_timeout_seconds=0.01,
    )
    # No work done; no exception.
    assert metrics.fallback_count == 0


# ---------------------------------------------------------------------------
# RVC placeholders must fail loudly when called directly
# ---------------------------------------------------------------------------

def test_rvc_engine_infer_array_raises_not_implemented():
    engine = RvcEngine()
    block = np.zeros(1024, dtype=np.float32)
    with pytest.raises(NotImplementedError) as excinfo:
        engine.infer_array(block, sample_rate=48000)
    assert (
        "RVC inference is Stage 2 and is not implemented in this skeleton."
        in str(excinfo.value)
    )


def test_rvc_engine_load_raises_not_implemented():
    engine = RvcEngine()
    assert engine.is_loaded is False
    with pytest.raises(NotImplementedError):
        engine.load()


def test_worker_process_rvc_raises_not_implemented():
    block = np.zeros(480, dtype=np.float32)
    with pytest.raises(NotImplementedError):
        process_rvc(block)


# ---------------------------------------------------------------------------
# Chunker sanity
# ---------------------------------------------------------------------------

def test_chunker_emits_full_chunks_only():
    acc = BlockAccumulator(ChunkerConfig(chunk_size=1000))
    out = acc.feed(np.zeros(400, dtype=np.float32))
    assert out == []
    assert acc.pending_samples == 400
    out = acc.feed(np.ones(700, dtype=np.float32))
    assert len(out) == 1
    assert out[0].shape == (1000,)
    assert acc.pending_samples == 100


def test_chunker_rejects_stereo_block():
    acc = BlockAccumulator(ChunkerConfig(chunk_size=100))
    with pytest.raises(ValueError):
        acc.feed(np.zeros((50, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# Crossfade sanity
# ---------------------------------------------------------------------------

def test_linear_crossfade_endpoints_match_inputs():
    n = 64
    tail = np.ones(n, dtype=np.float32)
    head = np.full(n, 2.0, dtype=np.float32)
    fade = linear_crossfade(tail, head)
    assert fade[0] == pytest.approx(1.0)
    assert fade[-1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Import safety: src.main must NOT open audio devices on import
# ---------------------------------------------------------------------------

def test_importing_src_main_does_not_open_audio_devices(monkeypatch):
    """Trip-wire: importing src.main must not import sounddevice."""
    # Drop cached modules under src.* so the import is exercised fresh,
    # which lets the trip-wire actually see the import attempts.
    for name in list(sys.modules):
        if name == "src.main" or name.startswith("src.main."):
            del sys.modules[name]
    had_sounddevice_before = "sounddevice" in sys.modules

    def _forbidden_import(name, *args, **kwargs):
        if name == "sounddevice" or name.startswith("sounddevice."):
            raise AssertionError(
                "src.main triggered an import of sounddevice at import "
                "time; it must stay lazy."
            )
        return _orig_import(name, *args, **kwargs)

    import builtins
    _orig_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _forbidden_import)

    module = importlib.import_module("src.main")

    if not had_sounddevice_before:
        assert "sounddevice" not in sys.modules

    assert hasattr(module, "main")
