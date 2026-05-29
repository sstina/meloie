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

from src.audio.chunker import (
    BlockAccumulator,
    ChunkerConfig,
    linear_resample,
    reconcile_to_length,
    resample_audio,
)
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
# RvcEngine is now Stage 2 (real adapter). The fallback-test path in the
# identity worker still uses ``process_rvc`` -> NotImplementedError so the
# identity-loop fallback semantics remain meaningful. The real engine
# behaviour is covered by tests/test_rvc_engine.py + tests/test_rvc_worker.py.
# ---------------------------------------------------------------------------

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
# linear_resample (Stage 2D)
# ---------------------------------------------------------------------------

def test_linear_resample_same_rate_is_noop_copy():
    audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out = linear_resample(audio, 48000, 48000)
    np.testing.assert_array_equal(out, audio)
    # must be a copy
    out[0] = 99.0
    assert audio[0] == pytest.approx(0.1)


def test_linear_resample_40k_to_48k_changes_length():
    # 40 kHz -> 48 kHz: 1.0 second of audio
    audio = np.sin(2 * np.pi * 440.0 * np.arange(40000) / 40000).astype(np.float32)
    out = linear_resample(audio, 40000, 48000)
    assert out.size == 48000
    assert out.dtype == np.float32
    # peak should still be ~1.0 (we don't lose energy at the peaks)
    assert float(np.max(np.abs(out))) == pytest.approx(1.0, abs=2e-2)


def test_linear_resample_48k_to_40k_changes_length_down():
    audio = np.ones(48000, dtype=np.float32) * 0.5
    out = linear_resample(audio, 48000, 40000)
    assert out.size == 40000
    np.testing.assert_allclose(out, 0.5 * np.ones(40000, dtype=np.float32), atol=1e-6)


def test_linear_resample_rejects_zero_or_negative_sr():
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError):
        linear_resample(audio, 0, 48000)
    with pytest.raises(ValueError):
        linear_resample(audio, 48000, -1)


def test_linear_resample_rejects_stereo():
    with pytest.raises(ValueError):
        linear_resample(np.zeros((100, 2), dtype=np.float32), 44100, 48000)


# ---------------------------------------------------------------------------
# resample_audio (preferred resampler used by the realtime worker)
# ---------------------------------------------------------------------------

def test_resample_audio_same_rate_is_noop_copy():
    audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out = resample_audio(audio, 48000, 48000)
    np.testing.assert_array_equal(out, audio)
    # Must be a copy.
    out[0] = 99.0
    assert audio[0] == pytest.approx(0.1)


def test_resample_audio_40k_to_48k_changes_length_and_is_finite():
    audio = np.sin(
        2.0 * np.pi * 440.0 * np.arange(40000) / 40000.0
    ).astype(np.float32)
    out = resample_audio(audio, 40000, 48000)
    assert out.size == 48000
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    # Peak stays close to the input peak (sinc resamplers preserve
    # peaks within a few % overshoot at most).
    assert 0.85 <= float(np.max(np.abs(out))) <= 1.15


def test_resample_audio_48k_to_40k_changes_length_down():
    audio = np.full(48000, 0.5, dtype=np.float32)
    out = resample_audio(audio, 48000, 40000)
    assert out.size == 40000
    # The center is stable; edge taps may carry a small sinc-window
    # transient, so only check the interior.
    interior = out[200:-200]
    np.testing.assert_allclose(
        interior, np.full_like(interior, 0.5), atol=5e-3
    )


def test_resample_audio_rejects_zero_or_negative_sr():
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError):
        resample_audio(audio, 0, 48000)
    with pytest.raises(ValueError):
        resample_audio(audio, 48000, -1)


def test_resample_audio_rejects_stereo():
    with pytest.raises(ValueError):
        resample_audio(np.zeros((100, 2), dtype=np.float32), 44100, 48000)


# ---------------------------------------------------------------------------
# Stage 4-E: reconcile_to_length (timeline reconciliation)
# ---------------------------------------------------------------------------

def test_reconcile_to_length_noop_when_size_matches():
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = reconcile_to_length(audio, 3, method="polyphase")
    np.testing.assert_array_equal(out, audio)
    # Must be a copy.
    out[0] = 99.0
    assert audio[0] == pytest.approx(0.1)


def test_reconcile_to_length_off_returns_input_unchanged_size():
    audio = np.ones(1234, dtype=np.float32)
    out = reconcile_to_length(audio, 9999, method="off")
    assert out.size == 1234  # unchanged
    assert out is audio  # exact passthrough is allowed for "off"


def test_reconcile_to_length_polyphase_stretches_to_target():
    # Simulate the kiki case: 47040 samples -> 48000 samples (50:49)
    audio = np.sin(
        2.0 * np.pi * 440.0 * np.arange(47040) / 48000.0
    ).astype(np.float32)
    out = reconcile_to_length(audio, 48000, method="polyphase")
    assert out.size == 48000
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    # Peak preserved within sinc-window overshoot tolerance.
    assert 0.85 <= float(np.max(np.abs(out))) <= 1.15


def test_reconcile_to_length_polyphase_compresses_to_target():
    # Excess case: 48800 samples -> 48000 samples.
    audio = np.full(48800, 0.5, dtype=np.float32)
    out = reconcile_to_length(audio, 48000, method="polyphase")
    assert out.size == 48000
    # Interior is stable.
    interior = out[200:-200]
    np.testing.assert_allclose(interior, np.full_like(interior, 0.5), atol=5e-3)


def test_reconcile_to_length_pad_zero_pads_short():
    audio = np.full(100, 0.5, dtype=np.float32)
    out = reconcile_to_length(audio, 200, method="pad_zero")
    assert out.size == 200
    np.testing.assert_array_equal(out[:100], audio)
    np.testing.assert_array_equal(out[100:], np.zeros(100, dtype=np.float32))


def test_reconcile_to_length_pad_zero_truncates_long():
    audio = np.full(200, 0.5, dtype=np.float32)
    out = reconcile_to_length(audio, 100, method="pad_zero")
    assert out.size == 100
    np.testing.assert_array_equal(out, np.full(100, 0.5, dtype=np.float32))


def test_reconcile_to_length_linear_stretches_to_target():
    audio = np.full(40000, 0.5, dtype=np.float32)
    out = reconcile_to_length(audio, 48000, method="linear")
    assert out.size == 48000
    np.testing.assert_allclose(out, np.full(48000, 0.5, dtype=np.float32), atol=1e-6)


def test_reconcile_to_length_rejects_stereo():
    with pytest.raises(ValueError):
        reconcile_to_length(np.zeros((100, 2), dtype=np.float32), 200)


def test_reconcile_to_length_rejects_negative_target():
    with pytest.raises(ValueError):
        reconcile_to_length(np.zeros(100, dtype=np.float32), -1)


def test_reconcile_to_length_rejects_unknown_method():
    with pytest.raises(ValueError):
        reconcile_to_length(np.zeros(100, dtype=np.float32), 200, method="bogus")


def test_reconcile_to_length_zero_target_returns_empty():
    out = reconcile_to_length(np.ones(100, dtype=np.float32), 0, method="polyphase")
    assert out.size == 0


def test_reconcile_to_length_empty_input_returns_zeros_at_target():
    out = reconcile_to_length(np.zeros(0, dtype=np.float32), 100, method="polyphase")
    assert out.size == 100
    np.testing.assert_array_equal(out, np.zeros(100, dtype=np.float32))


def test_reconcile_to_length_polyphase_falls_back_when_scipy_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.signal" or name.startswith("scipy.signal."):
            raise ImportError("simulated missing scipy.signal")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    audio = np.full(40000, 0.5, dtype=np.float32)
    out = reconcile_to_length(audio, 48000, method="polyphase")
    assert out.size == 48000
    np.testing.assert_allclose(out, np.full(48000, 0.5, dtype=np.float32), atol=1e-6)


def test_resample_audio_falls_back_when_scipy_missing(monkeypatch):
    """If scipy is unavailable, resample_audio must still work via
    np.interp. Simulate by hiding scipy.signal from the import system."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.signal" or name.startswith("scipy.signal."):
            raise ImportError("simulated missing scipy.signal")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    audio = np.full(40000, 0.5, dtype=np.float32)
    out = resample_audio(audio, 40000, 48000)
    assert out.size == 48000
    np.testing.assert_allclose(
        out, np.full(48000, 0.5, dtype=np.float32), atol=1e-6
    )


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
