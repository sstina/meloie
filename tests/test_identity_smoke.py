"""Stage 1 identity smoke test + import-safety guard.

This test protects two invariants that the rest of the project relies on:

1. The Stage 1 identity worker is a true passthrough.
2. Importing the top-level ``src.main`` CLI does not open any audio
   devices (it must be safe to import in test runners, CI, GUIs, etc.).
"""

from __future__ import annotations

import importlib
import sys

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


# ---------------------------------------------------------------------------
# Identity worker
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
# RVC placeholders must fail loudly
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


def test_worker_loop_scaffold_raises_not_implemented():
    # Calling the loop must raise rather than silently start a thread.
    with pytest.raises(NotImplementedError):
        worker_loop(WorkerConfig(mode=WorkerMode.IDENTITY), None, None)


# ---------------------------------------------------------------------------
# Chunker sanity
# ---------------------------------------------------------------------------

def test_chunker_emits_full_chunks_only():
    acc = BlockAccumulator(ChunkerConfig(chunk_size=1000))
    out = acc.feed(np.zeros(400, dtype=np.float32))
    assert out == []          # not enough yet
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
    """If src.main even *imported* sounddevice on import we would notice
    a top-level ``sounddevice`` module after the import. Guard against
    that by ensuring no such module was added to ``sys.modules`` purely
    as a side effect of importing src.main."""
    sys.modules.pop("src.main", None)
    had_sounddevice_before = "sounddevice" in sys.modules

    # Trip-wire: if anything tries to import sounddevice during this
    # import, surface it loudly instead of silently opening a device.
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

    # If sounddevice was NOT already present before, it still shouldn't
    # be present as a side effect of importing src.main.
    if not had_sounddevice_before:
        assert "sounddevice" not in sys.modules

    # The module loaded; quick sanity check that the CLI is wired.
    assert hasattr(module, "main")
