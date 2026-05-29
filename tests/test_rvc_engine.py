"""Tests for the Stage 2 RVC engine adapter.

These tests never load a real RVC model, never import torch, and never
require ``infer_rvc_python`` to be installed. The default backend's
``load()`` is exercised only through the ``DependencyMissingError`` /
``ModelLoadError`` paths; an injectable fake backend covers the happy
path so the engine's plumbing is testable end-to-end.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.engine.rvc_engine import (
    DependencyMissingError,
    ModelLoadError,
    RvcBackend,
    RvcEngine,
    RvcEngineConfig,
    RvcInferenceError,
)


# ---------------------------------------------------------------------------
# RvcEngineConfig
# ---------------------------------------------------------------------------

def test_config_defaults_validate():
    cfg = RvcEngineConfig()
    cfg.validate()
    assert cfg.backend == "infer_rvc_python"
    assert cfg.f0_method == "rmvpe"
    assert cfg.index_rate == pytest.approx(0.5)
    assert cfg.protect == pytest.approx(0.33)
    assert cfg.filter_radius == 3
    assert cfg.rms_mix_rate == pytest.approx(1.0)
    assert cfg.pitch_shift == 0
    assert cfg.device == "auto"
    assert cfg.resample_sr == 0


def test_config_rejects_unknown_device():
    with pytest.raises(ValueError):
        RvcEngineConfig(device="rocm").validate()


def test_config_rejects_negative_resample_sr():
    with pytest.raises(ValueError):
        RvcEngineConfig(resample_sr=-1).validate()


def test_config_rejects_out_of_range_index_rate():
    with pytest.raises(ValueError):
        RvcEngineConfig(index_rate=2.0).validate()


def test_config_rejects_negative_filter_radius():
    with pytest.raises(ValueError):
        RvcEngineConfig(filter_radius=-1).validate()


def test_config_rejects_out_of_range_protect():
    with pytest.raises(ValueError):
        RvcEngineConfig(protect=0.9).validate()


# ---------------------------------------------------------------------------
# RvcEngine plumbing
# ---------------------------------------------------------------------------

def test_engine_infer_before_load_raises():
    engine = RvcEngine(RvcEngineConfig(model_path="/nope.pth"))
    with pytest.raises(RvcInferenceError):
        engine.infer_array(np.zeros(480, dtype=np.float32), sample_rate=48000)


def test_engine_default_backend_dependency_missing(monkeypatch):
    """If infer_rvc_python is not installed, load() must surface a clean
    DependencyMissingError — never silently fake success."""
    import importlib
    import sys

    # Pretend infer_rvc_python is uninstallable for the duration of this test.
    monkeypatch.setitem(sys.modules, "infer_rvc_python", None)
    cfg = RvcEngineConfig(model_path="/does/not/exist.pth")
    engine = RvcEngine(cfg)
    with pytest.raises(DependencyMissingError) as excinfo:
        engine.load()
    msg = str(excinfo.value)
    assert "infer_rvc_python" in msg
    assert "pip install" in msg


def test_engine_unknown_backend_raises_dependency_missing():
    cfg = RvcEngineConfig(model_path="/tmp/x.pth", backend="nonexistent_xyz")
    engine = RvcEngine(cfg)
    with pytest.raises(DependencyMissingError) as excinfo:
        engine.load()
    assert "nonexistent_xyz" in str(excinfo.value)


def test_engine_directml_experimental_not_implemented():
    """Reserved future backend must fail loudly rather than silently fall
    back to CPU."""
    from src.engine.rvc_engine import _InferRvcPythonBackend
    backend = _InferRvcPythonBackend()
    cfg = RvcEngineConfig(model_path="/fake.pth", device="directml_experimental")
    with pytest.raises(NotImplementedError):
        backend._resolve_only_cpu(cfg)


def test_engine_device_cpu_forces_cpu_without_touching_torch():
    """device='cpu' must not require torch — useful for headless CI."""
    from src.engine.rvc_engine import _InferRvcPythonBackend
    backend = _InferRvcPythonBackend()
    cfg = RvcEngineConfig(model_path="/fake.pth", device="cpu")
    assert backend._resolve_only_cpu(cfg) is True
    assert backend.resolved_device == "cpu"


def test_engine_force_cpu_back_compat_forces_cpu():
    from src.engine.rvc_engine import _InferRvcPythonBackend
    backend = _InferRvcPythonBackend()
    cfg = RvcEngineConfig(model_path="/fake.pth", device="auto", force_cpu=True)
    assert backend._resolve_only_cpu(cfg) is True
    assert backend.resolved_device == "cpu"


# ---------------------------------------------------------------------------
# Fake backend — exercises the happy path without any real RVC code
# ---------------------------------------------------------------------------

class _FakeBackend(RvcBackend):
    """In-memory fake backend. Multiplies the audio by 0.5 and rewraps
    the sample rate so we can verify the engine plumbing."""

    name = "fake"

    def __init__(self) -> None:
        self.load_calls = 0
        self.infer_calls = 0
        self.last_sr = None
        self.last_size = None
        self.raise_on_infer = False

    def load(self, config: RvcEngineConfig) -> None:
        self.load_calls += 1
        if not config.model_path:
            raise ModelLoadError("fake backend: model_path empty")

    def infer(self, audio, sample_rate):
        self.infer_calls += 1
        self.last_sr = int(sample_rate)
        self.last_size = int(audio.size)
        if self.raise_on_infer:
            raise RuntimeError("simulated backend failure")
        return audio.astype(np.float32, copy=False) * np.float32(0.5), int(sample_rate)


def test_engine_with_injected_backend_runs_inference():
    backend = _FakeBackend()
    cfg = RvcEngineConfig(model_path="/fake/m.pth")
    engine = RvcEngine(cfg, backend=backend)
    engine.load()
    assert backend.load_calls == 1
    assert engine.is_loaded

    audio = np.full(960, 0.4, dtype=np.float32)
    out, out_sr = engine.infer_array(audio, sample_rate=48000)
    assert out.shape == audio.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, audio * 0.5, rtol=1e-6)
    assert out_sr == 48000
    assert backend.infer_calls == 1


def test_engine_load_is_idempotent():
    backend = _FakeBackend()
    engine = RvcEngine(RvcEngineConfig(model_path="/fake/m.pth"), backend=backend)
    engine.load()
    engine.load()
    assert backend.load_calls == 1


def test_engine_rejects_non_array_input():
    engine = RvcEngine(RvcEngineConfig(model_path="/fake/m.pth"), backend=_FakeBackend())
    engine.load()
    with pytest.raises(TypeError):
        engine.infer_array([0.0, 0.1, 0.2], sample_rate=48000)  # type: ignore[arg-type]


def test_engine_rejects_stereo_input():
    engine = RvcEngine(RvcEngineConfig(model_path="/fake/m.pth"), backend=_FakeBackend())
    engine.load()
    with pytest.raises(ValueError):
        engine.infer_array(np.zeros((100, 2), dtype=np.float32), sample_rate=48000)


def test_engine_rejects_zero_sample_rate():
    engine = RvcEngine(RvcEngineConfig(model_path="/fake/m.pth"), backend=_FakeBackend())
    engine.load()
    with pytest.raises(ValueError):
        engine.infer_array(np.zeros(480, dtype=np.float32), sample_rate=0)


# ---------------------------------------------------------------------------
# Stage 2D: warmup() helper
# ---------------------------------------------------------------------------

def test_warmup_runs_count_inferences():
    backend = _FakeBackend()
    engine = RvcEngine(RvcEngineConfig(model_path="/fake/m.pth"), backend=backend)
    engine.load()
    timings = engine.warmup(chunk_samples=4800, sample_rate=48000, count=3)
    assert len(timings) == 3
    assert all(t >= 0.0 for t in timings)
    assert backend.infer_calls == 3
    # Each warmup call used the requested chunk size
    assert backend.last_size == 4800


def test_warmup_zero_count_returns_empty_list():
    backend = _FakeBackend()
    engine = RvcEngine(RvcEngineConfig(model_path="/fake/m.pth"), backend=backend)
    engine.load()
    timings = engine.warmup(chunk_samples=4800, sample_rate=48000, count=0)
    assert timings == []
    assert backend.infer_calls == 0


def test_warmup_requires_loaded_engine():
    engine = RvcEngine(RvcEngineConfig(model_path="/fake/m.pth"))
    with pytest.raises(RvcInferenceError):
        engine.warmup(chunk_samples=480, sample_rate=48000, count=1)


def test_warmup_rejects_bad_args():
    backend = _FakeBackend()
    engine = RvcEngine(RvcEngineConfig(model_path="/fake/m.pth"), backend=backend)
    engine.load()
    with pytest.raises(ValueError):
        engine.warmup(chunk_samples=0, sample_rate=48000, count=1)
    with pytest.raises(ValueError):
        engine.warmup(chunk_samples=480, sample_rate=0, count=1)


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------

def test_importing_rvc_engine_does_not_import_torch(monkeypatch):
    """Importing src.engine.rvc_engine must not import torch or
    infer_rvc_python at module load."""
    import builtins
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "src.engine.rvc_engine":
            del sys.modules[name]

    orig_import = builtins.__import__

    def _forbidden(name, *args, **kwargs):
        if name in ("torch", "infer_rvc_python") or name.startswith(
            ("torch.", "infer_rvc_python.")
        ):
            raise AssertionError(
                f"src.engine.rvc_engine triggered a top-level import of "
                f"{name!r}; that import must stay lazy."
            )
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _forbidden)
    module = importlib.import_module("src.engine.rvc_engine")
    assert hasattr(module, "RvcEngine")
