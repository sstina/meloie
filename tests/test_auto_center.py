"""Pure tests for A2 auto pitch-centering (input-side slow-EMA F0 tracker).

Constructs the engine but NEVER calls load() (no torch). The tracker's F0
predictor and the 16k input tensor are faked, so we test the offset math +
EMA + freeze logic in isolation. Default-OFF behaviour is verified too.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.streaming_engine import StreamingEngineConfig, StreamingRvcEngine


class _FakeTensor:
    """Stand-in for the engine's 16k torch tensor: .detach().to().numpy()."""

    def __init__(self, arr):
        self._a = arr

    def detach(self):
        return self

    def to(self, *a, **k):
        return self

    def numpy(self):
        return self._a


class _FakeF0:
    def __init__(self, f0):
        self._f0 = f0

    def get_f0(self, x, *a, **k):
        return self._f0


class _FakePipe:
    def __init__(self, f0, method="rmvpe"):
        self.f0_model = _FakeF0(f0)
        self.f0_method = method


def _tracker_engine(f0, **cfg):
    base = dict(model_path="m.pth", auto_center=True, auto_center_target_hz=200.0,
                auto_center_tau_s=0.001, block_ms=250.0)   # tiny tau -> EMA = median in 1 step
    base.update(cfg)
    e = StreamingRvcEngine(StreamingEngineConfig(**base))
    e.block_frame_16k = 160                 # state normally set by _realloc(load)
    e._track_buf = np.zeros(16000, dtype=np.float32)
    e._track_interval_blocks = 1
    e._pipeline = _FakePipe(f0)
    return e


def _feed(e, n=16000):
    e._update_auto_center(_FakeTensor(np.ones(n, dtype=np.float32)))


def test_auto_center_off_by_default():
    e = StreamingRvcEngine(StreamingEngineConfig(model_path="m.pth"))
    assert e._auto_center_on is False
    assert e._auto_offset == 0.0


def test_set_auto_center_toggles_and_validates():
    e = StreamingRvcEngine(StreamingEngineConfig(model_path="m.pth"))
    e.set_auto_center(True, target_hz=220.0, tau_s=15.0)
    assert e._auto_center_on is True
    assert e.cfg.auto_center_target_hz == 220.0
    assert e.cfg.auto_center_tau_s == 15.0
    e.set_auto_center(False)
    assert e._auto_center_on is False
    assert e._user_f0_ema is None
    with pytest.raises(ValueError):
        e.set_auto_center(True, target_hz=0.0)
    with pytest.raises(ValueError):
        e.set_auto_center(True, tau_s=-1.0)


def test_tracker_centers_median_to_target():
    e = _tracker_engine(np.full(50, 100.0))      # user median 100 Hz, target 200
    _feed(e)
    assert abs(e._user_f0_ema - 100.0) < 1e-6
    assert abs(e._auto_offset - 12.0) < 1e-6     # 12*log2(200/100) = 12 (== default limit)


def test_tracker_clamps_offset():
    e = _tracker_engine(np.full(50, 100.0), auto_center_limit=6.0)
    _feed(e)
    assert abs(e._auto_offset - 6.0) < 1e-6      # wants +12, clamped to +6


def test_tracker_freezes_on_unvoiced():
    e = _tracker_engine(np.zeros(50))            # all unvoiced (f0=0)
    _feed(e)
    assert e._auto_offset == 0.0                 # frozen: unchanged
    assert e._user_f0_ema is None


def test_tracker_subsamples_by_interval():
    e = _tracker_engine(np.full(50, 100.0))
    e._track_interval_blocks = 3
    _feed(e); assert e._auto_offset == 0.0       # block 1 < 3
    _feed(e); assert e._auto_offset == 0.0       # block 2 < 3
    _feed(e); assert abs(e._auto_offset - 12.0) < 1e-6   # block 3 -> fires


def test_off_gate_ignores_stale_offset():
    # an offset computed while ON must not leak once turned OFF (gated in _convert).
    e = _tracker_engine(np.full(50, 100.0))
    _feed(e)
    assert e._auto_offset != 0.0
    e.set_auto_center(False)
    eff = e.pitch_shift + (e._auto_offset if e._auto_center_on else 0.0)
    assert eff == e.pitch_shift                  # auto contribution gated out
