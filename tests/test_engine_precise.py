"""Engine-wiring tests for precise F0 mapping (no torch model, fake pipeline).

Builds a real StreamingRvcEngine (its __init__ is torch-free) but swaps in a fake
pipeline + stub buffers, so we can assert that set_precise_mapping attaches/detaches
``pipeline.f0_remap`` and that _convert applies REPLACE semantics (f0_up_key=0,
autotune/proposed off) when precise mapping is on.
"""

from __future__ import annotations

import numpy as np
import pytest

from meloie.engine.streaming_engine import (
    StreamingEngineConfig,
    StreamingEngineError,
    StreamingRvcEngine,
)


class FakePipeline:
    def __init__(self):
        self.f0_remap = None
        self.f0_model = None
        self.f0_method = "rmvpe"
        self.last_kwargs = None

    def voice_conversion(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return "ok"


def _engine(**cfg):
    e = StreamingRvcEngine(StreamingEngineConfig(model_path="x.pth", **cfg))
    e._loaded = True
    e._pipeline = FakePipeline()
    # stub the buffers/sizes _convert reads (passed through verbatim to the fake)
    e._convert_buffer = object()
    e._pitch_buffer = object()
    e._pitchf_buffer = object()
    e._convert_feature_size_16k = 10
    e._skip_head = 1
    e._return_length = 5
    e.block_frame_16k = 3
    return e


_SRC_Q = np.linspace(6.0, 8.0, 48)
_TGT_Q = np.linspace(7.0, 9.0, 48)


def test_set_precise_mapping_attaches_then_drops_remap():
    e = _engine()
    e.set_precise_mapping(True, _SRC_Q, _TGT_Q, "rmvpe")
    assert e._precise_on is True
    assert callable(e._pipeline.f0_remap)
    # closure maps voiced frames, leaves unvoiced 0, preserves dtype
    out = e._pipeline.f0_remap(np.array([0.0, 2.0 ** 6.0, 2.0 ** 8.0], dtype=np.float32))
    assert out[0] == 0.0 and out.dtype == np.float32
    e.set_precise_mapping(False)
    assert e._precise_on is False and e._pipeline.f0_remap is None


def test_convert_replaces_all_f0_knobs_when_precise_on():
    e = _engine(pitch_shift=7, f0_autotune=True, proposed_pitch=True)
    e.set_precise_mapping(True, _SRC_Q, _TGT_Q, "rmvpe")
    e._convert()
    kw = e._pipeline.last_kwargs
    assert kw["f0_up_key"] == 0.0          # CDF map is the sole F0 transform
    assert kw["f0_autotune"] is False
    assert kw["proposed_pitch"] is False


def test_convert_uses_manual_pitch_when_precise_off():
    e = _engine(pitch_shift=7)
    e._convert()
    assert e._pipeline.last_kwargs["f0_up_key"] == 7


def test_set_precise_mapping_requires_loaded_engine():
    e = StreamingRvcEngine(StreamingEngineConfig(model_path="x.pth"))   # not loaded
    with pytest.raises(StreamingEngineError):
        e.set_precise_mapping(True, _SRC_Q, _TGT_Q)


def test_set_precise_on_without_quantiles_raises():
    e = _engine()
    with pytest.raises(StreamingEngineError):
        e.set_precise_mapping(True)        # no anchors built/passed
