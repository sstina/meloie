"""Pure tests for RealtimeSession (Phase 0 control facade).

Uses an injected fake engine + fake stream runner, so NO audio devices / torch /
sounddevice are touched. Threading is exercised with Event-gated fakes and
bounded polling (no fixed sleeps that could flake).
"""

from __future__ import annotations

import threading
import time

import pytest

from src.control import RealtimeSession, SessionError, SessionState


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeEngine:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.loaded = False
        self.calls = []

    def load(self):
        self.loaded = True

    def __getattr__(self, name):
        # record any set_* delegation generically
        if name.startswith("set_"):
            def _rec(*a, **k):
                self.calls.append((name, a, k))
            return _rec
        raise AttributeError(name)


def make_factory():
    created = []

    def factory(cfg):
        e = FakeEngine(cfg)
        created.append(e)
        return e

    return factory, created


def looping_runner(audio_cfg, engine, *, stop_event, metrics, print_metrics, **kw):
    """Runs until stop_event is set, bumping a metric so the snapshot grows."""
    assert print_metrics is False           # session must silence console output
    while not stop_event.wait(0.005):
        metrics.input_frames += 1
    return metrics


def immediate_runner(audio_cfg, engine, *, stop_event, metrics, print_metrics, **kw):
    return metrics                          # simulates duration elapsed


def raising_runner(audio_cfg, engine, *, stop_event, metrics, print_metrics, **kw):
    raise RuntimeError("boom")


def _wait(pred, timeout=2.0, interval=0.005):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(interval)
    return pred()


CFG = object()
AUDIO = object()


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------

def test_load_transitions_to_loaded():
    factory, created = make_factory()
    s = RealtimeSession(engine_factory=factory, stream_runner=immediate_runner)
    assert s.state is SessionState.IDLE
    s.load(CFG)
    assert s.state is SessionState.LOADED
    assert len(created) == 1 and created[0].loaded is True


def test_load_failure_sets_error_and_reraises():
    def bad_factory(cfg):
        raise ValueError("nope")

    s = RealtimeSession(engine_factory=bad_factory, stream_runner=immediate_runner)
    with pytest.raises(ValueError):
        s.load(CFG)
    assert s.state is SessionState.ERROR
    assert "nope" in (s.last_error or "")


def test_start_is_nonblocking_then_stop():
    factory, _ = make_factory()
    s = RealtimeSession(engine_factory=factory, stream_runner=looping_runner)
    s.load(CFG)
    s.start(AUDIO)                       # must return immediately
    assert s.state is SessionState.RUNNING
    assert _wait(lambda: s.metrics_snapshot().get("input_frames", 0) > 0)
    s.stop()
    assert s.state is SessionState.LOADED


def test_runner_return_transitions_to_loaded():
    factory, _ = make_factory()
    s = RealtimeSession(engine_factory=factory, stream_runner=immediate_runner)
    s.load(CFG)
    s.start(AUDIO)
    assert _wait(lambda: s.state is SessionState.LOADED)


def test_runner_raise_sets_error():
    factory, _ = make_factory()
    s = RealtimeSession(engine_factory=factory, stream_runner=raising_runner)
    s.load(CFG)
    s.start(AUDIO)
    assert _wait(lambda: s.state is SessionState.ERROR)
    assert "boom" in (s.last_error or "")


def test_reload_stops_then_loads_new_engine():
    factory, created = make_factory()
    s = RealtimeSession(engine_factory=factory, stream_runner=looping_runner)
    s.load(CFG)
    s.start(AUDIO)
    assert _wait(lambda: s.state is SessionState.RUNNING)
    s.reload(CFG)
    assert s.state is SessionState.LOADED
    assert len(created) == 2             # a fresh engine was built


# --------------------------------------------------------------------------
# live setters
# --------------------------------------------------------------------------

def test_set_delegates_to_engine_while_running():
    factory, created = make_factory()
    s = RealtimeSession(engine_factory=factory, stream_runner=looping_runner)
    s.load(CFG)
    s.start(AUDIO)
    s.set_pitch_shift(5)
    s.set_formant(True, timbre=0.25)
    s.set_auto_center(True)
    s.stop()
    names = [c[0] for c in created[0].calls]
    assert ("set_pitch_shift", (5,), {}) in created[0].calls
    assert "set_formant" in names
    assert ("set_auto_center", (True, None, None), {}) in created[0].calls


def test_set_before_load_raises():
    s = RealtimeSession(engine_factory=make_factory()[0], stream_runner=immediate_runner)
    with pytest.raises(SessionError):
        s.set_pitch_shift(3)


def test_set_f0_method_delegates_to_engine_while_running():
    factory, created = make_factory()
    s = RealtimeSession(engine_factory=factory, stream_runner=looping_runner)
    s.load(CFG)
    s.start(AUDIO)
    s.set_f0_method("rmvpe")
    s.stop()
    assert ("set_f0_method", ("rmvpe",), {}) in created[0].calls


def test_set_f0_method_before_load_raises():
    s = RealtimeSession(engine_factory=make_factory()[0], stream_runner=immediate_runner)
    with pytest.raises(SessionError):
        s.set_f0_method("rmvpe")


def test_metrics_snapshot_empty_before_start():
    s = RealtimeSession(engine_factory=make_factory()[0], stream_runner=immediate_runner)
    assert s.metrics_snapshot() == {}
    s.load(CFG)
    assert s.metrics_snapshot() == {}     # still no run yet


# --------------------------------------------------------------------------
# monitor (headphone sink — routing, gated by a shared MonitorState)
# --------------------------------------------------------------------------

def test_set_monitor_enabled_flips_shared_state():
    captured = {}

    def capturing_runner(audio_cfg, engine, *, stop_event, metrics, print_metrics, **kw):
        captured["monitor_state"] = kw.get("monitor_state")
        while not stop_event.wait(0.005):
            metrics.input_frames += 1
        return metrics

    factory, _ = make_factory()
    s = RealtimeSession(engine_factory=factory, stream_runner=capturing_runner)
    s.load(CFG)
    s.start(AUDIO, monitor_enabled=True)
    assert _wait(lambda: captured.get("monitor_state") is not None)
    ms = captured["monitor_state"]
    assert ms.enabled is True and s.monitor_enabled is True
    s.set_monitor_enabled(False)            # the SAME object the runner reads flips live
    assert ms.enabled is False and s.monitor_enabled is False
    s.set_monitor_enabled(True)
    assert ms.enabled is True
    s.stop()


def test_set_monitor_enabled_noop_before_start():
    s = RealtimeSession(engine_factory=make_factory()[0], stream_runner=immediate_runner)
    s.set_monitor_enabled(True)             # no stream yet -> no-op, must not raise
    assert s.monitor_enabled is False


# --------------------------------------------------------------------------
# state-change callback + contract
# --------------------------------------------------------------------------

def test_on_state_change_fires_expected_sequence():
    events = []
    factory, _ = make_factory()
    s = RealtimeSession(
        engine_factory=factory, stream_runner=looping_runner,
        on_state_change=lambda o, n: events.append((o, n)),
    )
    s.load(CFG)
    s.start(AUDIO)
    assert _wait(lambda: s.state is SessionState.RUNNING)
    s.stop()
    assert events == [
        (SessionState.IDLE, SessionState.LOADING),
        (SessionState.LOADING, SessionState.LOADED),
        (SessionState.LOADED, SessionState.RUNNING),
        (SessionState.RUNNING, SessionState.STOPPING),
        (SessionState.STOPPING, SessionState.LOADED),
    ]


def test_session_has_no_output_shaping_setters():
    s = RealtimeSession(engine_factory=make_factory()[0], stream_runner=immediate_runner)
    for bad in ("set_rms_mix_rate", "set_output_denoise", "set_gain", "set_volume_envelope"):
        assert not hasattr(s, bad)
    forbidden = ("rms", "gain", "volume", "output", "eq", "limiter", "reverb", "compress")
    for name in dir(s):
        if name.startswith("set_"):
            assert not any(tok in name.lower() for tok in forbidden), name
