"""Tests for Backend's game mode (off / dgpu_light / cpu_zero).

Game mode trades precision/latency for low or zero dGPU usage while gaming. The
DEVICE/block overrides ride the load bundle (ca.build_configs_for_model(game_mode=));
the "sacrifice precision" LIVE levers (index_rate -> 0, silence gate on) go through
the normal setters so _desired stays the source of truth and "off" restores them.

Drives the real Backend (a QObject) under a QCoreApplication with a FAKE
RealtimeSession (no torch / audio / QML), mirroring test_backend_desired.py. The
flaky QThread reload worker is bypassed: _beginLoad is monkeypatched to a recorder
(its real config-build wiring is covered separately in test_config_assembly.py and
in test_setgamemode_threads_mode_into_config_build below).
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication

from meloie.control import RealtimeSession
from meloie.ui import backend as backend_mod
from meloie.ui.backend import Backend


# --------------------------------------------------------------------------- fakes
class FakeEngine:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.calls = []

    def load(self):
        pass

    def __getattr__(self, name):
        if name.startswith("set_"):
            def _rec(*a, **k):
                self.calls.append((name, a, k))
            return _rec
        raise AttributeError(name)


def _looping_runner(audio_cfg, engine, *, stop_event, metrics, print_metrics, **kw):
    while not stop_event.wait(0.005):
        metrics.input_frames += 1
    return metrics


@pytest.fixture(scope="module")
def qapp():
    return QCoreApplication.instance() or QCoreApplication([])


@pytest.fixture
def make(qapp, monkeypatch):
    built = []

    def _make():
        monkeypatch.setattr(backend_mod.ca, "list_device_dicts", lambda: [])
        monkeypatch.setattr(backend_mod.ca, "list_model_files", lambda: [])
        created = []

        def factory(cfg):
            e = FakeEngine(cfg)
            created.append(e)
            return e

        session = RealtimeSession(engine_factory=factory, stream_runner=_looping_runner)
        b = Backend(session=session)
        built.append((b, session))
        return b, session, created

    yield _make

    for b, session in built:
        try:
            session.stop()
        except Exception:
            pass
        try:
            b._metricsTimer.stop()
        except Exception:
            pass


def _running(b, session, created):
    """Bring the fake session to RUNNING with one engine built."""
    session.load(object())
    session.start(object())
    return created[0]


# --------------------------------------------------------------------------- tests
def test_gamemode_defaults_off(make):
    b, session, created = make()
    assert b._gameMode == "off"
    assert b.gameMode == "off"


def test_setgamemode_unknown_is_rejected(make):
    b, session, created = make()
    errors = []
    b.errorOccurred.connect(errors.append)
    b.setGameMode("turbo")
    assert b._gameMode == "off"
    assert errors and "未知" in errors[0]


def test_setgamemode_active_sets_levers_and_reloads_when_running(make, monkeypatch):
    b, session, created = make()
    b._desired["index_rate"] = 0.5            # a real index in use -> game mode forces it to 0
    eng = _running(b, session, created)
    b._lastLoadArgs = ("m.pth", "mic", "out", "fcpe", "")
    calls = []
    monkeypatch.setattr(b, "_beginLoad", lambda *a, **k: calls.append((a, k)))

    modes = []
    b.gameModeChanged.connect(lambda: modes.append(b.gameMode))
    b.setGameMode("cpu_zero")

    assert b._gameMode == "cpu_zero" and b.gameMode == "cpu_zero"
    assert modes == ["cpu_zero"]              # signalled once
    # precision levers pushed live onto the running engine
    assert ("set_index_rate", (0.0,), {}) in eng.calls
    assert any(c[0] == "set_silence_gate" and c[1][0] is not None for c in eng.calls)
    # reloaded in place onto the new device bundle (is_reload=True, same routing)
    assert calls and calls[-1][1].get("is_reload") is True
    assert calls[-1][0][0] == "m.pth"


def test_setgamemode_skips_index_when_no_index(make, monkeypatch):
    # index_rate already 0 (no index) -> don't poke set_index_rate (avoids a raise),
    # but still arm the silence gate.
    b, session, created = make()
    b._desired["index_rate"] = 0.0
    eng = _running(b, session, created)
    b._lastLoadArgs = ("m.pth", "mic", "out", "fcpe", "")
    monkeypatch.setattr(b, "_beginLoad", lambda *a, **k: None)

    b.setGameMode("dgpu_light")
    assert not any(c[0] == "set_index_rate" for c in eng.calls)
    assert any(c[0] == "set_silence_gate" and c[1][0] is not None for c in eng.calls)


def test_setgamemode_off_restores_prior_knobs(make, monkeypatch):
    b, session, created = make()
    b._desired["index_rate"] = 0.5            # user's pre-game precision (no silence set)
    eng = _running(b, session, created)
    b._lastLoadArgs = ("m.pth", "mic", "out", "fcpe", "")
    monkeypatch.setattr(b, "_beginLoad", lambda *a, **k: None)

    b.setGameMode("cpu_zero")                 # snapshot {index_rate:0.5, silence:None}
    eng.calls.clear()
    b.setGameMode("off")                      # restore

    assert b._gameMode == "off"
    assert ("set_index_rate", (0.5,), {}) in eng.calls           # restored exactly
    assert any(c[0] == "set_silence_gate" and c[1][0] is None for c in eng.calls)  # gate back off


def test_setgamemode_off_restores_prior_silence_on(make, monkeypatch):
    # if the user had the silence gate ON before game mode, "off" restores THAT (not off).
    b, session, created = make()
    eng = _running(b, session, created)
    b.setSilenceGate(True, -55.0)            # user's pre-game gate
    b._lastLoadArgs = ("m.pth", "mic", "out", "fcpe", "")
    monkeypatch.setattr(b, "_beginLoad", lambda *a, **k: None)

    b.setGameMode("cpu_zero")
    eng.calls.clear()
    b.setGameMode("off")
    # restored to the user's -55 (set_silence_gate gets the raw dbfs, not None)
    assert any(c[0] == "set_silence_gate" and c[1][0] == -55.0 for c in eng.calls)


def test_setgamemode_idle_only_stores_mode(make, monkeypatch):
    b, session, created = make()
    calls = []
    monkeypatch.setattr(b, "_beginLoad", lambda *a, **k: calls.append(1))
    b.setGameMode("cpu_zero")
    assert b._gameMode == "cpu_zero"
    assert calls == []                        # nothing loaded -> no reload
    assert created == []                       # no engine built (live setters are no-ops)


def test_setgamemode_loaded_drops_fastpath_key(make, monkeypatch):
    # LOADED-but-stopped: don't auto-start; drop the fast-path key so the next Start
    # does a full reload onto the new device.
    b, session, created = make()
    session.load(object())                    # LOADED, not running
    b._loaded_key = "m.pth"
    b._lastLoadArgs = ("m.pth", "mic", "out", "fcpe", "")
    calls = []
    monkeypatch.setattr(b, "_beginLoad", lambda *a, **k: calls.append(1))

    b.setGameMode("cpu_zero")
    assert b._loaded_key is None              # fast path invalidated
    assert calls == []                         # not reloaded now (deferred to next Start)


def test_setgamemode_threads_mode_into_config_build(make, monkeypatch):
    # Pins the real _beginLoad wiring: build_configs_for_model is called with
    # game_mode=self._gameMode. We make the build raise so _beginLoad aborts BEFORE
    # spawning its QThread (deterministic, no background thread to join).
    b, session, created = make()
    _running(b, session, created)
    b._lastLoadArgs = ("m.pth", "mic", "out", "fcpe", "")
    seen = {}

    def rec(model, ins, outs, f0="fcpe", monitor_substr=None, game_mode="off"):
        seen["game_mode"] = game_mode
        raise RuntimeError("abort before thread")

    monkeypatch.setattr(backend_mod.ca, "build_configs_for_model", rec)
    errors = []
    b.errorOccurred.connect(errors.append)

    b.setGameMode("cpu_zero")

    assert seen.get("game_mode") == "cpu_zero"   # the wiring under test
    assert b._busy is False                       # aborted build -> never stuck busy
    assert any("config error" in e for e in errors)
