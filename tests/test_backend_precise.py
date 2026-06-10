"""Backend wiring tests for precise F0 mapping (QCoreApplication + fake session).

Drives the real Backend (QObject) with a fake RealtimeSession (fake engine that
records set_* and provides build_precise_map). No torch, no audio, no QML, no file
dialog. Covers: enable gating, the OFF path, _onPreciseDone caching + apply, the
cheap reload-replay (no rebuild on the GUI thread), a model swap resetting state,
and the off-thread PreciseMapWorker build path pumped via the event loop.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from meloie.control import RealtimeSession
from meloie.ui import backend as backend_mod
from meloie.ui import precise_store as ps
from meloie.ui.backend import Backend


class FakeEngine:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.calls = []
        self.build_count = 0

    def load(self):
        pass

    def build_precise_map(self, voice, target):
        self.build_count += 1
        return np.linspace(6.0, 8.0, 48), np.linspace(7.0, 9.0, 48), "rmvpe"

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
def make(qapp, monkeypatch, tmp_path_factory):
    built = []

    def _make():
        monkeypatch.setattr(backend_mod.ca, "list_device_dicts", lambda: [])
        monkeypatch.setattr(backend_mod.ca, "list_model_files", lambda: [])
        # hermetic saved-maps dir (don't touch the real config/precise_maps)
        monkeypatch.setattr(backend_mod.ps, "PRECISE_DIR",
                            str(tmp_path_factory.mktemp("pmaps")))
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


def _precise_calls(engine):
    return [c for c in engine.calls if c[0] == "set_precise_mapping"]


def _pump(qapp, pred, timeout=3.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if pred():
            return True
        time.sleep(0.005)
    return pred()


# --------------------------------------------------------------------------- gating
def test_enable_refused_without_wavs(make):
    b, session, created = make()
    errors = []
    b.errorOccurred.connect(errors.append)
    session.load(object())
    b.setPreciseMapping(True)                    # no wavs selected
    assert b.preciseMappingOn is False
    assert errors                                # a toast told the user what's missing
    assert b._precise_thread is None             # no worker spawned


def test_enable_refused_when_not_loaded(make):
    b, session, created = make()
    errors = []
    b.errorOccurred.connect(errors.append)
    b._preciseVoiceWav = "you.wav"
    b._preciseTargetWav = "model.wav"
    b.setPreciseMapping(True)                     # engine not loaded (IDLE)
    assert b.preciseMappingOn is False and errors
    assert b._precise_thread is None


# --------------------------------------------------------------------------- off path
def test_disable_drops_map_live(make):
    b, session, created = make()
    session.load(object())
    session.start(object())
    b._preciseMappingOn = True
    b.setPreciseMapping(False)
    pc = _precise_calls(created[0])
    assert pc and pc[-1][1][0] is False          # set_precise_mapping(False, ...)
    assert b.preciseMappingOn is False
    assert b._desired["precise"] == (False, None, None, None)


# --------------------------------------------------------------- _onPreciseDone direct
def test_on_precise_done_success_caches_and_applies(make):
    b, session, created = make()
    session.load(object())
    session.start(object())
    src_q = np.linspace(6.0, 8.0, 48)
    tgt_q = np.linspace(7.0, 9.0, 48)
    b._set_busy(True)                             # mimic in-flight build
    b._onPreciseDone(True, src_q, tgt_q, "rmvpe", "")
    assert b.preciseMappingOn is True
    assert b._desired["precise"][0] is True and b._desired["precise"][3] == "rmvpe"
    pc = _precise_calls(created[0])
    assert pc and pc[-1][1][0] is True and pc[-1][1][3] == "rmvpe"   # applied live


def test_on_precise_done_failure_disables_and_toasts(make):
    b, session, created = make()
    errors = []
    b.errorOccurred.connect(errors.append)
    session.load(object())
    b._set_busy(True)
    b._onPreciseDone(False, None, None, "", "boom")
    assert b.preciseMappingOn is False
    assert errors and "boom" in errors[-1]


# ------------------------------------------------------------- reload replay (cheap)
def test_reload_replays_cached_quantiles_without_rebuild(make):
    b, session, created = make()
    src_q = np.linspace(6.0, 8.0, 48)
    tgt_q = np.linspace(7.0, 9.0, 48)
    b._desired["precise"] = (True, src_q, tgt_q, "rmvpe")
    session.load(object())
    b._pending_key = "m.pth"
    b._pending_audio = object()
    b._onLoaded(True, "")                         # replay path
    pc = _precise_calls(created[0])
    assert pc and pc[-1][1][0] is True            # set_precise_mapping(True, ...) replayed
    assert created[0].build_count == 0            # NO multi-second rebuild on the GUI thread


# --------------------------------------------------------------- model-swap reset
def test_model_swap_resets_precise(make, monkeypatch):
    b, session, created = make()
    monkeypatch.setattr(backend_mod.ca, "model_default_params",
                        lambda path: {"pitch_shift": 0, "protect": 0.33,
                                      "index_rate": 0.0, "formant_timbre": 1.0,
                                      "formant_on": False})
    session.load(object())
    b._preciseMappingOn = True
    b._preciseTargetWav = "old_model.wav"
    b._desired["precise"] = (True, np.zeros(48), np.zeros(48), "rmvpe")
    b.selectModel("new/Model.pth")
    assert b.preciseMappingOn is False
    assert b._preciseTargetWav == ""              # model-specific target cleared
    assert "precise" not in b._desired


# --------------------------------------------------------------- worker build path
def test_worker_build_then_apply(make, qapp):
    b, session, created = make()
    session.load(object())
    session.start(object())
    b._preciseVoiceWav = "you.wav"
    b._preciseTargetWav = "model.wav"
    b.setPreciseMapping(True)                     # spawns PreciseMapWorker
    assert b._precise_thread is not None
    assert _pump(qapp, lambda: b._precise_thread is None)   # build + apply complete
    assert created[0].build_count == 1
    assert b.preciseMappingOn is True
    pc = _precise_calls(created[0])
    assert pc and pc[-1][1][0] is True


# ----------------------------------------------------- saved-map load / save (dropdown)
def _save_map(tmp_path, name="m1", method="rmvpe", voice="me.wav", target="A.wav"):
    src = np.linspace(6.0, 8.0, 48)
    tgt = np.linspace(7.0, 9.0, 48)
    return ps.save_precise_map(name, method, voice, target, src, tgt, str(tmp_path)), src, tgt


def test_load_stages_without_enabling(make, tmp_path):
    # selecting a saved map from the dropdown only LOADS it; the user flips 启用 to apply.
    b, session, created = make()
    session.load(object())
    f, _, _ = _save_map(tmp_path)
    b.loadPreciseMap(f)
    assert b.precisePending is True
    assert b.preciseMappingOn is False                 # NOT auto-enabled (user choice)
    assert b.preciseVoiceName == "me.wav" and b.preciseTargetName == "A.wav"
    assert _precise_calls(created[0]) == []            # nothing applied yet


def test_enable_after_load_applies_without_build(make, tmp_path):
    b, session, created = make()
    session.load(object()); session.start(object())
    f, _, _ = _save_map(tmp_path, method="fcpe")
    b.loadPreciseMap(f)
    b.setPreciseMapping(True)
    assert b.preciseMappingOn is True
    assert created[0].build_count == 0                 # loaded map applied, NO rebuild
    pc = _precise_calls(created[0])
    assert pc and pc[-1][1][0] is True and pc[-1][1][3] == "fcpe"


def test_enable_loaded_map_before_engine_loaded_then_replays(make, tmp_path):
    # a loaded map needs no engine to enable; it replays on the next load.
    b, session, created = make()
    errors = []
    b.errorOccurred.connect(errors.append)
    f, _, _ = _save_map(tmp_path, name="m3")
    b.loadPreciseMap(f)                                # engine NOT loaded (IDLE)
    b.setPreciseMapping(True)
    assert errors == [] and b.preciseMappingOn is True
    assert b._desired["precise"][0] is True            # staged for replay
    session.load(object())
    b._pending_key = "m.pth"; b._pending_audio = object()
    b._onLoaded(True, "")
    pc = _precise_calls(created[0])
    assert pc and pc[-1][1][0] is True                 # replayed onto the fresh engine


def test_save_requires_a_map(make):
    b, session, created = make()
    errors = []
    b.errorOccurred.connect(errors.append)
    assert b.savePreciseMap("x") is False              # nothing built/loaded yet
    assert errors


def test_save_then_listed_in_maps(make):
    b, session, created = make()
    session.load(object()); session.start(object())
    src = np.linspace(6.0, 8.0, 48); tgt = np.linspace(7.0, 9.0, 48)
    b._set_busy(True); b._onPreciseDone(True, src, tgt, "rmvpe", "")    # builds a pending map
    assert b.precisePending is True
    assert b.savePreciseMap("我的映射") is True
    assert "我的映射" in [m["name"] for m in b.preciseMaps]
    # round-trips back out of the store
    assert "我的映射" in [m["name"] for m in ps.list_precise_maps(backend_mod.ps.PRECISE_DIR)]
