"""Regression tests for Backend's desired-knob capture + replay-after-load.

Pins the fix for the bug where a creative-card slider (变调 pitch etc.) adjusted
BEFORE Start or DURING the ~30s model load was lost — only runtime edits landed.
The Backend now records every input-side knob in ``self._desired`` and replays it
onto the freshly-(re)loaded engine in ``_onLoaded`` (before the stream starts).

Drives the real ``Backend`` (a QObject) under a ``QCoreApplication`` with a FAKE
RealtimeSession (fake engine + fake runner) — no torch, no audio, no QML. The
flaky QThread load worker is bypassed: we drive ``session.load`` directly and call
``backend._onLoaded(True, "")`` like the worker's ``finished`` slot would, so the
test is deterministic. Mirrors tests/smoke_ui_backend.py's fakes.
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
        # record any set_* delegation generically; everything else (e.g.
        # num_speakers) raises -> session.num_speakers falls back to 1 via getattr.
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
    # one QCoreApplication per process (two instances abort); reuse if present.
    return QCoreApplication.instance() or QCoreApplication([])


@pytest.fixture
def make(qapp, monkeypatch):
    """Build a Backend over a fake session. Returns a factory; cleans up streams +
    the metrics QTimer for every Backend built (even if a test fails mid-way)."""
    built = []

    def _make():
        # hermetic init: no real devices / model enumeration -> _desired seeds neutral.
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


def _pitch_calls(engine):
    return [c for c in engine.calls if c[0] == "set_pitch_shift"]


def _finish_load(b, session):
    """Mimic the LoadWorker.finished slot: load done -> _onLoaded marshals it."""
    b._pending_key = "fake.pth"
    b._pending_audio = object()
    b._onLoaded(True, "")


# --------------------------------------------------------------------------- tests
def test_pitch_before_start_is_captured_without_toast(make):
    # The headline regression: pre-start, moving the slider must NOT raise an error
    # toast and must NOT be lost — it is captured in _desired for replay.
    b, session, created = make()
    errors = []
    b.errorOccurred.connect(errors.append)

    b.setPitch(7)

    assert errors == []                 # today's bug emitted "no active engine" here
    assert created == []                # no engine exists yet -> nothing applied live
    assert b._desired["pitch"] == 7     # captured


def test_pitch_set_before_start_is_replayed_on_load(make):
    b, session, created = make()
    errors = []
    b.errorOccurred.connect(errors.append)

    b.setPitch(7)
    session.load(object())              # engine built (profile defaults), LOADED
    _finish_load(b, session)            # _onLoaded -> replay -> _startStream

    assert ("set_pitch_shift", (7,), {}) in created[0].calls
    assert errors == []


def test_during_load_edit_latest_value_wins(make):
    # An edit made while the load is in flight (busy) must also be captured, and the
    # latest value wins; the earlier (never-applied) value must not leak through.
    b, session, created = make()
    errors = []
    b.errorOccurred.connect(errors.append)

    b.setPitch(7)
    b._set_busy(True)                   # load in flight
    b.setPitch(9)                       # during-load edit -> stored, not applied live
    assert errors == []

    session.load(object())
    b._set_busy(False)
    _finish_load(b, session)

    pc = _pitch_calls(created[0])
    assert pc and pc[-1] == ("set_pitch_shift", (9,), {})
    assert ("set_pitch_shift", (7,), {}) not in created[0].calls   # overwritten, never applied


def test_pitch_while_running_applies_live(make):
    b, session, created = make()
    errors = []
    b.errorOccurred.connect(errors.append)

    session.load(object())
    session.start(object())             # RUNNING
    b.setPitch(3)

    assert ("set_pitch_shift", (3,), {}) in created[0].calls
    assert errors == []


def test_advanced_toggle_survives_reload(make):
    # Bonus fix: denoise/silence/autotune/auto-pitch used to silently reset on every
    # reload (UI showed ON, engine was OFF). They must now persist onto the new engine.
    b, session, created = make()

    session.load(object())
    session.start(object())
    b.setDenoise(True, 0.5, True)
    assert any(c[0] == "set_denoise" for c in created[0].calls)

    session.reload(object())            # stop + build a fresh engine (created[1])
    _finish_load(b, session)            # replay onto the new engine

    assert len(created) == 2
    assert any(c[0] == "set_denoise" for c in created[1].calls)


def test_select_model_reseeds_desired_pitch(make, monkeypatch):
    # Selecting a model re-seeds the profile-driven knobs (pitch here) so a later
    # load reproduces THAT model's default, not a stale one.
    b, session, created = make()
    monkeypatch.setattr(backend_mod.ca, "model_default_params",
                        lambda path: {"pitch_shift": 12, "protect": 0.33,
                                      "index_rate": 0.0, "formant_timbre": 1.0,
                                      "formant_on": False})
    b.selectModel("whatever/X.pth")
    assert b._desired["pitch"] == 12
