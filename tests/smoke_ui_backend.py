"""Backend-logic smoke for the GUI bridge (NOT a pytest test).

Drives ``meloie.ui.backend.Backend`` with a FAKE RealtimeSession (fake engine +
fake stream runner) under a ``QCoreApplication`` event loop — no QML window, no
audio devices, no torch. Proves: setters delegate to the engine, the metrics
QTimer poll emits a growing QVariantMap, and cross-thread state transitions reach
the GUI thread via the queued signal. Run in .venv-applio:

    . .\\setup_env_applio.ps1
    python tests\\smoke_ui_backend.py

Named smoke_*.py with no test_* functions, so the default pytest run ignores it.
"""

import os
import sys

RVC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RVC)
os.chdir(RVC)

from PySide6.QtCore import QCoreApplication, QTimer   # noqa: E402

from meloie.control import RealtimeSession, SessionState  # noqa: E402
from meloie.ui.backend import Backend                      # noqa: E402


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


def run() -> int:
    app = QCoreApplication(sys.argv)

    created = []

    def factory(cfg):
        e = FakeEngine(cfg)
        created.append(e)
        return e

    session = RealtimeSession(engine_factory=factory, stream_runner=_looping_runner)
    backend = Backend(session=session)

    states, metrics_seen, errors = [], [], []
    backend.stateChanged.connect(states.append)
    backend.metricsChanged.connect(metrics_seen.append)
    backend.errorOccurred.connect(errors.append)

    # drive the session directly (skip the QThread load worker for determinism)
    session.load(object())
    session.start(object())
    backend.setPitch(7)
    backend.setFormant(True, 0.25)
    backend.setSilenceGate(True, -50.0)
    backend.setSid(0)
    backend.setF0Method("rmvpe")
    backend.setMonitor(True)
    backend.setAutoCenter(True)

    result = {}

    def check():
        eng = created[0]
        result["pitch_delegated"] = ("set_pitch_shift", (7,), {}) in eng.calls
        result["formant_delegated"] = any(c[0] == "set_formant" for c in eng.calls)
        result["silence_delegated"] = any(c[0] == "set_silence_gate" for c in eng.calls)
        result["sid_delegated"] = any(c[0] == "set_sid" for c in eng.calls)
        result["f0_delegated"] = any(c[0] == "set_f0_method" for c in eng.calls)
        result["auto_center_delegated"] = any(c[0] == "set_auto_center" for c in eng.calls)
        result["monitor_delegated"] = (session.monitor_enabled is True)
        result["model_api"] = (hasattr(backend, "models") and hasattr(backend, "mergeModels")
                               and hasattr(backend, "selectModel"))
        result["save_defaults_api"] = hasattr(backend, "saveModelDefaults")
        result["sid_reset_signal"] = hasattr(backend, "sidReset")
        result["presets_removed"] = (not hasattr(backend, "presetsChanged")
                                     and not hasattr(backend, "presets"))
        result["running_state_seen"] = SessionState.RUNNING.value in states
        result["metrics_growing"] = bool(metrics_seen) and metrics_seen[-1].get("input_frames", 0) > 0
        result["no_errors"] = (errors == [])
        session.stop()
        app.quit()

    QTimer.singleShot(500, check)
    app.exec()

    ok = all(result.values())
    print("results:", result)
    if errors:
        print("errors:", errors)
    print("PASS: ui backend smoke" if ok else "FAIL: ui backend smoke")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
