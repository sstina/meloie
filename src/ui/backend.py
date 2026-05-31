"""``Backend(QObject)`` — the QML bridge over the Qt-free ``RealtimeSession``.

Correctness rules (see plan): the only blocking call (``load``/``reload`` ~30s)
runs in a ``LoadWorker`` on a ``QThread``; ``RealtimeSession.on_state_change``
fires on the session/stream thread and is marshalled to the GUI thread via a
queued signal; a ``QTimer`` on the GUI thread polls ``metrics_snapshot()``. Every
QML-invoked setter is wrapped so a validation error becomes ``errorOccurred``
instead of an exception crossing into Qt. NO output-shaping slot exists.
"""

from __future__ import annotations

import os
import re

from PySide6.QtCore import (
    Property, QObject, QThread, QTimer, Qt, Signal, Slot,
)

from ..control import RealtimeSession, SessionState
from . import config_assembly as ca
from . import presets as pr


class LoadWorker(QObject):
    """Runs the blocking ``load``/``reload`` off the GUI thread (lives in a QThread)."""

    finished = Signal(bool, str)        # (ok, error_message)

    def __init__(self, session: RealtimeSession, engine_cfg, *, is_reload: bool):
        super().__init__()
        self._session = session
        self._cfg = engine_cfg
        self._is_reload = is_reload

    @Slot()
    def run(self) -> None:
        try:
            if self._is_reload:
                self._session.reload(self._cfg)
            else:
                self._session.load(self._cfg)
            self.finished.emit(True, "")
        except BaseException as exc:    # session already set ERROR + last_error
            self.finished.emit(False, f"{type(exc).__name__}: {exc}")


class MergeWorker(QObject):
    """Runs the (blocking) offline model merge + save off the GUI thread (lives in
    a QThread, mirrors LoadWorker). Reuses src.engine.model_merge; torch is lazy."""

    finished = Signal(bool, str, str)   # (ok, merged_model_path, error_message)

    def __init__(self, model_paths, strengths, out_path, model_path, profile_obj):
        super().__init__()
        self._paths = model_paths
        self._strengths = strengths
        self._out = out_path
        self._prof_path = model_path
        self._prof = profile_obj

    @Slot()
    def run(self) -> None:
        try:
            import json
            import torch
            from ..engine.model_merge import merge_checkpoints
            merged_cpt, _common, _alphas = merge_checkpoints(self._paths, self._strengths)
            out_dir = os.path.dirname(self._out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            torch.save(merged_cpt, self._out)
            # companion recipe so the merged voice loads at a sane pitch (inherited
            # from the base) and the model<->recipe lookup finds it. Best-effort:
            # the .pth is the artifact that matters.
            try:
                os.makedirs(os.path.dirname(self._prof_path), exist_ok=True)
                with open(self._prof_path, "w", encoding="utf-8") as f:
                    json.dump(self._prof, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            self.finished.emit(True, self._out, "")
        except BaseException as exc:
            self.finished.emit(False, "", f"{type(exc).__name__}: {exc}")


class Backend(QObject):
    # ---- signals QML / internals listen to ----
    stateChanged = Signal(str)
    metricsChanged = Signal("QVariantMap")
    errorOccurred = Signal(str)
    busyChanged = Signal(bool)
    devicesChanged = Signal()
    modelsChanged = Signal()
    modelParamsChanged = Signal()
    modelMerged = Signal(str)           # after a merge: the merged model's .pth path
    numSpeakersChanged = Signal()
    sidReset = Signal()                 # every successful (re)load rebuilds the engine at sid 0
    _stateRaw = Signal(str, str)        # emitted on the session thread; queued to GUI thread

    def __init__(self, *, session: RealtimeSession | None = None, parent=None):
        super().__init__(parent)
        if session is None:
            self._session = RealtimeSession(on_state_change=self._on_state_change)
        else:
            self._session = session
            self._session._on_state_change = self._on_state_change  # route state events here

        self._state = self._session.state.value
        self._busy = False
        self._devices = ca.list_device_dicts()
        self._models = ca.list_model_files()      # .pth files discovered in models/
        self._modelParams = {}
        if self._models:                   # pre-init slider defaults from the first model
            try:
                self._modelParams = ca.model_default_params(self._models[0]["path"])
            except Exception:
                self._modelParams = {}

        self._pending_audio = None
        self._pending_key = None
        self._loaded_key = None        # model_path of the loaded engine (f0 is live-swappable)
        self._numSpeakers = 1
        self._monitorOn = False        # desired headphone-monitor state (live + start init)
        self._trayActive = False       # set once at startup (app.py); gates close-to-tray
        self._thread = None
        self._worker = None
        self._merge_thread = None
        self._merge_worker = None

        # session thread -> GUI thread (queued); never touch QML off-thread
        self._stateRaw.connect(self._applyState, Qt.QueuedConnection)

        self._metricsTimer = QTimer(self)
        self._metricsTimer.setInterval(150)
        self._metricsTimer.timeout.connect(self._pollMetrics)
        self._metricsTimer.start()

    # ------------------------------------------------------------ properties
    def _get_state(self):
        return self._state

    state = Property(str, _get_state, notify=stateChanged)

    def _get_busy(self):
        return self._busy

    busy = Property(bool, _get_busy, notify=busyChanged)

    def _get_devices(self):
        return self._devices

    devices = Property("QVariantList", _get_devices, notify=devicesChanged)

    def _get_models(self):
        return self._models

    models = Property("QVariantList", _get_models, notify=modelsChanged)

    def _get_modelParams(self):
        return self._modelParams

    modelParams = Property("QVariantMap", _get_modelParams, notify=modelParamsChanged)

    def _get_presets(self):
        return pr.BUILTIN_PRESETS

    presets = Property("QVariantList", _get_presets, constant=True)

    def _get_numSpeakers(self):
        return self._numSpeakers

    numSpeakers = Property(int, _get_numSpeakers, notify=numSpeakersChanged)

    def _get_trayActive(self):
        return self._trayActive

    trayActive = Property(bool, _get_trayActive, constant=True)   # set once before QML load

    def set_tray_active(self, value: bool) -> None:
        """Called from app.py before QML loads; gates the window's close-to-tray
        handler (True -> close hides to tray; False -> close quits, as a fallback
        when no system tray is available)."""
        self._trayActive = bool(value)

    # --------------------------------------------------- state plumbing (safe)
    def _on_state_change(self, old: SessionState, new: SessionState) -> None:
        # CALLED ON THE SESSION THREAD — only emit a signal, never touch QML.
        self._stateRaw.emit(old.value, new.value)

    @Slot(str, str)
    def _applyState(self, old: str, new: str) -> None:        # GUI thread (queued)
        self._state = new
        self.stateChanged.emit(new)
        if new == SessionState.ERROR.value and self._session.last_error:
            self.errorOccurred.emit(self._session.last_error)

    def _set_busy(self, value: bool) -> None:
        if self._busy != value:
            self._busy = value
            self.busyChanged.emit(value)

    # --------------------------------------------------- metrics poll (GUI thread)
    @Slot()
    def _pollMetrics(self) -> None:
        # Only poll while the stream is actually RUNNING. Once started, the session
        # keeps a (now-frozen) metrics object after stop/error, so polling it forever
        # is pure churn — the last values simply stay on screen.
        if self._state != SessionState.RUNNING.value:
            return
        snap = self._session.metrics_snapshot()
        if snap:
            self.metricsChanged.emit(snap)

    # --------------------------------------------------------------- lifecycle
    @Slot(str)
    def selectModel(self, model_path: str) -> None:
        try:
            self._modelParams = ca.model_default_params(model_path)
            self.modelParamsChanged.emit()
        except Exception as exc:
            self.errorOccurred.emit(f"model error: {exc}")

    @Slot()
    def refreshModels(self) -> None:
        self._models = ca.list_model_files()
        self.modelsChanged.emit()

    @Slot(str, "QVariantMap", result=bool)
    def saveModelDefaults(self, model_path, params) -> bool:
        """Save the current carrier knobs as this model's default (its
        <stem>.json profile), so it auto-loads with them next time. Input-side
        only -> contract-safe. Returns True on success."""
        if not model_path:
            self.errorOccurred.emit("没有选中模型")
            return False
        try:
            pr.save_model_profile(str(model_path), dict(params), ca.PROFILES_DIR)
            return True
        except Exception as exc:
            self.errorOccurred.emit(f"保存失败：{exc}")
            return False

    @Slot(str, str, str, str, str, bool)
    def startOrStop(self, model_path: str, input_substr: str, output_substr: str, f0: str,
                    monitor_substr: str = "", monitor_on: bool = False) -> None:
        """The Start/Stop button. running -> stop; loaded with the same MODEL ->
        start instantly (no reload; f0 is live-swappable); else (re)load + start."""
        if self._busy:                      # a load/merge is in flight; ignore re-entry
            return
        self._monitorOn = bool(monitor_on)
        st = self._state
        if st == SessionState.RUNNING.value:
            self.stop()
            return
        if st == SessionState.LOADED.value and self._loaded_key == model_path:
            self._pending_audio = self._build_audio(input_substr, output_substr, monitor_substr)
            self._startStream(self._pending_audio)
            return
        self._beginLoad(model_path, input_substr, output_substr, f0, monitor_substr,
                        is_reload=(self._session.engine is not None))

    @Slot(str, str, str, str, str, bool)
    def reloadModel(self, model_path: str, input_substr: str, output_substr: str, f0: str,
                    monitor_substr: str = "", monitor_on: bool = False) -> None:
        """Live model / f0 swap while running (stop -> reload -> start)."""
        if self._busy:                      # in-flight (re)load; drop the re-entrant request
            return
        self._monitorOn = bool(monitor_on)
        self._beginLoad(model_path, input_substr, output_substr, f0, monitor_substr, is_reload=True)

    def _beginLoad(self, model_path, input_substr, output_substr, f0, monitor_substr="",
                   *, is_reload) -> None:
        if self._busy:
            return
        try:
            scfg, acfg = ca.build_configs_for_model(
                model_path, input_substr or None, output_substr or None,
                f0=(f0 or ca.DEFAULT_F0), monitor_substr=(monitor_substr or None),
            )
        except Exception as exc:
            self.errorOccurred.emit(f"config error: {exc}")
            return
        self._pending_audio = acfg
        self._pending_key = model_path
        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = LoadWorker(self._session, scfg, is_reload=is_reload)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._onLoaded)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)   # free the QThread too
        self._thread.finished.connect(self._clearLoadThread)      # drop our refs
        self._thread.start()

    @Slot(bool, str)
    def _onLoaded(self, ok: bool, err: str) -> None:           # GUI thread (queued)
        self._set_busy(False)
        if not ok:
            # the ERROR state transition already surfaced this via _applyState — do
            # not double-emit. (err carries the same "Type: msg" string.)
            return
        self._loaded_key = self._pending_key
        n = int(self._session.num_speakers)
        if n != self._numSpeakers:
            self._numSpeakers = n
            self.numSpeakersChanged.emit()
        # a fresh (re)load always rebuilds the engine at sid 0; re-sync the spinbox
        # even when the speaker count is unchanged (numSpeakersChanged wouldn't fire).
        self.sidReset.emit()
        self._startStream(self._pending_audio)

    @Slot()
    def _clearLoadThread(self) -> None:        # GUI thread (queued via thread.finished)
        self._thread = None
        self._worker = None

    def _build_audio(self, input_substr, output_substr, monitor_substr=""):
        # single source of truth shared with the load path (build_configs_for_model)
        return ca.build_audio_config(input_substr or None, output_substr or None,
                                     monitor_substr or None)

    def _startStream(self, acfg) -> None:
        # session.start() only does synchronous validation then launches the stream
        # thread; real start failures (bad device, PortAudioError, cable guard) arrive
        # asynchronously via the ERROR state -> _applyState. This try/except only
        # catches the synchronous SessionError guards.
        try:
            self._session.start(acfg, allow_virtual_cable_input=False,
                                monitor_enabled=self._monitorOn)
        except Exception as exc:
            self.errorOccurred.emit(str(exc))

    @Slot()
    def stop(self) -> None:
        try:
            self._session.stop()
        except Exception as exc:
            self.errorOccurred.emit(str(exc))

    @Slot()
    def refreshDevices(self) -> None:
        self._devices = ca.list_device_dicts()
        self.devicesChanged.emit()

    def shutdown(self) -> None:
        self._metricsTimer.stop()
        try:
            self._session.close()
        except Exception:
            pass
        # A load/merge worker runs ONE long blocking call (cold engine.load() ~tens of
        # seconds; merge torch.save), so quit() can't preempt it. Wait WITHOUT a finite
        # timeout: the worker is bounded and will finish, and the thread MUST terminate
        # before its QObject is destroyed, or Qt aborts ("QThread: Destroyed while thread
        # is still running"). Already-finished threads are nil (skipped).
        for t in (self._thread, self._merge_thread):
            if t is not None:
                t.quit()
                t.wait()

    # ----------------------------------------------- live INPUT-side setters
    def _guard(self, fn) -> None:
        if self._busy:
            # A load/reload/merge is tearing down/rebuilding the engine; skip the
            # live setter rather than poke a half-built engine (which would raise a
            # SessionError -> spurious error toast). Slider state is re-synced after
            # the load completes via modelParamsChanged / sidReset.
            return
        try:
            fn()
        except Exception as exc:
            self.errorOccurred.emit(str(exc))

    @Slot(int)
    def setPitch(self, v):
        self._guard(lambda: self._session.set_pitch_shift(int(v)))

    @Slot(float)
    def setProtect(self, v):
        self._guard(lambda: self._session.set_protect(float(v)))

    @Slot(float)
    def setIndexRate(self, v):
        self._guard(lambda: self._session.set_index_rate(float(v)))

    @Slot(bool, float)
    def setFormant(self, on, timbre):
        self._guard(lambda: self._session.set_formant(bool(on), timbre=float(timbre)))

    @Slot(bool, float, bool)
    def setDenoise(self, on, strength, nonstationary):
        self._guard(lambda: self._session.set_denoise(
            bool(on), strength=float(strength), nonstationary=bool(nonstationary)))

    @Slot(bool, float)
    def setSilenceGate(self, on, dbfs):
        self._guard(lambda: self._session.set_silence_gate(float(dbfs) if on else None))

    @Slot(bool, float)
    def setAutotune(self, on, strength):
        self._guard(lambda: self._session.set_autotune(bool(on), strength=float(strength)))

    @Slot(bool, float)
    def setAutoPitch(self, on, threshold):
        self._guard(lambda: self._session.set_auto_pitch(bool(on), threshold=float(threshold)))

    @Slot(int)
    def setSid(self, v):
        self._guard(lambda: self._session.set_sid(int(v)))

    @Slot(str)
    def setF0Method(self, v):
        # Live F0-estimator swap (rmvpe<->fcpe); input-side carrier conditioning,
        # no reload (f0 is not part of the reload key).
        self._guard(lambda: self._session.set_f0_method(str(v)))

    @Slot(bool)
    def setMonitor(self, v):
        # Live headphone-monitor on/off (routing duplicate, not output shaping).
        # Remember the desired state so a subsequent (re)start preserves it.
        self._monitorOn = bool(v)
        self._guard(lambda: self._session.set_monitor_enabled(bool(v)))

    # ------------------------------------------------- model merge (融合模式)
    @Slot(str, float, "QVariantList", str, int, str)
    def mergeModels(self, base_path, base_weight, others, out_name, base_pitch, base_f0):
        """Blend the base model with the selected others (each weighted) into a new
        models/<name>.pth, write a companion recipe (pitch inherited from the base),
        then refresh + auto-select it. Runs off the GUI thread. Contract-safe: the
        merged MODEL defines the voice; the runtime never reshapes output."""
        if self._busy:
            return
        if not base_path:
            self.errorOccurred.emit("没有基础模型")
            return
        others = list(others or [])
        if not others:
            self.errorOccurred.emit("选至少一个要融合的模型")
            return
        name = re.sub(r'[<>:"/\\|?*]+', "_", (out_name or "").strip()) or "merged"
        out_path = os.path.join(ca.models_dir(), name + ".pth")
        if os.path.exists(out_path):
            self.errorOccurred.emit(f"模型 “{name}” 已存在，换个名字")
            return
        paths = [base_path] + [o["path"] for o in others]
        strengths = [float(base_weight)] + [float(o.get("weight", 1.0)) for o in others]
        rel = "models/" + name + ".pth"     # flat models dir; correct in source + frozen
        prof = {
            "name": name, "model_path": rel,
            "f0_method": str(base_f0 or ca.DEFAULT_F0),
            "index_rate": 0.0, "pitch_shift": int(base_pitch),
            "notes": "merged in GUI; index_rate 0 (no shared index).",
        }
        prof_path = os.path.join(ca.PROFILES_DIR, name + ".json")
        self._set_busy(True)
        self._merge_thread = QThread(self)
        self._merge_worker = MergeWorker(paths, strengths, out_path, prof_path, prof)
        self._merge_worker.moveToThread(self._merge_thread)
        self._merge_thread.started.connect(self._merge_worker.run)
        self._merge_worker.finished.connect(self._onMerged)
        self._merge_worker.finished.connect(self._merge_thread.quit)
        self._merge_thread.finished.connect(self._merge_worker.deleteLater)
        self._merge_thread.finished.connect(self._merge_thread.deleteLater)
        self._merge_thread.finished.connect(self._clearMergeThread)
        self._merge_thread.start()

    @Slot(bool, str, str)
    def _onMerged(self, ok, out_path, err):       # GUI thread (queued)
        self._set_busy(False)
        if not ok:
            self.errorOccurred.emit(f"融合失败：{err}")
            return
        self._models = ca.list_model_files()
        self.modelsChanged.emit()
        self.modelMerged.emit(out_path)           # QML selects it + reloads if running

    @Slot()
    def _clearMergeThread(self) -> None:       # GUI thread (queued via thread.finished)
        self._merge_thread = None
        self._merge_worker = None
