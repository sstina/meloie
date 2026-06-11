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

from PySide6.QtCore import (
    Property, QObject, QThread, QTimer, Qt, Signal, Slot,
)

from ..control import RealtimeSession, SessionState
from . import config_assembly as ca
from . import precise_store as ps
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
    a QThread, mirrors LoadWorker). Reuses meloie.engine.model_merge; torch is lazy."""

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
            from ..engine.model_merge import (
                merge_checkpoints, save_merged_checkpoint, write_merge_profile,
            )
            merged_cpt, _common, _alphas = merge_checkpoints(self._paths, self._strengths)
            # shared save + integrity re-load (same path as the CLI): a truncated /
            # non-v2 write surfaces as a merge failure here, not a load error later.
            save_merged_checkpoint(self._out, merged_cpt)
            # companion recipe so the merged voice loads at a sane pitch (inherited
            # from the base) and the model<->recipe lookup finds it. Best-effort:
            # the .pth is the artifact that matters.
            try:
                write_merge_profile(self._prof_path, **self._prof)
            except Exception:
                pass
            self.finished.emit(True, self._out, "")
        except BaseException as exc:
            self.finished.emit(False, "", f"{type(exc).__name__}: {exc}")


class PreciseMapWorker(QObject):
    """Builds the precise CDF F0 map (load 2 wavs + run the f0 estimator) off the GUI
    thread (lives in a QThread, mirrors LoadWorker/MergeWorker). The estimator runs off
    the engine's audio lock, so it never stalls realtime inference."""

    finished = Signal(bool, object, object, str, str)   # (ok, src_q, tgt_q, method, error)

    def __init__(self, session: RealtimeSession, voice_wav: str, target_wav: str):
        super().__init__()
        self._session = session
        self._voice = voice_wav
        self._target = target_wav

    @Slot()
    def run(self) -> None:
        try:
            src_q, tgt_q, method = self._session.build_precise_map(self._voice, self._target)
            self.finished.emit(True, src_q, tgt_q, method, "")
        except BaseException as exc:
            self.finished.emit(False, None, None, "", f"{type(exc).__name__}: {exc}")


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
    preciseChanged = Signal()           # precise-mapping state (paths / on / status)
    gameModeChanged = Signal()          # game-mode (off / dgpu_light / cpu_zero)
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

        # Desired INPUT-side carrier knobs (the live UI state we must reproduce on a
        # fresh engine). Seeded from the model's profile defaults — identical to how
        # QML seeds the sliders — and updated by every live setter. Replayed onto the
        # engine right after (re)load (see _apply_desired_to_engine), so edits made
        # BEFORE Start or DURING the ~30s load are not lost. All input-side (faithful-
        # carrier contract); sid/f0/monitor are deliberately excluded (see their setters).
        self._desired = {}
        self._seed_desired_from_params(self._modelParams)

        self._pending_audio = None
        self._pending_key = None
        self._loaded_key = None        # model_path of the loaded engine (f0 is live-swappable)
        self._numSpeakers = 1
        self._monitorOn = False        # desired headphone-monitor state (live + start init)

        # Game mode (off / dgpu_light / cpu_zero): trades precision/latency for low or
        # zero dGPU usage while gaming. The DEVICE/block overrides ride the load bundle
        # (ca.build_configs_for_model(game_mode=)); the "sacrifice precision" LIVE levers
        # (index_rate -> 0, silence gate on) go through the normal setters so _desired
        # stays the source of truth and "off" can restore them. _preGameKnobs snapshots
        # those levers on entry so "off" restores the user's exact prior values.
        # _lastLoadArgs remembers the last full (model, devices, f0, monitor) so a mode
        # change while RUNNING can reload onto the new device with the same routing.
        self._gameMode = "off"
        self._preGameKnobs = None
        self._lastLoadArgs = None
        self._trayActive = False       # set once at startup (app.py); gates close-to-tray
        self._thread = None
        self._worker = None
        self._merge_thread = None
        self._merge_worker = None
        self._precise_thread = None
        self._precise_worker = None

        # Precise F0 mapping (input-side). The two wav paths the user picks, plus the
        # live on/off + a short status line. The built quantiles live in
        # self._desired["precise"] (cheap reload-replay); the wav PATHS are kept here so
        # an f0-method swap can rebuild the map under the new estimator.
        self._preciseVoiceWav = ""     # 你的声音 (source: the live user's own voice)
        self._preciseTargetWav = ""    # 模型原声 (target: the model's native voice)
        self._preciseVoiceLabel = ""   # display name (wav basename OR a loaded map's meta)
        self._preciseTargetLabel = ""
        self._preciseMappingOn = False
        self._preciseStatus = ""
        # the map ready to apply when 启用 is flipped on: (src_q, tgt_q, method), set
        # by a finished build OR by loading a saved map (then no rebuild is needed).
        self._precisePendingQ = None
        self._preciseMaps = ps.list_precise_maps(ps.PRECISE_DIR)   # saved maps for the dropdown

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

    def _get_numSpeakers(self):
        return self._numSpeakers

    numSpeakers = Property(int, _get_numSpeakers, notify=numSpeakersChanged)

    # game mode — exposed for the QML selector (which option is active)
    def _get_gameMode(self):
        return self._gameMode

    gameMode = Property(str, _get_gameMode, notify=gameModeChanged)

    # precise F0 mapping — exposed for the QML card + the "gray out pitch controls" binding
    def _get_preciseMappingOn(self):
        return self._preciseMappingOn

    preciseMappingOn = Property(bool, _get_preciseMappingOn, notify=preciseChanged)

    def _get_preciseVoiceName(self):
        return self._preciseVoiceLabel

    preciseVoiceName = Property(str, _get_preciseVoiceName, notify=preciseChanged)

    def _get_preciseTargetName(self):
        return self._preciseTargetLabel

    preciseTargetName = Property(str, _get_preciseTargetName, notify=preciseChanged)

    def _get_preciseStatus(self):
        return self._preciseStatus

    preciseStatus = Property(str, _get_preciseStatus, notify=preciseChanged)

    def _get_precisePending(self):
        return self._precisePendingQ is not None

    precisePending = Property(bool, _get_precisePending, notify=preciseChanged)

    def _get_preciseMaps(self):
        return self._preciseMaps

    preciseMaps = Property("QVariantList", _get_preciseMaps, notify=preciseChanged)

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
        # a switch during an in-flight ~30s load would re-seed _desired to the NEW
        # model while the OLD one finishes loading (then _onLoaded replays the new
        # knobs onto the old engine) -> ignore, like every other busy-gated slot.
        if self._busy:
            return
        try:
            self._modelParams = ca.model_default_params(model_path)
            # re-seed the desired carrier knobs to the new model's recipe (mirrors
            # QML initSliders resetting pitch/protect/index/formant/auto-center) so a
            # subsequent load reproduces THIS model's defaults, not the prior model's.
            self._seed_desired_from_params(self._modelParams)
            # an active game mode owns index_rate (forced 0): keep the lever applied
            # over the new model's seed, and retarget the off-restore snapshot at the
            # NEW profile's value (not the previous model's recipe).
            if self._gameMode != "off":
                self._desired["index_rate"] = 0.0
                if self._preGameKnobs is not None:
                    self._preGameKnobs["index_rate"] = float(
                        self._modelParams.get("index_rate", 0.0))
            self.modelParamsChanged.emit()
            # a model swap makes the chosen 模型原声 (target) stale -> drop the precise
            # map + clear the target wav (model-specific); keep the user's voice wav.
            self._reset_precise_for_model_change()
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
        # remember the full routing so a later game-mode switch can reload in place.
        self._lastLoadArgs = (model_path, input_substr, output_substr, f0, monitor_substr)
        st = self._state
        if st == SessionState.RUNNING.value:
            self.stop()
            return
        if st == SessionState.LOADED.value and self._loaded_key == model_path:
            # fast path: engine already LOADED -> start without a reload. No knob replay
            # needed here — every prior setter applied live to this same engine (it has
            # existed since the earlier load), so self._desired and the engine are in sync.
            # Defensively re-apply the combo's f0 (live-swappable; the engine no-ops
            # when unchanged) so the stream can never start on a stale estimator.
            self._apply_live(lambda: self._session.set_f0_method(str(f0 or ca.DEFAULT_F0)))
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
        self._lastLoadArgs = (model_path, input_substr, output_substr, f0, monitor_substr)
        self._beginLoad(model_path, input_substr, output_substr, f0, monitor_substr, is_reload=True)

    def _beginLoad(self, model_path, input_substr, output_substr, f0, monitor_substr="",
                   *, is_reload) -> None:
        if self._busy:
            return
        try:
            scfg, acfg = ca.build_configs_for_model(
                model_path, input_substr or None, output_substr or None,
                f0=(f0 or ca.DEFAULT_F0), monitor_substr=(monitor_substr or None),
                game_mode=self._gameMode,
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
        # replay the live UI knobs onto the fresh engine BEFORE the stream starts, so
        # pitch/protect/index/formant (+ advanced toggles) chosen before Start or during
        # the load take effect — and a reload/F0-swap preserves the user's current knobs.
        self._apply_desired_to_engine()
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
        for t in (self._thread, self._merge_thread, self._precise_thread):
            if t is not None:
                try:
                    t.quit()
                    t.wait()
                except RuntimeError:    # C++ QThread already deleted (deleteLater
                    pass                # race) -> keep cleaning the remaining ones

    # ----------------------------------------------- live INPUT-side setters
    # Every setter records its value in self._desired FIRST (so it survives a load),
    # then tries to apply it live via _apply_live. self._desired is replayed onto a
    # freshly-(re)loaded engine in _onLoaded, so edits made before Start or during the
    # ~30s load are captured (not lost / not an error toast). All input-side.

    def _seed_desired_from_params(self, params) -> None:
        """Seed the four profile-driven carrier knobs (pitch/protect/index_rate/
        formant) from a model's default params (mirrors QML initSliders). The advanced
        toggles (denoise/silence/autotune/auto_pitch) are PRESERVED — QML keeps those
        checkboxes across a model change; auto_center is dropped — QML resets it off.
        Tolerates an empty/partial dict (model with no profile, or enumeration failure)."""
        p = params or {}
        nd = ca.NEUTRAL_DEFAULTS                     # single source of the fallbacks
        self._desired["pitch"] = int(p.get("pitch_shift", nd["pitch_shift"]))
        self._desired["protect"] = float(p.get("protect", nd["protect"]))
        self._desired["index_rate"] = float(p.get("index_rate", nd["index_rate"]))
        self._desired["formant"] = (bool(p.get("formant_on", False)),
                                    float(p.get("formant_timbre", nd["formant_timbre"])))
        self._desired.pop("auto_center", None)
        self._desired.pop("precise", None)        # a model swap invalidates the target voice

    def _apply_desired_to_engine(self) -> None:
        """Replay the captured input-side knobs onto the just-loaded engine, right
        before the stream starts (called from _onLoaded: GUI thread, engine LOADED,
        not busy, stream not yet running). Best-effort: each knob is applied in a
        try/except that SWALLOWS errors — an automatic replay must never block the
        stream start nor spam error toasts (e.g. a malformed profile with index_rate>0
        but no index file would raise; the user still sees a live error if they touch
        that slider while running)."""
        d = self._desired
        s = self._session
        plan = []
        if "pitch" in d:       plan.append(lambda: s.set_pitch_shift(d["pitch"]))
        if "protect" in d:     plan.append(lambda: s.set_protect(d["protect"]))
        if "formant" in d:     plan.append(lambda: s.set_formant(d["formant"][0], timbre=d["formant"][1]))
        if "denoise" in d:     plan.append(lambda: s.set_denoise(d["denoise"][0], strength=d["denoise"][1], nonstationary=d["denoise"][2]))
        if "silence" in d:     plan.append(lambda: s.set_silence_gate(d["silence"][1] if d["silence"][0] else None))
        if "autotune" in d:    plan.append(lambda: s.set_autotune(d["autotune"][0], strength=d["autotune"][1]))
        if "auto_pitch" in d:  plan.append(lambda: s.set_auto_pitch(d["auto_pitch"][0], threshold=d["auto_pitch"][1]))
        if "index_rate" in d:  plan.append(lambda: s.set_index_rate(d["index_rate"]))   # after index is loaded
        if "auto_center" in d: plan.append(lambda: s.set_auto_center(d["auto_center"][0]))
        # precise mapping LAST (it REPLACES the pitch knobs above): cheap replay of the
        # cached quantiles (NEVER re-runs the multi-second build on the GUI thread).
        if d.get("precise", (False,))[0]:
            prc = d["precise"]
            plan.append(lambda prc=prc: s.set_precise_mapping(True, prc[1], prc[2], prc[3]))
        for fn in plan:
            try:
                fn()
            except Exception:
                pass

    def _apply_live(self, fn) -> None:
        """Push a knob to the LIVE engine — only when one exists and we're not mid-
        (re)load. Otherwise the value is already captured in self._desired and gets
        replayed when the engine finishes loading, so pre-start / during-load edits
        are preserved instead of raising. Gate on the session's authoritative state
        (mirrors RealtimeSession._require_active), not the queued GUI string mirror."""
        if (self._busy or self._session.engine is None
                or self._session.state not in (SessionState.LOADED, SessionState.RUNNING)):
            return
        try:
            fn()
        except Exception as exc:
            self.errorOccurred.emit(str(exc))

    @Slot(int)
    def setPitch(self, v):
        self._desired["pitch"] = int(v)
        self._apply_live(lambda: self._session.set_pitch_shift(int(v)))

    @Slot(float)
    def setProtect(self, v):
        self._desired["protect"] = float(v)
        self._apply_live(lambda: self._session.set_protect(float(v)))

    @Slot(float)
    def setIndexRate(self, v):
        self._desired["index_rate"] = float(v)
        self._apply_live(lambda: self._session.set_index_rate(float(v)))

    @Slot(bool, float)
    def setFormant(self, on, timbre):
        self._desired["formant"] = (bool(on), float(timbre))
        self._apply_live(lambda: self._session.set_formant(bool(on), timbre=float(timbre)))

    @Slot(bool, float, bool)
    def setDenoise(self, on, strength, nonstationary):
        self._desired["denoise"] = (bool(on), float(strength), bool(nonstationary))
        self._apply_live(lambda: self._session.set_denoise(
            bool(on), strength=float(strength), nonstationary=bool(nonstationary)))

    @Slot(bool, float)
    def setSilenceGate(self, on, dbfs):
        # store the raw (on, dbfs) pair, not the collapsed None, so a toggle off->on
        # across a reload keeps the chosen dbfs.
        self._desired["silence"] = (bool(on), float(dbfs))
        self._apply_live(lambda: self._session.set_silence_gate(float(dbfs) if on else None))

    @Slot(bool, float)
    def setAutotune(self, on, strength):
        self._desired["autotune"] = (bool(on), float(strength))
        self._apply_live(lambda: self._session.set_autotune(bool(on), strength=float(strength)))

    @Slot(bool, float)
    def setAutoPitch(self, on, threshold):
        self._desired["auto_pitch"] = (bool(on), float(threshold))
        self._apply_live(lambda: self._session.set_auto_pitch(bool(on), threshold=float(threshold)))

    @Slot(bool)
    def setAutoCenter(self, on):
        # A2 auto pitch-centering: just flip the flag; the per-model target_hz was
        # seeded into the engine cfg at load (from the profile's target_f0_median).
        self._desired["auto_center"] = (bool(on),)
        self._apply_live(lambda: self._session.set_auto_center(bool(on)))

    # sid / f0 / monitor are NOT tracked in self._desired: sid resets to 0 on every
    # (re)load by design (sidReset); f0 flows through build_configs + _lastLoadArgs
    # (setF0Method keeps the latter current) + the fast-path defensive re-apply;
    # monitor is remembered in self._monitorOn and re-applied by start(). They still
    # use the live-apply gate so a pre-start poke is a quiet no-op (no error toast).
    @Slot(int)
    def setSid(self, v):
        self._apply_live(lambda: self._session.set_sid(int(v)))

    @Slot(str)
    def setF0Method(self, v):
        # Live F0-estimator swap (rmvpe<->fcpe); input-side carrier conditioning,
        # no reload (f0 is not part of the reload key). Works in LOADED and RUNNING
        # via the live-apply gate; with nothing loaded it is a quiet no-op (Start
        # passes the combo value into build_configs anyway).
        v = str(v)
        # keep the remembered routing in sync so a later in-place reload (game-mode
        # switch) rebuilds with the NEW estimator, not the one from the last Start.
        if self._lastLoadArgs is not None:
            mp, ins, outs, _f0, mon = self._lastLoadArgs
            self._lastLoadArgs = (mp, ins, outs, v, mon)
        self._apply_live(lambda: self._session.set_f0_method(v))
        # rmvpe/fcpe have different F0 statistics -> a map built under one is invalid
        # under the other; rebuild from the live wavs under the new method. A map LOADED
        # from disk has no live wavs -> keep its quantiles (can't rebuild without sources).
        if self._preciseMappingOn and self._preciseVoiceWav and self._preciseTargetWav:
            self._precisePendingQ = None              # force a fresh build, not a reuse
            self.setPreciseMapping(True)

    @Slot(bool)
    def setMonitor(self, v):
        # Live headphone-monitor on/off (routing duplicate, not output shaping).
        # Remember the desired state so a subsequent (re)start preserves it.
        self._monitorOn = bool(v)
        self._apply_live(lambda: self._session.set_monitor_enabled(bool(v)))

    # ------------------------------------------------- game mode (游戏模式)
    @Slot(str)
    def setGameMode(self, mode) -> None:
        """Switch game mode (off / dgpu_light / cpu_zero). Sets the precision levers
        live (index_rate -> 0 + silence gate on for an active mode; restores them on
        off) and reloads onto the mode's device/block bundle — reloading in place when
        RUNNING, deferring to the next Start when merely LOADED/idle. Contract-safe:
        these are input-side load knobs, never output shaping; the model still defines
        the voice (just at lower precision / higher latency)."""
        mode = str(mode)
        if mode not in ca.GAME_MODES:
            self.errorOccurred.emit(f"未知游戏模式：{mode}")
            return
        if self._busy or mode == self._gameMode:    # a load is in flight, or no change
            self.gameModeChanged.emit()             # snap the segmented pill back to the true mode
            return
        prev = self._gameMode
        if mode == "off":
            self._gameMode = "off"
            self._restore_pre_game_knobs()
        else:
            if prev == "off":                       # snapshot once, so off restores the
                self._snapshot_pre_game_knobs()     # user's true pre-game precision knobs
            self._gameMode = mode
            # sacrifice-precision levers shared by both active modes (only touch
            # index_rate when an index is actually in use, to avoid a no-index raise).
            if float(self._desired.get("index_rate", 0.0)) > 0.0:
                self.setIndexRate(0.0)
            self.setSilenceGate(True, -45.0)
        self.gameModeChanged.emit()
        self._reload_for_game_mode()

    def _snapshot_pre_game_knobs(self) -> None:
        self._preGameKnobs = {
            "index_rate": float(self._desired.get("index_rate", 0.0)),
            "silence": self._desired.get("silence"),   # (on, dbfs) or None if never set
        }

    def _restore_pre_game_knobs(self) -> None:
        snap = self._preGameKnobs or {}
        self._preGameKnobs = None
        rate = float(snap.get("index_rate",
                              self._modelParams.get("index_rate", 0.0)))
        # only restore a >0 rate when the (possibly switched-to) model actually has
        # an index — set_index_rate(>0) raises without one (mirrors setGameMode only
        # zeroing when >0). Unknown params (hermetic {}) -> trust the snapshot.
        if rate > 0.0 and self._modelParams and not self._modelParams.get("has_index"):
            rate = 0.0
        self.setIndexRate(rate)
        sil = snap.get("silence")
        if sil is None:
            self.setSilenceGate(False, -45.0)          # was never set -> leave it off
        else:
            self.setSilenceGate(bool(sil[0]), float(sil[1]))

    def _reload_for_game_mode(self) -> None:
        # Apply the mode's device/block bundle. RUNNING -> reload in place (stop ->
        # reload onto the new device -> restart, exactly like a live model swap).
        # LOADED-but-stopped -> drop the fast-path key so the NEXT Start does a full
        # reload onto the new device (never auto-start audio from a stopped state).
        # IDLE/ERROR -> nothing is loaded; the mode simply applies on the next Start.
        # Gate on the session's AUTHORITATIVE state (like _apply_live), not the queued
        # GUI mirror, so a switch right after Start sees the real RUNNING state.
        st = self._session.state
        if st == SessionState.RUNNING and self._lastLoadArgs is not None:
            mp, ins, outs, f0, mon = self._lastLoadArgs
            self._beginLoad(mp, ins, outs, f0, mon, is_reload=True)
        elif st == SessionState.LOADED:
            self._loaded_key = None

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
        name = ca.safe_filename(out_name, "merged")
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
        self.refreshModels()                      # re-list models/ (picks up the new .pth)
        self.modelMerged.emit(out_path)           # QML selects it + reloads if running

    @Slot()
    def _clearMergeThread(self) -> None:       # GUI thread (queued via thread.finished)
        self._merge_thread = None
        self._merge_worker = None

    # ------------------------------------------- precise F0 mapping (精确映射)
    @Slot()
    def choosePreciseVoiceWav(self) -> None:
        self._pickPreciseWav(is_target=False)

    @Slot()
    def choosePreciseTargetWav(self) -> None:
        self._pickPreciseWav(is_target=True)

    def _pickPreciseWav(self, *, is_target: bool) -> None:
        # Native picker via QtWidgets (the app runs on QApplication) -> a clean local
        # path, no QML Dialogs module / URL conversion. Imported lazily so module import
        # stays cheap and QtWidgets is only needed when actually picking.
        from PySide6.QtWidgets import QFileDialog
        caption = "选择模型原声 (.wav)" if is_target else "选择你的声音 (.wav)"
        path, _ = QFileDialog.getOpenFileName(None, caption, "", "WAV 音频 (*.wav)")
        if not path:
            return
        name = os.path.basename(path)
        if is_target:
            self._preciseTargetWav = path
            self._preciseTargetLabel = name
        else:
            self._preciseVoiceWav = path
            self._preciseVoiceLabel = name
        # a new source wav means "build fresh on enable" -> drop any pending/loaded map.
        self._precisePendingQ = None
        if self._preciseMappingOn:                  # was on -> turn off; re-enable rebuilds
            self._preciseMappingOn = False
            self._desired["precise"] = (False, None, None, None)
            self._apply_live(lambda: self._session.set_precise_mapping(False))
            self._preciseStatus = "文件已更换，点启用重新构建"
        self.preciseChanged.emit()

    def _apply_precise_quantiles(self, src_q, tgt_q, method) -> None:
        # attach the (loaded or freshly-built) quantiles: cache for cheap reload-replay,
        # remember as the pending map (so a re-enable reuses it, no rebuild), and push
        # live (or no-op now + replay on load if not yet loaded). Turns the mapping on.
        self._desired["precise"] = (True, src_q, tgt_q, method)
        self._precisePendingQ = (src_q, tgt_q, method)
        self._apply_live(
            lambda: self._session.set_precise_mapping(True, src_q, tgt_q, method))
        self._preciseMappingOn = True

    @Slot(bool)
    def setPreciseMapping(self, on) -> None:
        """启用/停用 precise CDF F0 mapping. ON: if a map is already in hand (loaded from
        the dropdown, or previously built) apply it instantly — no rebuild, no engine
        required (replays on Start if not yet loaded); else build from the two wavs OFF
        the GUI thread (needs a loaded engine). OFF: drop the map live (it stays cached
        so re-enabling reuses it). Input-side carrier conditioning (faithful-carrier)."""
        on = bool(on)
        if not on:
            self._preciseMappingOn = False
            self._desired["precise"] = (False, None, None, None)
            self._preciseStatus = ""
            self._apply_live(lambda: self._session.set_precise_mapping(False))
            self.preciseChanged.emit()
            return
        if self._busy:                              # a build/load is in flight
            return
        if self._precisePendingQ is not None:       # loaded/built map -> instant, no build
            self._apply_precise_quantiles(*self._precisePendingQ)
            self._preciseStatus = "✓ 精确映射已启用"
            self.preciseChanged.emit()
            return
        if not (self._preciseVoiceWav and self._preciseTargetWav):
            self.errorOccurred.emit("请先选择两段 .wav，或从下拉加载一个已保存映射")
            return
        if (self._session.engine is None
                or self._session.state not in (SessionState.LOADED, SessionState.RUNNING)):
            self.errorOccurred.emit("请先 Start 加载模型，再构建精确映射")
            return
        self._set_busy(True)
        self._preciseStatus = "正在分析两段语音…"
        self.preciseChanged.emit()
        self._precise_thread = QThread(self)
        self._precise_worker = PreciseMapWorker(
            self._session, self._preciseVoiceWav, self._preciseTargetWav)
        self._precise_worker.moveToThread(self._precise_thread)
        self._precise_thread.started.connect(self._precise_worker.run)
        self._precise_worker.finished.connect(self._onPreciseDone)
        self._precise_worker.finished.connect(self._precise_thread.quit)
        self._precise_thread.finished.connect(self._precise_worker.deleteLater)
        self._precise_thread.finished.connect(self._precise_thread.deleteLater)
        self._precise_thread.finished.connect(self._clearPreciseThread)
        self._precise_thread.start()

    @Slot(bool, object, object, str, str)
    def _onPreciseDone(self, ok, src_q, tgt_q, method, err) -> None:    # GUI thread (queued)
        self._set_busy(False)
        if not ok:
            self._preciseMappingOn = False
            self._preciseStatus = "分析失败"
            self.preciseChanged.emit()
            self.errorOccurred.emit(f"精确映射失败：{err}")
            return
        self._apply_precise_quantiles(src_q, tgt_q, method)
        self._preciseStatus = "✓ 精确映射已启用"
        self.preciseChanged.emit()

    @Slot(str)
    def loadPreciseMap(self, file) -> None:
        """Load a saved map (dropdown): stage its quantiles + labels but do NOT enable —
        the user flips 启用 to apply (their choice). No build, no engine needed. If a
        mapping is already on, swap the live map to the loaded one."""
        if not file:
            return
        try:
            m = ps.load_precise_map(str(file))
        except Exception as exc:
            self.errorOccurred.emit(f"加载映射失败：{exc}")
            return
        self._precisePendingQ = (m["src_q"], m["tgt_q"], m["method"])
        self._preciseVoiceLabel = m["voice_name"] or "（已保存）"
        self._preciseTargetLabel = m["target_name"] or "（已保存）"
        # a loaded map carries no live wav paths -> f0-swap won't try to rebuild it.
        self._preciseVoiceWav = ""
        self._preciseTargetWav = ""
        if self._preciseMappingOn:                  # already running -> swap live now
            self._apply_precise_quantiles(*self._precisePendingQ)
            self._preciseStatus = f"✓ 已加载并启用：{m['name']}"
        else:
            self._preciseStatus = f"已加载「{m['name']}」，点启用生效"
        self.preciseChanged.emit()

    @Slot(str, result=bool)
    def savePreciseMap(self, name) -> bool:
        """Save the current map (loaded or built) under ``name`` so it shows in the
        dropdown next time — avoids re-picking wavs + rebuilding. Returns True on success."""
        if self._precisePendingQ is None:
            self.errorOccurred.emit("还没有可保存的映射：先构建或加载一个")
            return False
        src_q, tgt_q, method = self._precisePendingQ
        try:
            ps.save_precise_map(str(name), method, self._preciseVoiceLabel,
                                self._preciseTargetLabel, src_q, tgt_q, ps.PRECISE_DIR)
        except Exception as exc:
            self.errorOccurred.emit(f"保存映射失败：{exc}")
            return False
        self._preciseMaps = ps.list_precise_maps(ps.PRECISE_DIR)
        self._preciseStatus = "✓ 映射已保存"
        self.preciseChanged.emit()
        return True

    @Slot()
    def refreshPreciseMaps(self) -> None:
        self._preciseMaps = ps.list_precise_maps(ps.PRECISE_DIR)
        self.preciseChanged.emit()

    @Slot()
    def _clearPreciseThread(self) -> None:        # GUI thread (queued via thread.finished)
        self._precise_thread = None
        self._precise_worker = None

    def _reset_precise_for_model_change(self) -> None:
        # a model swap makes the chosen 模型原声 (and any built/loaded map) stale -> drop
        # the map + target + pending; keep the user's voice wav (reusable across models).
        # _desired["precise"] is already popped by _seed_desired_from_params.
        self._precisePendingQ = None
        self._preciseTargetWav = ""
        self._preciseTargetLabel = ""
        if not self._preciseMappingOn:
            self.preciseChanged.emit()
            return
        self._preciseMappingOn = False
        self._desired.pop("precise", None)        # fully reset (seed already popped it)
        self._preciseStatus = ""
        self._apply_live(lambda: self._session.set_precise_mapping(False))
        self.preciseChanged.emit()
