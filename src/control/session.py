"""``RealtimeSession`` — the Qt-free facade a GUI binds to.

It owns the engine, the realtime stream (run in a background daemon thread), and
the live ``RuntimeMetrics``, exposing:

* lifecycle: :meth:`load`, :meth:`start` (non-blocking), :meth:`stop`,
  :meth:`reload`, :meth:`close`;
* live INPUT-side setters (delegated to the engine, which validates) — pitch,
  protect, index_rate / index_path, autotune, auto-pitch, formant, denoise,
  silence gate, f0-method (rmvpe<->fcpe). **No output-shaping setter exists**
  (faithful-carrier contract);
* observation: :attr:`state`, :meth:`metrics_snapshot`, :attr:`last_error`.

Dependency-injectable for tests: pass an ``engine_factory`` and/or
``stream_runner`` to drive it with fakes (no audio devices / torch). The real
defaults (``StreamingRvcEngine`` / ``run_streaming_stream``) are imported lazily
so importing this module is cheap and side-effect free.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any, Callable, Dict, Optional

from ..audio.streams import MonitorState
from ..safety.metrics import RuntimeMetrics


class SessionError(RuntimeError):
    """Invalid session operation (e.g. start() before load())."""


class SessionState(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class RealtimeSession:
    """Lifecycle + live-control facade over ``StreamingRvcEngine`` + the
    realtime stream. Methods are called from the control thread (GUI / tests);
    the stream itself runs in a background daemon thread."""

    def __init__(
        self,
        *,
        engine_factory: Optional[Callable[[Any], Any]] = None,
        stream_runner: Optional[Callable[..., RuntimeMetrics]] = None,
        on_state_change: Optional[Callable[[SessionState, SessionState], None]] = None,
    ) -> None:
        self._engine_factory = engine_factory      # default: StreamingRvcEngine (lazy)
        self._stream_runner = stream_runner        # default: run_streaming_stream (lazy)
        self._on_state_change = on_state_change

        self._state = SessionState.IDLE
        self._state_lock = threading.RLock()
        self._engine = None
        self._engine_cfg = None
        self._metrics: Optional[RuntimeMetrics] = None
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
        self._monitor_state: Optional[MonitorState] = None

    # ---------------------------------------------------------------- state
    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def engine(self):
        """The underlying loaded engine (or None). For diagnostics/advanced use."""
        return self._engine

    @property
    def num_speakers(self) -> int:
        """Trained speaker count of the loaded model (1 if not loaded)."""
        return int(getattr(self._engine, "num_speakers", 1)) if self._engine is not None else 1

    def _set_state(self, new: SessionState) -> None:
        with self._state_lock:
            old = self._state
            if old == new:
                return
            self._state = new
        self._fire(old, new)

    def _transition_if(self, allowed, new: SessionState) -> bool:
        with self._state_lock:
            old = self._state
            if old not in allowed:
                return False
            self._state = new
        if old != new:
            self._fire(old, new)
        return True

    def _fire(self, old: SessionState, new: SessionState) -> None:
        if self._on_state_change is not None:
            try:
                self._on_state_change(old, new)
            except Exception:  # a UI callback must never break the session
                pass

    # ------------------------------------------------------------ lifecycle
    def load(self, engine_cfg) -> None:
        """Build + load the engine (blocking; ~tens of seconds cold). Allowed
        from IDLE / LOADED / ERROR. Call from a worker thread if you must keep a
        UI responsive — this method itself is synchronous."""
        if not self._transition_if(
            {SessionState.IDLE, SessionState.LOADED, SessionState.ERROR}, SessionState.LOADING
        ):
            raise SessionError(f"cannot load() from state {self._state.value}")
        try:
            factory = self._engine_factory
            if factory is None:
                from ..engine.streaming_engine import StreamingRvcEngine
                factory = StreamingRvcEngine
            engine = factory(engine_cfg)
            engine.load()
        except BaseException as exc:  # noqa: BLE001 - surface, don't crash
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._engine = None
            self._set_state(SessionState.ERROR)
            raise
        self._engine = engine
        self._engine_cfg = engine_cfg
        self._last_error = None
        self._set_state(SessionState.LOADED)

    def start(
        self,
        audio_cfg,
        *,
        duration_seconds: Optional[float] = None,
        allow_virtual_cable_input: bool = False,
        rvc_queue_ms: float = 6000.0,
        rvc_prebuffer_ms: Optional[float] = None,
        drop_stale_input: bool = True,
        monitor_enabled: bool = False,
    ) -> None:
        """Start the realtime stream in a background thread (non-blocking).
        Allowed only from LOADED. ``monitor_enabled`` is the initial state of the
        headphone-monitor sink (the device comes from ``audio_cfg``); flip it live
        with :meth:`set_monitor_enabled`."""
        if self._engine is None:
            raise SessionError("no engine loaded; call load() first")
        if not self._transition_if({SessionState.LOADED}, SessionState.RUNNING):
            raise SessionError(f"cannot start() from state {self._state.value}")

        runner = self._stream_runner
        if runner is None:
            from ..audio.streaming_stream import run_streaming_stream
            runner = run_streaming_stream

        self._metrics = RuntimeMetrics()
        self._stop_event = threading.Event()
        self._monitor_state = MonitorState(enabled=bool(monitor_enabled))
        runner_kwargs = dict(
            duration_seconds=duration_seconds,
            allow_virtual_cable_input=allow_virtual_cable_input,
            rvc_queue_ms=rvc_queue_ms,
            rvc_prebuffer_ms=rvc_prebuffer_ms,
            drop_stale_input=drop_stale_input,
            monitor_state=self._monitor_state,
        )
        self._thread = threading.Thread(
            target=self._run, args=(audio_cfg, runner, runner_kwargs),
            name="rvc-session-stream", daemon=True,
        )
        self._thread.start()

    def _run(self, audio_cfg, runner, runner_kwargs) -> None:
        try:
            runner(
                audio_cfg, self._engine,
                stop_event=self._stop_event, metrics=self._metrics,
                print_metrics=False, **runner_kwargs,
            )
        except BaseException as exc:  # noqa: BLE001 - report, never crash the thread silently
            if self._stop_event is not None and self._stop_event.is_set():
                # A stop was requested concurrently; an error raised while tearing
                # the stream down is not a real failure -> settle back to LOADED,
                # not ERROR (so the GUI doesn't see a bogus ERROR after a stop).
                self._transition_if({SessionState.RUNNING, SessionState.STOPPING}, SessionState.LOADED)
                return
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._set_state(SessionState.ERROR)
            return
        # normal return (external stop or duration elapsed)
        self._transition_if({SessionState.RUNNING, SessionState.STOPPING}, SessionState.LOADED)

    def stop(self, *, join_timeout: float = 5.0) -> None:
        """Signal the stream to stop and join it. Safe to call when not running
        (no-op). After a clean stop the session is back in LOADED."""
        if self._state not in (SessionState.RUNNING, SessionState.STOPPING):
            return
        self._transition_if({SessionState.RUNNING}, SessionState.STOPPING)
        if self._stop_event is not None:
            self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=join_timeout)
            if t.is_alive():
                self._last_error = "stream thread did not stop within timeout"
                self._set_state(SessionState.ERROR)
        self._thread = None

    def reload(self, new_engine_cfg) -> None:
        """Stop (if running), drop the engine, and load a new one."""
        if self._state in (SessionState.RUNNING, SessionState.STOPPING):
            self.stop()
        self._engine = None
        self.load(new_engine_cfg)

    def close(self) -> None:
        """Stop the stream and release the engine."""
        self.stop()
        self._engine = None
        self._set_state(SessionState.IDLE)

    # --------------------------------------------------------- live setters
    # All INPUT-side (faithful-carrier contract). Valid in LOADED or RUNNING.
    # Validation lives in the engine; the session only gates on state.
    def _require_active(self) -> None:
        if self._engine is None or self._state not in (SessionState.LOADED, SessionState.RUNNING):
            raise SessionError(
                f"no active engine (state={self._state.value}); call load()/start() first"
            )

    def set_pitch_shift(self, semitones) -> None:
        self._require_active(); self._engine.set_pitch_shift(semitones)

    def set_protect(self, p) -> None:
        self._require_active(); self._engine.set_protect(p)

    def set_index_rate(self, r) -> None:
        self._require_active(); self._engine.set_index_rate(r)

    def set_index_path(self, path) -> None:
        self._require_active(); self._engine.set_index_path(path)

    def set_autotune(self, on, strength=None) -> None:
        self._require_active(); self._engine.set_autotune(on, strength)

    def set_auto_pitch(self, on, threshold=None) -> None:
        self._require_active(); self._engine.set_auto_pitch(on, threshold)

    def set_auto_center(self, on, target_hz=None, tau_s=None) -> None:
        self._require_active(); self._engine.set_auto_center(on, target_hz, tau_s)

    def set_formant(self, on, timbre=None, qfrency=None) -> None:
        self._require_active(); self._engine.set_formant(on, timbre, qfrency)

    def set_denoise(self, on, strength=None, nonstationary=None) -> None:
        self._require_active(); self._engine.set_denoise(on, strength, nonstationary)

    def set_silence_gate(self, dbfs, hangover_ms=None) -> None:
        self._require_active(); self._engine.set_silence_gate(dbfs, hangover_ms)

    def set_sid(self, sid) -> None:
        self._require_active(); self._engine.set_sid(sid)

    def set_f0_method(self, method) -> None:
        self._require_active(); self._engine.set_f0_method(method)

    # ---------------------------------------------------- monitor (routing only)
    # NOT an input-side engine setter and NOT output shaping: it gates a parallel
    # sink that plays the SAME converted samples to headphones. Faithful-carrier
    # is untouched (no reshaping). Valid anytime; no-op if no monitor stream.
    def set_monitor_enabled(self, on: bool) -> None:
        if self._monitor_state is not None:
            self._monitor_state.enabled = bool(on)

    @property
    def monitor_enabled(self) -> bool:
        return bool(self._monitor_state.enabled) if self._monitor_state is not None else False

    # ----------------------------------------------------------- observation
    def metrics_snapshot(self) -> Dict[str, Any]:
        """A plain-dict snapshot of the live metrics (``{}`` before the first
        ``start()``). Safe to poll from a UI timer."""
        m = self._metrics
        if m is None:
            return {}
        try:
            return m.to_dict()
        except Exception:  # extremely unlikely race on a concurrently-mutated list
            return {}
