"""Realtime audio stream layer (Stage 1 identity passthrough).

Architecture:

    physical microphone
       -> sounddevice InputStream (callback, non-blocking)
       -> in_queue (bounded)
       -> worker thread (identity in Stage 1, RVC in Stage 2)
       -> out_queue (bounded)
       -> sounddevice OutputStream (callback, non-blocking)
       -> CABLE Input

Hard rules baked into this module:

  * Importing this module must NOT import sounddevice. The import is
    lazy and lives inside the functions that actually touch hardware.
  * Audio callbacks must never block. They do put_nowait/get_nowait
    only; the worker thread does any meaningful work.
  * The input device must be a physical microphone. The output device
    must be ``CABLE Input``. Selecting ``CABLE Output`` for either side
    is refused (it would feed the cable's capture endpoint back to its
    render endpoint via Discord/OBS — a feedback loop).
  * No system / device defaults are changed; no settings are written.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional

import numpy as np

from ..safety.guard import dbfs_peak, dbfs_rms, scrub_nan_inf
from ..safety.metrics import RuntimeMetrics
from .devices import (
    AudioDeviceInfo,
    FeedbackLoopRisk,
    is_probable_cable_output,
    iter_device_infos,
    normalize_device_name,
    select_device_by_substring,
)


# Sentinel placed on the input queue to tell the worker thread to exit.
_SHUTDOWN_SENTINEL = object()


@dataclass(frozen=True)
class AudioRuntimeConfig:
    """Runtime config for the realtime audio loop.

    Fields map 1-to-1 to keys in ``config/runtime.example.json``.
    """

    sample_rate: int = 48000
    block_size: int = 480           # 10 ms at 48 kHz
    channels: int = 1
    input_device_substring: str = "Microphone"
    output_device_substring: str = "CABLE Input"
    queue_blocks: int = 64
    mode: str = "identity"

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if self.queue_blocks <= 0:
            raise ValueError("queue_blocks must be > 0")
        if self.mode not in ("identity", "rvc_not_implemented"):
            raise ValueError(
                f"mode must be 'identity' or 'rvc_not_implemented', "
                f"got {self.mode!r}"
            )


@dataclass
class StreamStatusSnapshot:
    """Lightweight status snapshot returned alongside metrics."""

    running: bool = False
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lazy enumeration helpers (sounddevice imported only when called)
# ---------------------------------------------------------------------------

def list_audio_devices() -> List[AudioDeviceInfo]:
    """Return wrapped ``AudioDeviceInfo`` records for the host.

    Lazy-imports ``sounddevice``; raises ``RuntimeError`` with a clear
    message if it is not installed.
    """
    try:
        import sounddevice as sd  # noqa: WPS433  (deliberately lazy)
    except ImportError as exc:  # pragma: no cover  - env-dependent
        raise RuntimeError(
            "sounddevice is not installed; cannot enumerate devices. "
            "Install it with `pip install sounddevice` when ready."
        ) from exc

    raw = sd.query_devices()
    return list(iter_device_infos(raw))


def describe_devices() -> str:
    """Human-readable device listing. Lazy-imports ``sounddevice``."""
    infos = list_audio_devices()
    lines = ["index  in  out  name"]
    for info in infos:
        lines.append(
            f"{info.index:>5}  "
            f"{info.max_input_channels:>2}  "
            f"{info.max_output_channels:>3}  "
            f"{info.name}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Device-resolution helpers (pure; tests use these with fake device lists)
# ---------------------------------------------------------------------------

def resolve_input_device(
    devices: Iterable[Mapping],
    substring: str,
    allow_virtual_cable: bool = False,
) -> AudioDeviceInfo:
    """Pick an input device by substring.

    By default refuses ``CABLE Output`` (the VB-CABLE capture endpoint)
    as the app's microphone — using it would loop Discord's outbound
    audio back into the app. ``allow_virtual_cable=True`` is the
    diagnostic override used only for offline through-cable validation.
    """
    if allow_virtual_cable:
        needle = normalize_device_name(substring)
        if not needle:
            raise ValueError("substring must be a non-empty string")
        for info in iter_device_infos(devices):
            if needle in normalize_device_name(info.name) and info.is_input_capable:
                return info
        raise LookupError(
            f"no input device matched substring {substring!r} "
            "(diagnostic override on)"
        )
    return select_device_by_substring(devices, substring, kind="input")


def resolve_output_device(
    devices: Iterable[Mapping],
    substring: str,
) -> AudioDeviceInfo:
    """Pick an output device by substring, refusing ``CABLE Output``.

    The app must render to ``CABLE Input`` (the cable's render
    endpoint), not ``CABLE Output`` (the cable's capture endpoint).
    """
    info = select_device_by_substring(devices, substring, kind="output")
    if is_probable_cable_output(info.name):
        raise FeedbackLoopRisk(
            f"refusing to render to {info.name!r}: that is the VB-CABLE "
            "capture endpoint. The app must render to 'CABLE Input' "
            "instead. See README §VB-CABLE routing rules."
        )
    return info


# ---------------------------------------------------------------------------
# Realtime identity stream
# ---------------------------------------------------------------------------

def _print_metrics_line(
    metrics: RuntimeMetrics,
    in_q: "queue.Queue",
    out_q: "queue.Queue",
) -> None:
    print(
        f"[{metrics.elapsed_seconds:6.1f}s] "
        f"in={metrics.input_frames:>9d}f "
        f"out={metrics.output_frames:>9d}f "
        f"qin={in_q.qsize():>3d} qout={out_q.qsize():>3d} "
        f"drop(in={metrics.input_queue_drops},out={metrics.output_queue_drops}) "
        f"under={metrics.output_underruns} "
        f"in_pk={metrics.input_peak_dbfs:6.1f}dB "
        f"in_rms={metrics.input_rms_dbfs:6.1f}dB "
        f"out_pk={metrics.output_peak_dbfs:6.1f}dB "
        f"out_rms={metrics.output_rms_dbfs:6.1f}dB "
        f"fb={metrics.fallback_count} "
        f"nan={metrics.nan_inf_scrub_count} "
        f"status(in={metrics.input_status_flag_count},out={metrics.output_status_flag_count})",
        flush=True,
    )


def run_identity_stream(
    config: AudioRuntimeConfig,
    duration_seconds: Optional[float] = None,
    allow_virtual_cable_input: bool = False,
    metrics_interval_seconds: float = 1.0,
) -> RuntimeMetrics:
    """Open the realtime identity audio loop and run it.

    Blocks until ``duration_seconds`` elapses, the user presses Ctrl+C,
    or an unrecoverable error occurs. Returns the final
    :class:`RuntimeMetrics` snapshot so the CLI (or a future test
    harness) can persist or print it.
    """
    config.validate()
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0 or None for unbounded")

    # Lazy hardware imports — only happen when we actually start audio.
    try:
        import sounddevice as sd  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover  - env-dependent
        raise RuntimeError(
            "sounddevice is not installed; cannot run realtime stream. "
            "Install with `pip install sounddevice`."
        ) from exc

    raw_devices = list(sd.query_devices())

    input_info = resolve_input_device(
        raw_devices,
        config.input_device_substring,
        allow_virtual_cable=allow_virtual_cable_input,
    )
    output_info = resolve_output_device(
        raw_devices, config.output_device_substring
    )

    print(
        f"input device  [{input_info.index:>3}]: {input_info.name}\n"
        f"output device [{output_info.index:>3}]: {output_info.name}\n"
        f"sample_rate={config.sample_rate} block_size={config.block_size} "
        f"channels={config.channels} queue_blocks={config.queue_blocks} "
        f"mode={config.mode}"
    )
    if allow_virtual_cable_input:
        print(
            "WARNING: --allow-virtual-cable-input is ON. Diagnostic mode "
            "only. Verify there is no feedback loop with Discord/OBS."
        )

    in_q: "queue.Queue" = queue.Queue(maxsize=config.queue_blocks)
    out_q: "queue.Queue" = queue.Queue(maxsize=config.queue_blocks)
    metrics = RuntimeMetrics()
    stop_event = threading.Event()

    # ---- Input callback (audio thread) -------------------------------
    def in_callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            metrics.input_status_flag_count += 1
        # indata is (frames, channels); take mono and copy (the buffer
        # is reused by PortAudio).
        block = indata[:, 0].astype(np.float32, copy=True)
        # Cheap level read for monitoring; np.max(abs) on ~480 samples
        # is negligible vs the audio thread budget.
        metrics.input_peak_dbfs = dbfs_peak(block)
        metrics.input_rms_dbfs = dbfs_rms(block)
        metrics.input_frames += int(frames)
        try:
            in_q.put_nowait(block)
        except queue.Full:
            metrics.input_queue_drops += 1

    # ---- Output callback (audio thread) ------------------------------
    def out_callback(outdata, frames, time_info, status):  # noqa: ANN001
        if status:
            metrics.output_status_flag_count += 1
        try:
            block = out_q.get_nowait()
        except queue.Empty:
            outdata.fill(0.0)
            metrics.output_underruns += 1
            metrics.output_peak_dbfs = dbfs_peak(np.zeros(frames, dtype=np.float32))
            metrics.output_rms_dbfs = dbfs_rms(np.zeros(frames, dtype=np.float32))
            metrics.output_frames += int(frames)
            return

        n = min(int(block.shape[0]), int(frames))
        outdata[:n, 0] = block[:n]
        if n < frames:
            outdata[n:, 0] = 0.0
        metrics.output_peak_dbfs = dbfs_peak(block[:n])
        metrics.output_rms_dbfs = dbfs_rms(block[:n])
        metrics.output_frames += int(frames)

    # ---- Worker thread -----------------------------------------------
    from ..engine.worker import WorkerConfig, WorkerMode, worker_loop

    worker_config = WorkerConfig(mode=WorkerMode.IDENTITY)
    worker_thread = threading.Thread(
        target=worker_loop,
        args=(worker_config, in_q, out_q, metrics, stop_event, _SHUTDOWN_SENTINEL),
        name="identity-worker",
        daemon=True,
    )

    # Try latency='low' first; fall back to default on TypeError-shaped
    # backend issues. ``with`` ensures clean teardown even on Ctrl+C.
    common = dict(
        samplerate=config.sample_rate,
        channels=config.channels,
        blocksize=config.block_size,
        dtype="float32",
        latency="low",
    )

    start_wall = time.monotonic()
    worker_thread.start()
    try:
        with sd.InputStream(
            device=input_info.index, callback=in_callback, **common
        ), sd.OutputStream(
            device=output_info.index, callback=out_callback, **common
        ):
            last_print = start_wall
            print("running. Ctrl+C to stop.", flush=True)
            while True:
                now = time.monotonic()
                metrics.elapsed_seconds = now - start_wall
                if duration_seconds is not None and metrics.elapsed_seconds >= duration_seconds:
                    break
                if now - last_print >= metrics_interval_seconds:
                    _print_metrics_line(metrics, in_q, out_q)
                    last_print = now
                time.sleep(0.05)
    except KeyboardInterrupt:
        metrics.notes.append("stopped_by_keyboard_interrupt")
        print("\nKeyboardInterrupt — stopping.", file=sys.stderr, flush=True)
    finally:
        stop_event.set()
        # Push sentinel so the worker can wake from a get() blocking on
        # an empty queue. Best-effort — if the queue is full, drop it;
        # stop_event will cause the worker to exit at the next poll.
        try:
            in_q.put_nowait(_SHUTDOWN_SENTINEL)
        except queue.Full:
            pass
        worker_thread.join(timeout=2.0)
        metrics.elapsed_seconds = time.monotonic() - start_wall
        _print_metrics_line(metrics, in_q, out_q)
        print(
            "\n--- final summary ---\n"
            f"elapsed_seconds         = {metrics.elapsed_seconds:.2f}\n"
            f"input_frames            = {metrics.input_frames}\n"
            f"output_frames           = {metrics.output_frames}\n"
            f"input_queue_drops       = {metrics.input_queue_drops}\n"
            f"output_queue_drops      = {metrics.output_queue_drops}\n"
            f"output_underruns        = {metrics.output_underruns}\n"
            f"fallback_count          = {metrics.fallback_count}\n"
            f"nan_inf_scrub_count     = {metrics.nan_inf_scrub_count}\n"
            f"input_status_flag_count = {metrics.input_status_flag_count}\n"
            f"output_status_flag_count= {metrics.output_status_flag_count}\n"
            f"input_device            = [{input_info.index}] {input_info.name}\n"
            f"output_device           = [{output_info.index}] {output_info.name}",
            flush=True,
        )

    return metrics
