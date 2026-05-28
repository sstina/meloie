"""Realtime audio stream layer.

Stage 1 ``run_identity_stream`` and Stage 2 ``run_rvc_stream`` share
the same audio plumbing — the only difference is which worker thread
is started. The shared core lives in ``_run_stream``; mode-specific
metadata + the worker-startup callable are injected by the two public
entry points.

Hard rules baked into this module:

* Importing this module must NOT import sounddevice. The import is
  lazy and lives inside the functions that actually touch hardware.
* Audio callbacks must never block. They do put_nowait/get_nowait
  only; the worker thread does any meaningful work.
* The input device must be a physical microphone. The output device
  must be ``CABLE Input``. ``CABLE Output`` is refused for both sides
  unless the diagnostic override is set.
* No system / device defaults are changed.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Mapping, Optional

import numpy as np

from ..safety.guard import dbfs_peak, dbfs_rms
from ..safety.metrics import RuntimeMetrics
from .devices import (
    AudioDeviceInfo,
    FeedbackLoopRisk,
    is_probable_cable_output,
    iter_device_infos,
    normalize_device_name,
    select_device_by_substring,
)


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
        if self.mode not in ("identity", "rvc", "rvc_not_implemented"):
            raise ValueError(
                f"mode must be 'identity', 'rvc', or 'rvc_not_implemented', "
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
# Lazy enumeration helpers
# ---------------------------------------------------------------------------

def list_audio_devices() -> List[AudioDeviceInfo]:
    try:
        import sounddevice as sd  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover  - env-dependent
        raise RuntimeError(
            "sounddevice is not installed; cannot enumerate devices. "
            "Install it with `pip install sounddevice` when ready."
        ) from exc
    raw = sd.query_devices()
    return list(iter_device_infos(raw))


def describe_devices() -> str:
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
# Pure device-resolution helpers
# ---------------------------------------------------------------------------

def resolve_input_device(
    devices: Iterable[Mapping],
    substring: str,
    allow_virtual_cable: bool = False,
) -> AudioDeviceInfo:
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
    info = select_device_by_substring(devices, substring, kind="output")
    if is_probable_cable_output(info.name):
        raise FeedbackLoopRisk(
            f"refusing to render to {info.name!r}: that is the VB-CABLE "
            "capture endpoint. The app must render to 'CABLE Input' "
            "instead. See README §VB-CABLE routing rules."
        )
    return info


# ---------------------------------------------------------------------------
# Shared stream runner
# ---------------------------------------------------------------------------

def _print_metrics_line(metrics, in_q, out_q, mode_label: str) -> None:
    base = (
        f"[{metrics.elapsed_seconds:6.1f}s] "
        f"in={metrics.input_frames:>9d}f out={metrics.output_frames:>9d}f "
        f"qin={in_q.qsize():>3d} qout={out_q.qsize():>3d} "
        f"drop(in={metrics.input_queue_drops},out={metrics.output_queue_drops}) "
        f"under={metrics.output_underruns} "
        f"in_pk={metrics.input_peak_dbfs:6.1f}dB out_pk={metrics.output_peak_dbfs:6.1f}dB "
        f"fb={metrics.fallback_count} nan={metrics.nan_inf_scrub_count} "
        f"st(in={metrics.input_status_flag_count},out={metrics.output_status_flag_count})"
    )
    if mode_label == "rvc":
        base += (
            f" rvc_n={metrics.rvc_chunks_processed} "
            f"infer_last={metrics.rvc_inference_last_ms:5.0f}ms "
            f"mean={metrics.rvc_inference_mean_ms:5.0f}ms "
            f"max={metrics.rvc_inference_max_ms:5.0f}ms "
            f"rfb={metrics.rvc_fallback_count} "
            f"stale={metrics.rvc_stale_chunk_drops} "
            f"enq={metrics.rvc_output_blocks_enqueued} "
            f"odrop={metrics.rvc_output_blocks_dropped} "
            f"rs_mean={metrics.rvc_resample_mean_ms:4.1f}ms"
        )
    print(base, flush=True)


def _print_summary(
    metrics, input_info: AudioDeviceInfo, output_info: AudioDeviceInfo, mode_label: str
) -> None:
    print("\n--- final summary ---")
    print(f"mode                     = {mode_label}")
    print(f"elapsed_seconds          = {metrics.elapsed_seconds:.2f}")
    print(f"input_frames             = {metrics.input_frames}")
    print(f"output_frames            = {metrics.output_frames}")
    print(f"input_queue_drops        = {metrics.input_queue_drops}")
    print(f"output_queue_drops       = {metrics.output_queue_drops}")
    print(f"output_underruns         = {metrics.output_underruns}")
    print(f"fallback_count           = {metrics.fallback_count}")
    print(f"nan_inf_scrub_count      = {metrics.nan_inf_scrub_count}")
    print(f"input_status_flag_count  = {metrics.input_status_flag_count}")
    print(f"output_status_flag_count = {metrics.output_status_flag_count}")
    print(f"input_device             = [{input_info.index}] {input_info.name}")
    print(f"output_device            = [{output_info.index}] {output_info.name}")
    if mode_label == "rvc":
        print(f"rvc_chunks_processed     = {metrics.rvc_chunks_processed}")
        print(f"rvc_inference_count      = {metrics.rvc_inference_count}")
        print(f"rvc_inference_mean_ms    = {metrics.rvc_inference_mean_ms:.2f}")
        print(f"rvc_inference_median_ms  = {metrics.inference_median_ms():.2f}")
        print(f"rvc_inference_p95_ms     = {metrics.inference_percentile_ms(95.0):.2f}")
        print(f"rvc_inference_max_ms     = {metrics.rvc_inference_max_ms:.2f}")
        print(f"rvc_fallback_count       = {metrics.rvc_fallback_count}")
        print(f"rvc_stale_chunk_drops    = {metrics.rvc_stale_chunk_drops}")
        print(f"rvc_output_blocks_enqueued = {metrics.rvc_output_blocks_enqueued}")
        print(f"rvc_output_blocks_dropped  = {metrics.rvc_output_blocks_dropped}")
        print(f"rvc_resample_count       = {metrics.rvc_resample_count}")
        print(f"rvc_resample_mean_ms     = {metrics.rvc_resample_mean_ms:.2f}")
        print(f"max_input_queue_depth    = {metrics.max_input_queue_depth}")
        print(f"max_output_queue_depth   = {metrics.max_output_queue_depth}")
        print(f"startup_output_underruns = {metrics.startup_output_underruns}")
        print(f"steady_state_output_underruns = {metrics.steady_state_output_underruns}")
        print(f"chunk_ms                 = {metrics.rvc_chunk_ms}")
        print(f"crossfade_ms             = {metrics.rvc_crossfade_ms}")
        print(f"model                    = {metrics.rvc_model_basename or '(none)'}")
        print(f"index                    = {metrics.rvc_index_basename or '(none)'}")
        print(f"f0_method                = {metrics.rvc_f0_method}")
        print(f"index_rate               = {metrics.rvc_index_rate}")
        print(f"protect                  = {metrics.rvc_protect}")
        print(f"pitch_shift              = {metrics.rvc_pitch_shift}")


def _run_stream(
    config: AudioRuntimeConfig,
    worker_factory: Callable,
    mode_label: str,
    duration_seconds: Optional[float],
    allow_virtual_cable_input: bool,
    metrics_interval_seconds: float,
    intro_extra: Optional[List[str]] = None,
    input_queue_capacity: Optional[int] = None,
    output_queue_capacity: Optional[int] = None,
    output_prebuffer_blocks: int = 0,
) -> RuntimeMetrics:
    config.validate()
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0 or None for unbounded")

    try:
        import sounddevice as sd  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
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
        f"mode={mode_label}"
    )
    if intro_extra:
        for line in intro_extra:
            print(line)
    if allow_virtual_cable_input:
        print(
            "WARNING: --allow-virtual-cable-input is ON. Diagnostic mode "
            "only. Verify there is no feedback loop with Discord/OBS."
        )

    in_cap = int(input_queue_capacity or config.queue_blocks)
    out_cap = int(output_queue_capacity or config.queue_blocks)
    in_q: "queue.Queue" = queue.Queue(maxsize=in_cap)
    out_q: "queue.Queue" = queue.Queue(maxsize=out_cap)
    metrics = RuntimeMetrics()
    stop_event = threading.Event()

    in_q_ms = in_cap * config.block_size * 1000.0 / max(1, config.sample_rate)
    out_q_ms = out_cap * config.block_size * 1000.0 / max(1, config.sample_rate)
    print(
        f"queues: input {in_cap} blocks (~{in_q_ms:.0f} ms)  "
        f"output {out_cap} blocks (~{out_q_ms:.0f} ms)"
    )

    # Optional silence prebuffer for the output stream so the first ~N
    # output blocks have something to play while the worker spins up
    # / finishes its first chunk's inference. Audible as "extra
    # latency" but masks startup underruns.
    if output_prebuffer_blocks > 0:
        silence_block = np.zeros(config.block_size, dtype=np.float32)
        actually_buffered = 0
        for _ in range(output_prebuffer_blocks):
            try:
                out_q.put_nowait(silence_block.copy())
                actually_buffered += 1
            except queue.Full:
                break
        pre_ms = actually_buffered * config.block_size * 1000.0 / max(1, config.sample_rate)
        print(
            f"prebuffered {actually_buffered} silence blocks "
            f"(~{pre_ms:.0f} ms of safety margin before first real audio)"
        )

    def in_callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            metrics.input_status_flag_count += 1
        block = indata[:, 0].astype(np.float32, copy=True)
        metrics.input_peak_dbfs = dbfs_peak(block)
        metrics.input_rms_dbfs = dbfs_rms(block)
        metrics.input_frames += int(frames)
        try:
            in_q.put_nowait(block)
        except queue.Full:
            metrics.input_queue_drops += 1

    def out_callback(outdata, frames, time_info, status):  # noqa: ANN001
        if status:
            metrics.output_status_flag_count += 1
        try:
            block = out_q.get_nowait()
        except queue.Empty:
            outdata.fill(0.0)
            metrics.output_underruns += 1
            # Bin into startup vs steady-state by checking whether the
            # worker has yet emitted any real audio.
            if metrics.first_real_output_seen:
                metrics.steady_state_output_underruns += 1
            else:
                metrics.startup_output_underruns += 1
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

    worker_thread = worker_factory(in_q, out_q, metrics, stop_event, _SHUTDOWN_SENTINEL)

    common = dict(
        samplerate=config.sample_rate,
        channels=config.channels,
        blocksize=config.block_size,
        dtype="float32",
        latency="low",
    )

    start_wall = time.monotonic()
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
                # Track high-water marks of queue depths.
                qi = in_q.qsize()
                qo = out_q.qsize()
                if qi > metrics.max_input_queue_depth:
                    metrics.max_input_queue_depth = qi
                if qo > metrics.max_output_queue_depth:
                    metrics.max_output_queue_depth = qo
                if duration_seconds is not None and metrics.elapsed_seconds >= duration_seconds:
                    break
                if now - last_print >= metrics_interval_seconds:
                    _print_metrics_line(metrics, in_q, out_q, mode_label)
                    last_print = now
                time.sleep(0.05)
    except KeyboardInterrupt:
        metrics.notes.append("stopped_by_keyboard_interrupt")
        print("\nKeyboardInterrupt — stopping.", file=sys.stderr, flush=True)
    finally:
        stop_event.set()
        try:
            in_q.put_nowait(_SHUTDOWN_SENTINEL)
        except queue.Full:
            pass
        worker_thread.join(timeout=2.0)
        metrics.elapsed_seconds = time.monotonic() - start_wall
        _print_metrics_line(metrics, in_q, out_q, mode_label)
        _print_summary(metrics, input_info, output_info, mode_label)

    return metrics


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_identity_stream(
    config: AudioRuntimeConfig,
    duration_seconds: Optional[float] = None,
    allow_virtual_cable_input: bool = False,
    metrics_interval_seconds: float = 1.0,
) -> RuntimeMetrics:
    """Open the realtime identity audio loop and run it."""

    def factory(in_q, out_q, metrics, stop_event, sentinel):
        from ..engine.worker import WorkerConfig, WorkerMode, worker_loop
        wc = WorkerConfig(mode=WorkerMode.IDENTITY)
        t = threading.Thread(
            target=worker_loop,
            args=(wc, in_q, out_q, metrics, stop_event, sentinel),
            name="identity-worker",
            daemon=True,
        )
        t.start()
        return t

    return _run_stream(
        config, factory, "identity",
        duration_seconds, allow_virtual_cable_input, metrics_interval_seconds,
    )


def queue_blocks_from_ms(
    queue_ms: float, block_size: int, sample_rate: int, minimum: int = 64
) -> int:
    """Convert a queue capacity in ms to a number of blocks.

    Caps to at least ``minimum`` blocks so identity-era defaults still
    work even with very small ``queue_ms`` values.
    """
    if block_size <= 0 or sample_rate <= 0:
        raise ValueError("block_size and sample_rate must be > 0")
    if queue_ms <= 0:
        return int(minimum)
    block_ms = float(block_size) * 1000.0 / float(sample_rate)
    n = int(round(float(queue_ms) / block_ms))
    return max(int(minimum), n)


def run_rvc_stream(
    config: AudioRuntimeConfig,
    engine,
    chunk_ms: float,
    crossfade_ms: float = 0.0,
    duration_seconds: Optional[float] = None,
    allow_virtual_cable_input: bool = False,
    metrics_interval_seconds: float = 1.0,
    rvc_queue_ms: float = 6000.0,
    rvc_prebuffer_ms: Optional[float] = None,
    drop_stale_input: bool = True,
    context_ms: float = 0.0,
) -> RuntimeMetrics:
    """Open the realtime RVC chunk loop and run it.

    The caller owns ``engine`` and must have already called
    ``engine.load()`` (so that any DependencyMissing / ModelLoad
    failure happens *before* we touch any audio device).
    """
    if engine is None:
        raise ValueError("engine must be provided for run_rvc_stream")
    if not getattr(engine, "is_loaded", False):
        raise RuntimeError(
            "engine.load() must be called before run_rvc_stream (so model "
            "/ dependency errors fail before any audio device opens)."
        )
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be > 0")
    if crossfade_ms < 0:
        raise ValueError("crossfade_ms must be >= 0")
    if context_ms < 0:
        raise ValueError("context_ms must be >= 0")

    chunk_size = max(1, int(round(chunk_ms / 1000.0 * config.sample_rate)))
    crossfade_size = max(0, int(round(crossfade_ms / 1000.0 * config.sample_rate)))
    context_size = max(0, int(round(context_ms / 1000.0 * config.sample_rate)))
    output_block_size = int(config.block_size)

    # RVC mode needs much larger queues than identity. The identity-era
    # default of 64 blocks (~640 ms at 48 kHz / 480-frame blocks) is too
    # small to hold a single ~1 s chunk's worth of output blocks, so
    # post-inference audio gets discarded.
    rvc_queue_blocks = queue_blocks_from_ms(
        float(rvc_queue_ms), config.block_size, config.sample_rate,
        minimum=config.queue_blocks,
    )
    # Prebuffer: silence blocks placed in out_q before opening the audio
    # stream. Hides the first-chunk inference time at the cost of that
    # much added latency. Default = 2 × chunk_ms (one chunk to accumulate,
    # roughly one chunk to infer).
    prebuffer_ms = (
        float(rvc_prebuffer_ms)
        if rvc_prebuffer_ms is not None
        else float(chunk_ms) * 2.0
    )
    prebuffer_blocks = max(
        0, int(round(prebuffer_ms * config.sample_rate / 1000.0 / config.block_size))
    )

    model_basename = os.path.basename(engine.config.model_path or "")
    index_basename = (
        os.path.basename(engine.config.index_path) if engine.config.index_path else ""
    )

    intro_extra = [
        f"chunk_ms={chunk_ms:.1f} ({chunk_size} samples)  "
        f"crossfade_ms={crossfade_ms:.1f} ({crossfade_size} samples)",
        f"model={model_basename or '(none)'}  "
        f"index={index_basename or '(none)'}  "
        f"backend={engine.backend_name}  "
        f"device={engine.resolved_device or '(unknown)'}"
        + (f" / {engine.cuda_device_name}" if engine.cuda_device_name else ""),
        f"resample_sr={engine.config.resample_sr}  "
        f"stream_sr={config.sample_rate}  "
        "(SR mismatch -> worker resample_audio, sinc-polyphase if scipy)",
        f"rvc_queue_blocks={rvc_queue_blocks}  "
        f"rvc_prebuffer_blocks={prebuffer_blocks}  "
        f"drop_stale_input={drop_stale_input}",
        f"context_ms={context_ms:.1f} ({context_size} samples) "
        f"-- input-left-context fed to model; output trimmed proportionally "
        f"so emit duration == chunk_ms (no timeline drift)",
        f"f0_method={engine.config.f0_method}  "
        f"index_rate={engine.config.index_rate}  "
        f"protect={engine.config.protect}  "
        f"pitch_shift={engine.config.pitch_shift}  "
        f"filter_radius={engine.config.filter_radius}  "
        f"rms_mix_rate={engine.config.rms_mix_rate}",
        "model-faithful posture: no post-model EQ / limiter / normalize. "
        "Identity fallback only on hard backend error. "
        "Per-chunk inference = the structural quality ceiling vs offline "
        "whole-file (no input-overlap; deferred).",
    ]
    if crossfade_size == 0:
        intro_extra.append(
            "NOTE: stitched crossfade is OFF (model-faithful default). "
            "The OUTPUT timeline is therefore not shifted; chunk boundaries "
            "may produce occasional clicks on plosive onsets. Set "
            "--crossfade-ms >0 only to reproduce legacy stitched-blend "
            "behaviour."
        )

    def factory(in_q, out_q, metrics, stop_event, sentinel):
        from ..engine.worker import rvc_worker_loop
        # Seed session metadata into metrics so the summary has it even
        # if the run is short.
        metrics.rvc_chunk_ms = float(chunk_ms)
        metrics.rvc_crossfade_ms = float(crossfade_ms)
        metrics.rvc_model_basename = model_basename
        metrics.rvc_index_basename = index_basename
        metrics.rvc_f0_method = engine.config.f0_method
        metrics.rvc_index_rate = float(engine.config.index_rate)
        metrics.rvc_protect = float(engine.config.protect)
        metrics.rvc_pitch_shift = int(engine.config.pitch_shift)

        t = threading.Thread(
            target=rvc_worker_loop,
            args=(engine, in_q, out_q, metrics, stop_event, sentinel),
            kwargs={
                "sample_rate": int(config.sample_rate),
                "chunk_size": int(chunk_size),
                "output_block_size": int(output_block_size),
                "crossfade_size": int(crossfade_size),
                "drop_stale_input": bool(drop_stale_input),
                "context_size": int(context_size),
            },
            name="rvc-worker",
            daemon=True,
        )
        t.start()
        return t

    return _run_stream(
        config, factory, "rvc",
        duration_seconds, allow_virtual_cable_input, metrics_interval_seconds,
        intro_extra=intro_extra,
        input_queue_capacity=rvc_queue_blocks,
        output_queue_capacity=rvc_queue_blocks,
        output_prebuffer_blocks=prebuffer_blocks,
    )
