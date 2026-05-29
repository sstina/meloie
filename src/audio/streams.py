"""Realtime audio stream layer for the RVC voice changer.

One job: open ``system default mic -> in_queue -> RVC worker -> out_queue
-> CABLE Input`` and run it. There is exactly one realtime path.

Hard rules baked into this module:

* Importing this module must NOT import sounddevice. The import is lazy
  and lives inside the functions that actually touch hardware.
* Audio callbacks must never block. They do put_nowait/get_nowait only;
  the worker thread does all meaningful work.
* The input device is the user's microphone — by default the Windows
  default recording device ("系统默认 mic"); an explicit substring may
  override it. The output device must be ``CABLE Input``. ``CABLE Output``
  is refused for both sides (feedback-loop guard).
* No system / device defaults are changed.
"""

from __future__ import annotations

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
    select_default_input_device,
    select_device_by_substring,
)


_SHUTDOWN_SENTINEL = object()


@dataclass(frozen=True)
class AudioRuntimeConfig:
    """Runtime config for the realtime audio loop.

    ``input_device_substring=None`` (the default) means "follow the
    Windows default recording device". Set a substring to pin a specific
    microphone instead.
    """

    sample_rate: int = 48000
    block_size: int = 480           # 10 ms at 48 kHz
    channels: int = 1               # mono only — the whole route is mono
    input_device_substring: Optional[str] = None
    output_device_substring: str = "CABLE Input"
    queue_blocks: int = 64

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.channels != 1:
            # The pipeline is mono end-to-end: input reads channel 0, the
            # worker/model/resampler/slice are all mono, output writes
            # channel 0. Stereo would silently drop the right input channel
            # and leave the right output channel uninitialised.
            raise ValueError("channels must be 1 (the route is mono)")
        if self.queue_blocks <= 0:
            raise ValueError("queue_blocks must be > 0")


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
    try:
        import sounddevice as sd  # noqa: WPS433
        default_in, default_out = sd.default.device
    except Exception:
        default_in, default_out = (None, None)
    infos = list_audio_devices()
    lines = ["index  in  out  default  name"]
    for info in infos:
        marker = ""
        if info.index == default_in:
            marker += "I"
        if info.index == default_out:
            marker += "O"
        lines.append(
            f"{info.index:>5}  "
            f"{info.max_input_channels:>2}  "
            f"{info.max_output_channels:>3}  "
            f"{marker:>7}  "
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
# Metrics printing
# ---------------------------------------------------------------------------

def _print_metrics_line(metrics, in_q, out_q) -> None:
    print(
        f"[{metrics.elapsed_seconds:6.1f}s] "
        f"in={metrics.input_frames:>9d}f out={metrics.output_frames:>9d}f "
        f"qin={in_q.qsize():>3d} qout={out_q.qsize():>3d} "
        f"drop(in={metrics.input_queue_drops},out={metrics.output_queue_drops}) "
        f"under={metrics.output_underruns} "
        f"in_pk={metrics.input_peak_dbfs:6.1f}dB out_pk={metrics.output_peak_dbfs:6.1f}dB "
        f"nan={metrics.nan_inf_scrub_count} "
        f"rvc_n={metrics.rvc_chunks_processed} "
        f"infer(last={metrics.rvc_inference_last_ms:4.0f} "
        f"mean={metrics.rvc_inference_mean_ms:4.0f} "
        f"max={metrics.rvc_inference_max_ms:4.0f})ms "
        f"fb={metrics.rvc_fallback_count} "
        f"stale={metrics.rvc_stale_chunk_drops} "
        f"odrop={metrics.rvc_output_blocks_dropped} "
        f"sola={metrics.rvc_sola_offset_last:+d}",
        flush=True,
    )


def _print_summary(
    metrics, input_info: AudioDeviceInfo, output_info: AudioDeviceInfo
) -> None:
    print("\n--- final summary ---")
    print(f"elapsed_seconds          = {metrics.elapsed_seconds:.2f}")
    print(f"input_device             = [{input_info.index}] {input_info.name}")
    print(f"output_device            = [{output_info.index}] {output_info.name}")
    print(f"input_frames             = {metrics.input_frames}")
    print(f"output_frames            = {metrics.output_frames}")
    print(f"input_queue_drops        = {metrics.input_queue_drops}")
    print(f"output_queue_drops       = {metrics.output_queue_drops}")
    print(f"output_underruns         = {metrics.output_underruns}")
    print(f"  startup_underruns      = {metrics.startup_output_underruns}")
    print(f"  steady_state_underruns = {metrics.steady_state_output_underruns}")
    print(f"nan_inf_scrub_count      = {metrics.nan_inf_scrub_count}")
    print(f"input_status_flag_count  = {metrics.input_status_flag_count}")
    print(f"output_status_flag_count = {metrics.output_status_flag_count}")
    print(f"rvc_chunks_processed     = {metrics.rvc_chunks_processed}")
    print(f"rvc_inference_mean_ms    = {metrics.rvc_inference_mean_ms:.1f}")
    print(f"rvc_inference_max_ms     = {metrics.rvc_inference_max_ms:.1f}")
    print(f"rvc_chunk_ms_budget      = {metrics.rvc_chunk_ms_budget:.1f}")
    print(f"rvc_fallback_count       = {metrics.rvc_fallback_count}")
    print(f"rvc_stale_chunk_drops    = {metrics.rvc_stale_chunk_drops}")
    print(f"rvc_output_blocks_enqueued = {metrics.rvc_output_blocks_enqueued}")
    print(f"rvc_output_blocks_dropped  = {metrics.rvc_output_blocks_dropped}")
    print(f"frame_restoration_shortfall_count = {metrics.frame_restoration_shortfall_count}")
    print(f"rvc_sola_applied_count   = {metrics.rvc_sola_applied_count}")
    print(f"rvc_sola_offset_last     = {metrics.rvc_sola_offset_last}")
    print(f"rvc_resample_mean_ms     = {metrics.rvc_resample_mean_ms:.2f}")
    print(f"max_input_queue_depth    = {metrics.max_input_queue_depth}")
    print(f"model                    = {metrics.rvc_model_basename or '(none)'}")
    print(f"index                    = {metrics.rvc_index_basename or '(none)'}")
    print(f"chunk_ms                 = {metrics.rvc_chunk_ms}")


# ---------------------------------------------------------------------------
# Queue sizing
# ---------------------------------------------------------------------------

def queue_blocks_from_ms(
    queue_ms: float, block_size: int, sample_rate: int, minimum: int = 64
) -> int:
    """Convert a queue capacity in ms to a number of blocks (>= minimum)."""
    if block_size <= 0 or sample_rate <= 0:
        raise ValueError("block_size and sample_rate must be > 0")
    if queue_ms <= 0:
        return int(minimum)
    block_ms = float(block_size) * 1000.0 / float(sample_rate)
    n = int(round(float(queue_ms) / block_ms))
    return max(int(minimum), n)


# ---------------------------------------------------------------------------
# The realtime RVC stream
# ---------------------------------------------------------------------------

def run_rvc_stream(
    config: AudioRuntimeConfig,
    engine,
    chunk_ms: float,
    duration_seconds: Optional[float] = None,
    allow_virtual_cable_input: bool = False,
    metrics_interval_seconds: float = 1.0,
    rvc_queue_ms: float = 6000.0,
    rvc_prebuffer_ms: Optional[float] = None,
    drop_stale_input: bool = True,
    context_ms: float = 500.0,
    tail_pad_ms: float = 30.0,
    sola_search_ms: float = 10.0,
) -> RuntimeMetrics:
    """Open ``system default mic -> RVC worker -> CABLE Input`` and run it.

    The caller owns ``engine`` and must have already called
    ``engine.load()`` so any dependency/model failure happens before any
    audio device is opened.
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
    if context_ms < 0:
        raise ValueError("context_ms must be >= 0")
    if tail_pad_ms < 0:
        raise ValueError("tail_pad_ms must be >= 0")
    if sola_search_ms < 0:
        raise ValueError("sola_search_ms must be >= 0")
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
    if config.input_device_substring:
        input_info = resolve_input_device(
            raw_devices,
            config.input_device_substring,
            allow_virtual_cable=allow_virtual_cable_input,
        )
        input_source = f"substring {config.input_device_substring!r}"
    else:
        try:
            default_in = int(sd.default.device[0])
        except (TypeError, ValueError, IndexError):
            default_in = -1
        input_info = select_default_input_device(raw_devices, default_in)
        input_source = "system default"
    output_info = resolve_output_device(raw_devices, config.output_device_substring)

    chunk_size = max(1, int(round(chunk_ms / 1000.0 * config.sample_rate)))
    context_size = max(0, int(round(context_ms / 1000.0 * config.sample_rate)))
    tail_pad_size = max(0, int(round(tail_pad_ms / 1000.0 * config.sample_rate)))
    # SOLA: search window (± shift) and the link signature length (2x the
    # search, for a robust correlation). 0 disables seam alignment.
    sola_search_size = max(0, int(round(sola_search_ms / 1000.0 * config.sample_rate)))
    sola_link_size = 2 * sola_search_size

    rvc_queue_blocks = queue_blocks_from_ms(
        float(rvc_queue_ms), config.block_size, config.sample_rate,
        minimum=config.queue_blocks,
    )
    # Prebuffer: silence inserted into the output queue before the first
    # real chunk lands. Hides first-chunk inference + absorbs an inference
    # spike, at the cost of that much added latency. Quality-first default
    # = 3 x chunk_ms.
    prebuffer_ms = (
        float(rvc_prebuffer_ms)
        if rvc_prebuffer_ms is not None
        else float(chunk_ms) * 3.0
    )
    prebuffer_blocks = max(
        0, int(round(prebuffer_ms * config.sample_rate / 1000.0 / config.block_size))
    )

    import os
    model_basename = os.path.basename(engine.config.model_path or "")
    index_basename = (
        os.path.basename(engine.config.index_path) if engine.config.index_path else ""
    )

    print(
        f"input device  [{input_info.index:>3}] ({input_source}): {input_info.name}\n"
        f"output device [{output_info.index:>3}]: {output_info.name}\n"
        f"sample_rate={config.sample_rate} block_size={config.block_size} "
        f"channels={config.channels}\n"
        f"chunk_ms={chunk_ms:.0f} ({chunk_size} samples)  "
        f"context_ms={context_ms:.0f} ({context_size})  "
        f"tail_pad_ms={tail_pad_ms:.0f} ({tail_pad_size})\n"
        f"sola_search_ms={sola_search_ms:.0f} ({sola_search_size} samples"
        f"{', OFF' if sola_search_size == 0 else ''}) -- seam alignment, no blend\n"
        f"model={model_basename or '(none)'}  index={index_basename or '(none)'}  "
        f"backend={engine.backend_name}  device={engine.resolved_device or '(unknown)'}"
        + (f" / {engine.cuda_device_name}" if engine.cuda_device_name else "")
        + f"\nqueue={rvc_queue_blocks} blocks (~{rvc_queue_ms:.0f} ms)  "
        f"prebuffer={prebuffer_blocks} blocks (~{prebuffer_ms:.0f} ms)  "
        f"drop_stale_input={drop_stale_input}\n"
        "faithful-carrier: model defines the voice; runtime only resamples "
        "to the stream SR, slices exactly, and scrubs NaN. No pitch/stretch/"
        "crossfade/EQ. Identity fallback only on backend error."
    )
    if allow_virtual_cable_input:
        print(
            "WARNING: --allow-virtual-cable-input is ON (diagnostic). Verify "
            "there is no feedback loop with Discord/OBS."
        )

    in_q: "queue.Queue" = queue.Queue(maxsize=rvc_queue_blocks)
    out_q: "queue.Queue" = queue.Queue(maxsize=rvc_queue_blocks)
    metrics = RuntimeMetrics()
    stop_event = threading.Event()

    # Seed session metadata so the summary is populated even on a short run.
    metrics.rvc_chunk_ms = float(chunk_ms)
    metrics.rvc_model_basename = model_basename
    metrics.rvc_index_basename = index_basename

    if prebuffer_blocks > 0:
        silence = np.zeros(config.block_size, dtype=np.float32)
        buffered = 0
        for _ in range(prebuffer_blocks):
            try:
                out_q.put_nowait(silence.copy())
                buffered += 1
            except queue.Full:
                break
        print(
            f"prebuffered {buffered} silence blocks "
            f"(~{buffered * config.block_size * 1000.0 / config.sample_rate:.0f} ms)"
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
            if metrics.first_real_output_seen:
                metrics.steady_state_output_underruns += 1
            else:
                metrics.startup_output_underruns += 1
            metrics.output_peak_dbfs = dbfs_peak(np.zeros(1, dtype=np.float32))
            metrics.output_frames += int(frames)
            return
        n = min(int(block.shape[0]), int(frames))
        outdata[:n, 0] = block[:n]
        if n < frames:
            outdata[n:, 0] = 0.0
        metrics.output_peak_dbfs = dbfs_peak(block[:n])
        metrics.output_rms_dbfs = dbfs_rms(block[:n])
        metrics.output_frames += int(frames)

    from ..engine.worker import rvc_worker_loop
    worker_thread = threading.Thread(
        target=rvc_worker_loop,
        args=(engine, in_q, out_q, metrics, stop_event, _SHUTDOWN_SENTINEL),
        kwargs={
            "sample_rate": int(config.sample_rate),
            "chunk_size": int(chunk_size),
            "output_block_size": int(config.block_size),
            "context_size": int(context_size),
            "tail_pad_size": int(tail_pad_size),
            "sola_search_size": int(sola_search_size),
            "sola_link_size": int(sola_link_size),
            "drop_stale_input": bool(drop_stale_input),
        },
        name="rvc-worker",
        daemon=True,
    )
    worker_thread.start()

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
                qi = in_q.qsize()
                if qi > metrics.max_input_queue_depth:
                    metrics.max_input_queue_depth = qi
                qo = out_q.qsize()
                if qo > metrics.max_output_queue_depth:
                    metrics.max_output_queue_depth = qo
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
        try:
            in_q.put_nowait(_SHUTDOWN_SENTINEL)
        except queue.Full:
            pass
        worker_thread.join(timeout=2.0)
        metrics.elapsed_seconds = time.monotonic() - start_wall
        _print_metrics_line(metrics, in_q, out_q)
        _print_summary(metrics, input_info, output_info)

    return metrics
