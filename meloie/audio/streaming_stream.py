"""Direct-mode realtime runner for the stateful ``StreamingRvcEngine`` (Path A).

Opens ``system default mic -> in_queue -> direct worker -> out_queue -> CABLE
Input`` and drives the persistent-buffer engine: the worker feeds raw
``block_frame`` blocks and the engine owns continuity (F0 cache + SOLA +
crossfade). Reuses streams.py's device resolution, feedback-loop guard,
callbacks, prebuffer, metrics, and shutdown so the proven plumbing is shared.
This is the sole realtime runner (v2-only build).
"""

from __future__ import annotations

import contextlib
import os
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

from ..safety.guard import dbfs_peak, dbfs_rms
from ..safety.metrics import RuntimeMetrics
from .devices import (
    is_probable_cable_input,
    select_default_input_device,
    select_device_by_substring,
)
from .streams import (
    AudioRuntimeConfig,
    MonitorState,
    _SHUTDOWN_SENTINEL,
    _print_metrics_line,
    _print_summary,
    queue_blocks_from_ms,
    resolve_input_device,
    resolve_output_device,
)


def run_streaming_stream(
    config: AudioRuntimeConfig,
    engine,
    *,
    duration_seconds: Optional[float] = None,
    allow_virtual_cable_input: bool = False,
    metrics_interval_seconds: float = 1.0,
    rvc_queue_ms: float = 6000.0,
    rvc_prebuffer_ms: Optional[float] = None,
    drop_stale_input: bool = True,
    stop_event=None,
    metrics: Optional[RuntimeMetrics] = None,
    print_metrics: bool = True,
    monitor_state: Optional[MonitorState] = None,
) -> RuntimeMetrics:
    """Run the direct-mode realtime loop. ``engine`` must be a loaded
    ``StreamingRvcEngine`` whose ``stream_sr`` matches ``config.sample_rate``.

    For programmatic control (e.g. a GUI via ``RealtimeSession``): inject a
    ``threading.Event`` as ``stop_event`` to stop the loop from another thread,
    inject a ``RuntimeMetrics`` to observe counters live while it runs, and set
    ``print_metrics=False`` to silence the per-second console line + summary.
    Omitting all three reproduces the original blocking CLI behaviour exactly."""
    if engine is None:
        raise ValueError("engine must be provided for run_streaming_stream")
    if not getattr(engine, "is_loaded", False):
        raise RuntimeError("engine.load() must be called before run_streaming_stream")
    config.validate()
    if int(engine.stream_sr) != int(config.sample_rate):
        raise RuntimeError(
            f"engine.stream_sr ({engine.stream_sr}) != config.sample_rate "
            f"({config.sample_rate}); rebuild the engine at the stream SR."
        )
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0 or None for unbounded")

    try:
        import sounddevice as sd  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sounddevice is not installed; cannot run realtime stream.") from exc

    raw_devices = list(sd.query_devices())
    if config.input_device_substring:
        input_info = resolve_input_device(
            raw_devices, config.input_device_substring,
            allow_virtual_cable=allow_virtual_cable_input,
        )
        input_source = f"substring {config.input_device_substring!r}"
    else:
        try:
            default_in = int(sd.default.device[0])
        except (TypeError, ValueError, IndexError):
            default_in = -1
        input_info = select_default_input_device(
            raw_devices, default_in, allow_virtual_cable=allow_virtual_cable_input,
        )
        input_source = "system default"
    output_info = resolve_output_device(raw_devices, config.output_device_substring)

    # Optional headphone-monitor sink: resolved only when the caller passes a
    # MonitorState (the GUI). The CLI omits it and stays a single CABLE sink. The
    # monitor is a real output (no CABLE feedback guard applies). monitor_index is
    # None => use the system default output (device=None on the OutputStream).
    monitor_index = None
    monitor_name = None
    monitor_blocked = False
    if monitor_state is not None:
        sub = config.monitor_device_substring
        if sub:
            try:
                minfo = select_device_by_substring(raw_devices, sub, kind="output")
                monitor_index, monitor_name = minfo.index, minfo.name
            except Exception as exc:
                print(f"monitor device {sub!r} not found ({exc}); using system default output")
        if monitor_name is None:
            try:
                di = int(sd.default.device[1])
                if di >= 0:
                    monitor_index, monitor_name = di, raw_devices[di]["name"]
                else:
                    monitor_name = "system default output"
            except Exception:
                monitor_name = "system default output"
        # Double-feed guard: if the monitor would render into CABLE Input (e.g.
        # the Windows DEFAULT playback device is the cable — a common VB-CABLE
        # setup), the same converted blocks would hit the cable through TWO
        # streams on independent clocks -> audible doubling/comb downstream.
        # Degrade to no monitor instead (mirrors the open-failure degrade path).
        if monitor_name is not None and is_probable_cable_input(str(monitor_name)):
            print(
                f"monitor disabled: {monitor_name!r} is the virtual cable's render "
                "endpoint (would double-feed the cable); pick a real headphone device."
            )
            monitor_blocked = True
            monitor_index = None

    block_frame = int(engine.block_frame)
    block_ms = block_frame * 1000.0 / float(config.sample_rate)

    rvc_queue_blocks = queue_blocks_from_ms(
        float(rvc_queue_ms), config.block_size, config.sample_rate, minimum=config.queue_blocks,
    )
    # Standing output prebuffer: the steady-state margin is prebuffer minus one
    # whole accumulate+infer cycle (~block_ms + inference). A fixed 800 ms is
    # comfortable at block 250 but leaves only ~30 ms at the cpu_zero game
    # mode's block 500 / ~270 ms CPU inference — so scale with the engine block
    # when the caller doesn't pin it. 250 ms blocks keep the historic 800 ms.
    prebuffer_ms = (
        float(rvc_prebuffer_ms) if rvc_prebuffer_ms is not None
        else max(800.0, block_ms + 400.0)
    )
    prebuffer_blocks = max(
        0, int(round(prebuffer_ms * config.sample_rate / 1000.0 / config.block_size))
    )

    model_basename = os.path.basename(getattr(engine.cfg, "model_path", "") or "")

    print(
        f"input device  [{input_info.index:>3}] ({input_source}): {input_info.name}\n"
        f"output device [{output_info.index:>3}]: {output_info.name}\n"
        f"sample_rate={config.sample_rate} block_size={config.block_size} "
        f"channels={config.channels}\n"
        f"ENGINE=direct (Applio persistent-buffer)  block_ms={block_ms:.0f} "
        f"({block_frame} samples)  context_ms={engine.cfg.context_ms:.0f}  "
        f"crossfade_ms={engine.cfg.crossfade_ms:.0f}\n"
        f"model={model_basename or '(none)'}  pitch={engine.pitch_shift:+d}  "
        f"f0={engine.cfg.f0_method}  protect={engine.protect}  "
        f"device={engine.resolved_device or '(unknown)'}"
        + (f" / {engine.cuda_device_name}" if engine.cuda_device_name else "")
        + f"\nqueue={rvc_queue_blocks} blocks (~{rvc_queue_ms:.0f} ms)  "
        f"prebuffer={prebuffer_blocks} blocks (~{prebuffer_ms:.0f} ms)  "
        f"drop_stale_input={drop_stale_input}\n"
        "faithful-carrier: the model + sanctioned seam crossfade define the voice; "
        "the worker only moves samples + identity-fallback on error."
    )
    if allow_virtual_cable_input:
        print("WARNING: --allow-virtual-cable-input is ON (diagnostic).")

    in_q: "queue.Queue" = queue.Queue(maxsize=rvc_queue_blocks)
    out_q: "queue.Queue" = queue.Queue(maxsize=rvc_queue_blocks)
    # Bounded monitor tap (~400 ms): independent clock from the CABLE stream, so
    # drop-oldest on overflow keeps monitor latency bounded; underflow -> silence.
    # Entries are OUTPUT-stream blocks (config.block_size, ~10 ms), not engine
    # blocks — size the cap from the right duration.
    monitor_q: Optional["queue.Queue"] = None
    if monitor_state is not None and not monitor_blocked:
        out_block_ms = config.block_size * 1000.0 / float(config.sample_rate)
        _mon_blocks = max(8, int(round(400.0 / max(1e-6, out_block_ms))))
        monitor_q = queue.Queue(maxsize=_mon_blocks)
    metrics = metrics if metrics is not None else RuntimeMetrics()
    stop_event = stop_event if stop_event is not None else threading.Event()
    metrics.rvc_chunk_ms = float(block_ms)
    metrics.rvc_model_basename = model_basename
    metrics.rvc_index_basename = os.path.basename(
        getattr(engine.cfg, "index_path", "") or ""
    )

    if prebuffer_blocks > 0:
        silence = np.zeros(config.block_size, dtype=np.float32)
        buffered = 0
        for _ in range(prebuffer_blocks):
            try:
                out_q.put_nowait(silence.copy())
                buffered += 1
            except queue.Full:
                break
        print(f"prebuffered {buffered} silence blocks "
              f"(~{buffered * config.block_size * 1000.0 / config.sample_rate:.0f} ms)")

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
            metrics.output_frames += int(frames)
            return
        n = min(int(block.shape[0]), int(frames))
        outdata[:n, 0] = block[:n]
        if n < frames:
            outdata[n:, 0] = 0.0
        metrics.output_peak_dbfs = dbfs_peak(block[:n])
        metrics.output_rms_dbfs = dbfs_rms(block[:n])
        metrics.output_frames += int(frames)
        # Tee the SAME converted block to the headphone monitor (faithful: no
        # reshaping). Non-blocking; drop-oldest keeps monitor latency bounded.
        if monitor_state is not None and monitor_state.enabled and monitor_q is not None:
            try:
                monitor_q.put_nowait(block)
            except queue.Full:
                try:
                    monitor_q.get_nowait()
                except queue.Empty:
                    pass     # consumer drained it between Full and here
                try:
                    monitor_q.put_nowait(block)   # single producer: cannot re-Full
                except queue.Full:
                    pass

    def monitor_callback(outdata, frames, time_info, status):  # noqa: ANN001
        # Plays the converted block on the headphone device when monitoring is on;
        # otherwise (and on underflow) silence. Writes the samples verbatim.
        if monitor_state is None or not monitor_state.enabled or monitor_q is None:
            outdata.fill(0.0)
            return
        try:
            block = monitor_q.get_nowait()
        except queue.Empty:
            outdata.fill(0.0)
            return
        n = min(int(block.shape[0]), int(frames))
        outdata[:n, 0] = block[:n]
        if n < frames:
            outdata[n:, 0] = 0.0

    from ..engine.streaming_worker import rvc_direct_worker_loop
    worker_thread = threading.Thread(
        target=rvc_direct_worker_loop,
        args=(engine, in_q, out_q, metrics, stop_event, _SHUTDOWN_SENTINEL),
        kwargs={
            "stream_sr": int(config.sample_rate),
            "block_frame": int(block_frame),
            "output_block_size": int(config.block_size),
            "drop_stale_input": bool(drop_stale_input),
        },
        name="rvc-direct-worker",
        daemon=True,
    )
    worker_thread.start()

    common = dict(
        samplerate=config.sample_rate, channels=config.channels,
        blocksize=config.block_size, dtype="float32", latency="low",
    )
    start_wall = time.monotonic()
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(sd.InputStream(
                device=input_info.index, callback=in_callback, **common))
            stack.enter_context(sd.OutputStream(
                device=output_info.index, callback=out_callback, **common))
            if monitor_state is not None and not monitor_blocked:
                # Always open the monitor stream (gated by the live ``enabled``
                # flag) so the toggle is instant; a failure here NEVER kills the
                # main CABLE link — we just degrade to no monitor.
                try:
                    stack.enter_context(sd.OutputStream(
                        device=monitor_index, callback=monitor_callback, **common))
                    print(f"monitor output: {monitor_name}  (enabled={monitor_state.enabled})")
                except Exception as exc:
                    print(f"monitor disabled (could not open '{monitor_name}'): {exc}")
            last_print = start_wall
            if print_metrics:
                print("running (direct engine). Ctrl+C to stop.", flush=True)
            while True:
                now = time.monotonic()
                metrics.elapsed_seconds = now - start_wall
                # max_input_queue_depth has a single producer: the worker (it
                # sees the drained backlog this 50 ms poll would miss).
                qo = out_q.qsize()
                if qo > metrics.max_output_queue_depth:
                    metrics.max_output_queue_depth = qo
                if stop_event.is_set():
                    break
                if duration_seconds is not None and metrics.elapsed_seconds >= duration_seconds:
                    break
                if print_metrics and now - last_print >= metrics_interval_seconds:
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
        if print_metrics:
            _print_metrics_line(metrics, in_q, out_q)
            _print_summary(metrics, input_info, output_info)

    return metrics
