"""Shared audio helpers for the realtime RVC path.

This module provides the device-layer building blocks the realtime runner
needs — device enumeration / resolution, the VB-CABLE feedback-loop guard,
queue sizing, the runtime-config dataclass, and metrics printing. The
realtime runner itself (mic -> worker -> CABLE Input) lives in
``streaming_stream.py`` (the v2 direct engine), which imports the helpers
below.

Hard rules baked into this module:

* Importing this module must NOT import sounddevice. The import is lazy
  and lives inside the functions that actually touch hardware.
* The input device is the user's microphone — by default the Windows
  default recording device ("系统默认 mic"); an explicit substring may
  override it. The output device must be ``CABLE Input``. ``CABLE Output``
  is refused for both sides (feedback-loop guard).
* No system / device defaults are changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional

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
    # Optional headphone-monitor sink. ``None`` => system default output. A second
    # (parallel) OutputStream plays the SAME converted block as CABLE — a duplicate
    # sink, never reshaped (faithful-carrier). Gated live by ``MonitorState.enabled``.
    monitor_device_substring: Optional[str] = None

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
class MonitorState:
    """Live on/off gate for the headphone-monitor sink. The control thread flips
    ``enabled`` (a single bool — atomic under the GIL, no lock); the audio
    callbacks read it. Created by ``RealtimeSession.start`` and shared with the
    runner so the monitor toggles instantly without opening/closing the stream."""

    enabled: bool = False


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
        f"sil={metrics.rvc_silence_skipped_count} "
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
    print(f"rvc_silence_skipped_count= {metrics.rvc_silence_skipped_count}")
    print(f"rvc_output_blocks_enqueued = {metrics.rvc_output_blocks_enqueued}")
    print(f"rvc_output_blocks_dropped  = {metrics.rvc_output_blocks_dropped}")
    print(f"rvc_sola_offset_last     = {metrics.rvc_sola_offset_last}")
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
