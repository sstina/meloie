"""CLI entry point for the realtime RVC voice changer.

The one job: ``system default mic -> RVC inference -> CABLE Input`` so a
downstream app (Discord / OBS / recorder) selecting ``CABLE Output`` as
its microphone hears your voice converted to the model's voice.

Usage::

    # list devices (which mic is the system default, where CABLE Input is)
    python -m src.main --list-devices

    # validate a runtime config
    python -m src.main --check-config config/runtime.example.json

    # run: system default mic -> kiki model -> CABLE Input
    python -m src.main --config config/runtime.example.json \\
        --model-profile config/model_profiles/kiki.example.json \\
        --device cuda

Voice identity comes entirely from the model profile (the trained model
defines the voice). The CLI only takes engineering knobs (device, chunk
size, queue/latency, which mic). There are no voice-shaping flags.

Importing this module must NOT open audio devices and must NOT import
sounddevice / torch / infer_rvc_python / rvc_engine. Those imports live
inside the run handler.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .audio.streams import AudioRuntimeConfig


def _force_utf8_stdio() -> None:
    """Make stdout/stderr tolerate non-GBK device names on a CN locale.

    Device names here include characters (e.g. Tai Viet glyphs in a
    Bluetooth headset name) that the Windows GBK console codec cannot
    encode, which previously crashed ``--list-devices``. Reconfigure to
    UTF-8 with replacement so printing device names never crashes the run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tvoice-rvc",
        description="Realtime RVC voice changer: system default mic -> RVC "
                    "model -> CABLE Input.",
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices (I/O marks the system "
                             "defaults) and exit.")
    parser.add_argument("--check-config", metavar="PATH",
                        help="Load and validate a runtime JSON config and exit.")
    parser.add_argument("--config", metavar="PATH",
                        help="Runtime config JSON (device/sample-rate/queue).")
    parser.add_argument("--duration-seconds", type=float, default=None,
                        help="Stop after N seconds. Omit to run until Ctrl+C.")
    parser.add_argument("--input-device", "--input-device-substring",
                        dest="input_device", default=None,
                        help="Mic device name fragment. Omit to follow the "
                             "Windows default recording device.")
    parser.add_argument("--output-device", "--output-device-substring",
                        dest="output_device", default=None,
                        help="Output device name fragment (must be CABLE Input).")
    parser.add_argument("--allow-virtual-cable-input", action="store_true",
                        help="Diagnostic only: allow VB-CABLE capture endpoint "
                             "as input.")

    voice = parser.add_argument_group("Voice (the trained model defines it)")
    voice.add_argument("--model-profile", default=None,
                       help="Path to a model profile JSON (recommended). "
                            "Supplies the voice identity + its inference params.")
    voice.add_argument("--model-path", default=None,
                       help="Path to RVC .pth (use instead of a profile for a "
                            "quick run; profile params then use defaults).")
    voice.add_argument("--index-path", default=None, help="Path to .index file.")
    voice.add_argument("--hubert-path", default=None, help="Path to hubert_base.pt.")
    voice.add_argument("--rmvpe-path", default=None, help="Path to rmvpe.pt.")
    voice.add_argument("--pitch", type=int, default=None, metavar="SEMITONES",
                       help="Transpose (变调) in semitones applied to the input "
                            "F0 before conversion -- THE main creative knob. A "
                            "female model typically needs about +12 for a male "
                            "voice (e.g. kiki's intended setting is +12). This "
                            "conditions the model's input pitch (not an output "
                            "pitch-shift). Overrides the profile's pitch_shift; "
                            "default: the profile's value.")

    eng = parser.add_argument_group("Engineering knobs (stability / latency)")
    eng.add_argument("--backend", default="infer_rvc_python",
                     help="RVC backend identifier.")
    eng.add_argument("--device", default="auto",
                     choices=["auto", "cpu", "cuda"],
                     help="Inference device. 'auto' = cuda if available else cpu.")
    eng.add_argument("--precision", default="auto",
                     choices=["auto", "fp32", "fp16"],
                     help="Inference numeric precision. 'auto' = backend default "
                          "(FP16 on most NVIDIA GPUs). 'fp32' = more stable and, "
                          "on this backend, uses a 1 s reflect-pad instead of 3 s "
                          "(less audio per inference); costs ~200 MB VRAM. Pure "
                          "precision -- does not reshape the voice.")
    eng.add_argument("--chunk-ms", type=float, default=500.0,
                     help="RVC chunk size in ms (accumulation latency). "
                          "Default 500 — the conservative low-latency setting "
                          "(steady-state underruns ~0 on an RTX 4080). Do NOT go "
                          "below ~400: worst-case inference is a fixed ~350 ms "
                          "(constant reflect-pad, not chunk-scaled), so a smaller "
                          "budget risks dropouts. Larger = more model context, "
                          "more latency.")
    eng.add_argument("--rvc-context-ms", type=float, default=500.0,
                     help="Input-side left-context fed to the model as warm-up, "
                          "then sliced away. Continuity only; no voice change. "
                          "Default 500 ms clears the RVC decoder's internal "
                          "~240 ms (24-frame) lead-in margin with headroom; "
                          "canonical realtime RVC uses up to 2500 ms. Larger = "
                          "fewer chunk-boundary artifacts, more inference cost.")
    eng.add_argument("--tail-pad-ms", type=float, default=30.0,
                     help="Look-ahead tail pad that absorbs the model's ~20 ms "
                          "tail-frame loss; sliced away. No stretch, no pitch.")
    eng.add_argument("--sola-search-ms", type=float, default=10.0,
                     help="SOLA seam-alignment search window (ms). Phase-matches "
                          "each chunk's seam to the previous chunk's tail by "
                          "choosing the cut offset (no crossfade, no blend, no "
                          "sample edit) -- removes chunk-boundary comb-filter "
                          "'电音'. 0 disables. Must be <= tail-pad-ms.")
    eng.add_argument("--rvc-queue-ms", type=float, default=6000.0,
                     help="Per-direction queue capacity in ms.")
    eng.add_argument("--rvc-prebuffer-ms", type=float, default=None,
                     help="Output silence prebuffer before first real audio = the "
                          "standing output latency. Default: 800 ms (an absolute "
                          "cushion sized to cover one ~350 ms inference spike plus "
                          "one chunk's output burst; decoupled from chunk_ms on "
                          "purpose). Lower = less latency but more underruns.")
    eng.add_argument("--warmup-rvc-count", type=int, default=2,
                     help="Dummy inferences before opening the stream (hides "
                          "the cold-start stall). 0 to disable.")
    eng.add_argument("--drop-stale-input",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="If inference falls behind, drop oldest chunks to "
                          "keep latency bounded. Default: on.")
    eng.add_argument("--silence-threshold-dbfs", type=float, default=None,
                     help="SilenceFront (w-okada borrow): skip RVC inference on "
                          "chunks whose input RMS is below this level (dBFS) and "
                          "emit silence -- saves GPU on silence, no voice change. "
                          "Default: OFF (an over-high value would silence soft "
                          "speech). Opt in with e.g. -60; watch the live in_rms "
                          "vs your soft-speech level and keep it well below.")
    eng.add_argument("--silence-hangover-ms", type=float, default=500.0,
                     help="Keep processing this long after the last voiced chunk "
                          "so soft/trailing syllables are never clipped by the "
                          "silence skip. Only used when --silence-threshold-dbfs "
                          "is set. Default 500.")
    eng.add_argument("--resample-sr", type=int, default=None,
                     help="Ask the backend to resample its output to this SR. "
                          "Default: profile's value or 0 (model-native; the "
                          "worker resamples to the stream SR).")

    return parser


def _load_config(path: str) -> AudioRuntimeConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    in_sub = data.get("input_device_substring", None)
    config = AudioRuntimeConfig(
        sample_rate=int(data.get("sample_rate", 48000)),
        block_size=int(data.get("block_size", 480)),
        channels=int(data.get("channels", 1)),
        input_device_substring=(str(in_sub) if in_sub else None),
        output_device_substring=str(data.get("output_device_substring", "CABLE Input")),
        queue_blocks=int(data.get("queue_blocks", 64)),
    )
    config.validate()
    return config


def _apply_device_overrides(
    config: AudioRuntimeConfig, args: argparse.Namespace
) -> AudioRuntimeConfig:
    in_sub = args.input_device if args.input_device is not None else config.input_device_substring
    out_sub = args.output_device or config.output_device_substring
    if in_sub == config.input_device_substring and out_sub == config.output_device_substring:
        return config
    return AudioRuntimeConfig(
        sample_rate=config.sample_rate,
        block_size=config.block_size,
        channels=config.channels,
        input_device_substring=in_sub,
        output_device_substring=out_sub,
        queue_blocks=config.queue_blocks,
    )


def _cmd_list_devices() -> int:
    try:
        from .audio.streams import describe_devices
        print(describe_devices())
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _cmd_check_config(path: str) -> int:
    try:
        config = _load_config(path)
    except FileNotFoundError:
        print(f"error: config file not found: {path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {path}: {exc}", file=sys.stderr)
        return 2
    except (TypeError, ValueError) as exc:
        print(f"error: invalid config in {path}: {exc}", file=sys.stderr)
        return 2

    print(f"ok: {path} parsed and validated")
    print(f"  sample_rate            = {config.sample_rate}")
    print(f"  block_size             = {config.block_size}")
    print(f"  channels               = {config.channels}")
    print(f"  input_device_substring = "
          f"{config.input_device_substring!r} "
          f"{'(system default)' if not config.input_device_substring else ''}")
    print(f"  output_device_substring= {config.output_device_substring!r}")
    print(f"  queue_blocks           = {config.queue_blocks}")
    return 0


def _require_config(args: argparse.Namespace) -> Optional[AudioRuntimeConfig]:
    if not args.config:
        print(
            "error: running requires --config PATH\n"
            "example: python -m src.main --config config/runtime.example.json "
            "--model-profile config/model_profiles/kiki.example.json --device cuda",
            file=sys.stderr,
        )
        return None
    try:
        return _load_config(args.config)
    except FileNotFoundError:
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        return None
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"error: invalid config in {args.config}: {exc}", file=sys.stderr)
        return None


def _cmd_run(args: argparse.Namespace) -> int:
    config = _require_config(args)
    if config is None:
        return 2
    config = _apply_device_overrides(config, args)

    from .engine.model_profile import ModelProfileError, load_model_profile

    profile = None
    if args.model_profile:
        try:
            profile = load_model_profile(args.model_profile)
        except ModelProfileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 7
        print(f"loaded model profile: name={profile.name!r} "
              f"model_path={profile.model_path!r}")

    model_path = args.model_path or (profile.model_path if profile else None)
    if not model_path:
        print(
            "error: requires --model-profile PATH or --model-path "
            "/path/to/model.pth. Place model files under models/ (gitignored).",
            file=sys.stderr,
        )
        return 2

    def pick(attr, default):
        return getattr(profile, attr) if profile is not None else default

    if args.resample_sr is not None:
        resample_sr = args.resample_sr
    else:
        resample_sr = profile.resample_sr if profile is not None else 0

    from .engine.rvc_engine import (
        DependencyMissingError,
        ModelLoadError,
        RvcEngine,
        RvcEngineConfig,
        RvcInferenceError,
    )
    from .audio.devices import FeedbackLoopRisk
    from .audio.streams import run_rvc_stream

    rvc_config = RvcEngineConfig(
        model_path=model_path,
        index_path=args.index_path or pick("index_path", None),
        backend=args.backend,
        f0_method=pick("f0_method", "rmvpe"),
        index_rate=pick("index_rate", 0.5),
        protect=pick("protect", 0.33),
        filter_radius=pick("filter_radius", 3),
        rms_mix_rate=pick("rms_mix_rate", 1.0),
        pitch_shift=(args.pitch if args.pitch is not None else pick("pitch_shift", 0)),
        sample_rate=config.sample_rate,
        resample_sr=resample_sr,
        device=args.device,
        precision=args.precision,
        hubert_path=args.hubert_path or pick("hubert_path", None),
        rmvpe_path=args.rmvpe_path or pick("rmvpe_path", None),
    )

    print(f"voice params: pitch={rvc_config.pitch_shift:+d} semitones  "
          f"index_rate={rvc_config.index_rate}  f0={rvc_config.f0_method}  "
          f"protect={rvc_config.protect}  rms_mix={rvc_config.rms_mix_rate}")
    engine = RvcEngine(rvc_config)
    print(f"loading RVC backend={rvc_config.backend} device={args.device} ...")
    try:
        engine.load()
    except DependencyMissingError as exc:
        print(f"error: dependency missing: {exc}", file=sys.stderr)
        return 10
    except ModelLoadError as exc:
        print(f"error: model load failed: {exc}", file=sys.stderr)
        return 11
    print(f"RVC engine loaded. resolved_device={engine.resolved_device} "
          f"cuda_device={engine.cuda_device_name or '(n/a)'} "
          f"precision={engine.resolved_precision or '(unknown)'} "
          f"resample_sr={resample_sr}")

    if args.warmup_rvc_count > 0:
        warmup_samples = max(1, int(round(args.chunk_ms / 1000.0 * config.sample_rate)))
        print(f"warming up RVC ({args.warmup_rvc_count} calls, "
              f"chunk_samples={warmup_samples} at {config.sample_rate} Hz)...")
        try:
            for i, ms in enumerate(
                engine.warmup(warmup_samples, config.sample_rate, args.warmup_rvc_count)
            ):
                print(f"  warmup #{i + 1}: {ms:.0f} ms")
        except (RvcInferenceError, ValueError) as exc:
            print(f"warning: warmup failed (continuing): {exc}", file=sys.stderr)

    try:
        run_rvc_stream(
            config,
            engine=engine,
            chunk_ms=args.chunk_ms,
            duration_seconds=args.duration_seconds,
            allow_virtual_cable_input=args.allow_virtual_cable_input,
            rvc_queue_ms=args.rvc_queue_ms,
            rvc_prebuffer_ms=args.rvc_prebuffer_ms,
            drop_stale_input=args.drop_stale_input,
            context_ms=args.rvc_context_ms,
            tail_pad_ms=args.tail_pad_ms,
            sola_search_ms=args.sola_search_ms,
            silence_threshold_dbfs=args.silence_threshold_dbfs,
            silence_hangover_ms=args.silence_hangover_ms,
        )
    except FeedbackLoopRisk as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    except RvcInferenceError as exc:
        print(f"error: rvc inference fatal: {exc}", file=sys.stderr)
        return 12
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        return _cmd_list_devices()
    if args.check_config:
        return _cmd_check_config(args.check_config)
    if args.config or args.model_profile or args.model_path:
        return _cmd_run(args)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
