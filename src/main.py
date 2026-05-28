"""CLI entry point for the Python RVC voice changer.

Subcommands / flags:

    --list-devices
        Enumerate audio devices via sounddevice.

    --check-config PATH
        Load + validate a runtime JSON config and exit.

    --mode identity --config PATH [...]
        Stage 1 realtime identity stream.

    --mode rvc --config PATH --model-path PATH [--index-path PATH] [...]
        Stage 2 realtime RVC stream. Lazy-imports ``rvc_engine``;
        a missing backend / model produces an actionable error before
        any audio device is opened.

Importing this module must NOT open audio devices and must NOT import
``sounddevice``, ``torch``, ``infer_rvc_python``, or ``rvc_engine``.
Those imports live inside ``_cmd_mode_identity`` / ``_cmd_mode_rvc``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .audio.streams import AudioRuntimeConfig


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tvoice-rvc",
        description="Python realtime RVC voice changer (Stage 1 + Stage 2).",
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices visible to sounddevice and exit.")
    parser.add_argument("--check-config", metavar="PATH",
                        help="Load and validate a runtime JSON config and exit.")
    parser.add_argument("--mode", choices=["identity", "rvc"],
                        help="Run the realtime worker in the given mode.")
    parser.add_argument("--config", metavar="PATH",
                        help="Runtime config JSON to use with --mode.")
    parser.add_argument("--duration-seconds", type=float, default=None,
                        help="Stop the stream after N seconds. Omit to run until Ctrl+C.")
    parser.add_argument("--input-device-substring",
                        help="Override the input device substring from the config.")
    parser.add_argument("--output-device-substring",
                        help="Override the output device substring from the config.")
    parser.add_argument("--allow-virtual-cable-input", action="store_true",
                        help="Diagnostic only: allow VB-CABLE capture endpoint as input.")

    # RVC-mode flags
    rvc = parser.add_argument_group("RVC mode (--mode rvc)")
    rvc.add_argument("--model-path", help="Path to RVC .pth model.")
    rvc.add_argument("--index-path", default=None, help="Path to .index file (optional).")
    rvc.add_argument("--hubert-path", default=None,
                     help="Path to hubert_base.pt (optional; backend downloads it if omitted).")
    rvc.add_argument("--rmvpe-path", default=None,
                     help="Path to rmvpe.pt (optional; backend downloads it if omitted).")
    rvc.add_argument("--backend", default="infer_rvc_python",
                     help="RVC backend identifier (default: infer_rvc_python).")
    rvc.add_argument("--f0-method", default="rmvpe")
    rvc.add_argument("--index-rate", type=float, default=0.5)
    rvc.add_argument("--protect", type=float, default=0.33)
    rvc.add_argument("--filter-radius", type=int, default=3)
    rvc.add_argument("--rms-mix-rate", type=float, default=0.25)
    rvc.add_argument("--pitch-shift", type=int, default=0)
    rvc.add_argument("--chunk-ms", type=float, default=180.0,
                     help="RVC chunk size in milliseconds (default 180).")
    rvc.add_argument("--crossfade-ms", type=float, default=20.0,
                     help="Crossfade length at chunk boundaries (default 20; "
                          "set 0 to disable, then expect occasional clicks "
                          "until Stage 3 refines this).")
    rvc.add_argument("--device", default="auto",
                     choices=["auto", "cpu", "cuda", "directml_experimental"],
                     help="Inference device. 'auto' picks cuda if available, "
                          "otherwise cpu. 'cuda' fails if CUDA is unavailable. "
                          "directml_experimental is reserved for a future milestone.")
    rvc.add_argument("--force-cpu", action="store_true",
                     help="DEPRECATED: equivalent to --device cpu.")
    rvc.add_argument("--resample-sr", type=int, default=None,
                     help="Ask the RVC backend to resample its output to this "
                          "sample rate. Default for realtime is the stream's "
                          "sample_rate (e.g. 48000) so the OutputStream sees "
                          "matching audio.")

    return parser


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(path: str) -> AudioRuntimeConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    config = AudioRuntimeConfig(
        sample_rate=int(data.get("sample_rate", 48000)),
        block_size=int(data.get("block_size", 480)),
        channels=int(data.get("channels", 1)),
        input_device_substring=str(data.get("input_device_substring", "Microphone")),
        output_device_substring=str(data.get("output_device_substring", "CABLE Input")),
        queue_blocks=int(data.get("queue_blocks", 64)),
        mode=str(data.get("mode", "identity")),
    )
    config.validate()
    return config


def _apply_device_overrides(
    config: AudioRuntimeConfig, args: argparse.Namespace
) -> AudioRuntimeConfig:
    in_sub = args.input_device_substring or config.input_device_substring
    out_sub = args.output_device_substring or config.output_device_substring
    if in_sub == config.input_device_substring and out_sub == config.output_device_substring:
        return config
    return AudioRuntimeConfig(
        sample_rate=config.sample_rate,
        block_size=config.block_size,
        channels=config.channels,
        input_device_substring=in_sub,
        output_device_substring=out_sub,
        queue_blocks=config.queue_blocks,
        mode=config.mode,
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

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
    print(f"  input_device_substring = {config.input_device_substring!r}")
    print(f"  output_device_substring= {config.output_device_substring!r}")
    print(f"  queue_blocks           = {config.queue_blocks}")
    print(f"  mode                   = {config.mode}")
    return 0


def _require_config(args: argparse.Namespace) -> Optional[AudioRuntimeConfig]:
    if not args.config:
        print(
            f"error: --mode {args.mode} requires --config PATH\n"
            "example: python -m src.main --mode identity --config "
            "config/runtime.example.json --duration-seconds 30",
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


def _cmd_mode_identity(args: argparse.Namespace) -> int:
    config = _require_config(args)
    if config is None:
        return 2
    config = _apply_device_overrides(config, args)

    from .audio.streams import run_identity_stream
    from .audio.devices import FeedbackLoopRisk

    try:
        run_identity_stream(
            config,
            duration_seconds=args.duration_seconds,
            allow_virtual_cable_input=args.allow_virtual_cable_input,
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
    return 0


def _cmd_mode_rvc(args: argparse.Namespace) -> int:
    config = _require_config(args)
    if config is None:
        return 2
    config = _apply_device_overrides(config, args)

    if not args.model_path:
        print(
            "error: --mode rvc requires --model-path /path/to/model.pth\n"
            "Place local model files under models/local/ (gitignored).",
            file=sys.stderr,
        )
        return 2

    # Lazy imports — engine pulls in numpy only at module load; the
    # backend (torch + infer_rvc_python) is imported inside engine.load().
    from .engine.rvc_engine import (
        DependencyMissingError,
        ModelLoadError,
        RvcEngine,
        RvcEngineConfig,
        RvcInferenceError,
    )
    from .audio.devices import FeedbackLoopRisk
    from .audio.streams import run_rvc_stream

    # For realtime, default --resample-sr to the stream's sample rate so the
    # OutputStream sees matching audio (the kiki model natively returns 40 kHz;
    # without this, a 40 kHz buffer would be played as if it were 48 kHz).
    resample_sr = args.resample_sr if args.resample_sr is not None else config.sample_rate

    rvc_config = RvcEngineConfig(
        model_path=args.model_path,
        index_path=args.index_path,
        backend=args.backend,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        protect=args.protect,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        pitch_shift=args.pitch_shift,
        sample_rate=config.sample_rate,
        resample_sr=resample_sr,
        device=args.device,
        force_cpu=args.force_cpu,
        hubert_path=args.hubert_path,
        rmvpe_path=args.rmvpe_path,
    )

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
    print(
        f"RVC engine loaded. resolved_device={engine.resolved_device} "
        f"cuda_device={engine.cuda_device_name or '(n/a)'} "
        f"resample_sr={resample_sr}"
    )

    try:
        run_rvc_stream(
            config,
            engine=engine,
            chunk_ms=args.chunk_ms,
            crossfade_ms=args.crossfade_ms,
            duration_seconds=args.duration_seconds,
            allow_virtual_cable_input=args.allow_virtual_cable_input,
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
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        return _cmd_list_devices()
    if args.check_config:
        return _cmd_check_config(args.check_config)
    if args.mode == "identity":
        return _cmd_mode_identity(args)
    if args.mode == "rvc":
        return _cmd_mode_rvc(args)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
