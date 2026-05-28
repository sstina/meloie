"""CLI entry point for the Python RVC voice changer.

Subcommands / flags:

    --list-devices
        Enumerate audio devices via sounddevice.

    --check-config PATH
        Load + validate a runtime JSON config and exit.

    --mode identity --config PATH [--duration-seconds N]
                                   [--input-device-substring STR]
                                   [--output-device-substring STR]
                                   [--allow-virtual-cable-input]
        Run the Stage 1 realtime identity stream.

Importing this module must NOT open audio devices and must NOT import
``sounddevice``. The ``sounddevice`` import lives inside the streams
layer's functions and only fires when audio is explicitly started.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .audio.streams import AudioRuntimeConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tvoice-rvc",
        description="Python realtime RVC voice changer (Stage 1).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices visible to sounddevice and exit.",
    )
    parser.add_argument(
        "--check-config",
        metavar="PATH",
        help="Load and validate a runtime JSON config and exit.",
    )
    parser.add_argument(
        "--mode",
        choices=["identity"],
        help="Run the realtime worker in the given mode.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Runtime config JSON to use with --mode.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Stop the stream after N seconds. Omit to run until Ctrl+C.",
    )
    parser.add_argument(
        "--input-device-substring",
        help="Override the input device substring from the config.",
    )
    parser.add_argument(
        "--output-device-substring",
        help="Override the output device substring from the config.",
    )
    parser.add_argument(
        "--allow-virtual-cable-input",
        action="store_true",
        help="Diagnostic only: allow the VB-CABLE capture endpoint as "
             "the app's input. Risk of feedback loop with Discord/OBS.",
    )
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


def _cmd_mode_identity(args: argparse.Namespace) -> int:
    if not args.config:
        print(
            "error: --mode identity requires --config PATH\n"
            "example: python -m src.main --mode identity "
            "--config config/runtime.example.json --duration-seconds 30",
            file=sys.stderr,
        )
        return 2

    try:
        config = _load_config(args.config)
    except FileNotFoundError:
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"error: invalid config in {args.config}: {exc}", file=sys.stderr)
        return 2

    # CLI overrides take precedence over the config file.
    if args.input_device_substring:
        config = AudioRuntimeConfig(
            sample_rate=config.sample_rate,
            block_size=config.block_size,
            channels=config.channels,
            input_device_substring=args.input_device_substring,
            output_device_substring=config.output_device_substring,
            queue_blocks=config.queue_blocks,
            mode=config.mode,
        )
    if args.output_device_substring:
        config = AudioRuntimeConfig(
            sample_rate=config.sample_rate,
            block_size=config.block_size,
            channels=config.channels,
            input_device_substring=config.input_device_substring,
            output_device_substring=args.output_device_substring,
            queue_blocks=config.queue_blocks,
            mode=config.mode,
        )

    # Lazy import: hardware code is only reachable on this branch.
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        return _cmd_list_devices()
    if args.check_config:
        return _cmd_check_config(args.check_config)
    if args.mode == "identity":
        return _cmd_mode_identity(args)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
