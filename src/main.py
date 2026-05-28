"""CLI entry point for the Python RVC voice changer.

Currently exposes three commands:

    --list-devices                  enumerate audio devices via sounddevice
    --check-config <runtime.json>   validate a runtime config file
    --mode identity                 stage-1 identity stream (not implemented)

Importing this module must not open audio devices and must not import
``sounddevice``. The ``sounddevice`` import is lazy and lives inside
``audio.streams.list_audio_devices``.
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
        description="Python realtime RVC voice changer (Stage 0/1 skeleton).",
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
        help="Run the realtime worker in the given mode. "
             "Stage 1 identity streaming is not implemented in this skeleton.",
    )
    return parser


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
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print(f"error: config file not found: {path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {path}: {exc}", file=sys.stderr)
        return 2

    try:
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


def _cmd_mode_identity() -> int:
    print(
        "Stage 1 realtime identity streaming is intentionally NOT "
        "implemented in this skeleton.\n"
        "Next step: implement the async-queue audio loop in "
        "src/audio/streams.py and src/engine/worker.py, then wire "
        "this command to it. See rvc.md §6 (Stage 1) for the design."
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        return _cmd_list_devices()
    if args.check_config:
        return _cmd_check_config(args.check_config)
    if args.mode == "identity":
        return _cmd_mode_identity()

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
