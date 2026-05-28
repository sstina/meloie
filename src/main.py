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

    # RVC voice identity comes from a model profile JSON. The runtime CLI
    # only takes engineering knobs; the voice-identity flags below are
    # developer/debug overrides and emit a warning when set.
    rvc = parser.add_argument_group(
        "RVC mode (--mode rvc)",
        description="Use --model-profile to load a trained model's intended "
                    "inference settings. The remaining voice-identity flags "
                    "(model/index/hubert/rmvpe paths, f0_method, index_rate, "
                    "protect, filter_radius, rms_mix_rate, pitch_shift) are "
                    "DEVELOPER OVERRIDES — not normal user controls.",
    )
    rvc.add_argument("--model-profile", default=None,
                     help="Path to a model profile JSON (see "
                          "config/model_profiles/kiki.example.json). "
                          "Supplies the voice identity. Recommended.")
    rvc.add_argument("--model-path", default=None,
                     help="DEVELOPER OVERRIDE: path to RVC .pth model. "
                          "Overrides the profile.")
    rvc.add_argument("--index-path", default=None,
                     help="DEVELOPER OVERRIDE: path to .index file.")
    rvc.add_argument("--hubert-path", default=None,
                     help="DEVELOPER OVERRIDE: path to hubert_base.pt.")
    rvc.add_argument("--rmvpe-path", default=None,
                     help="DEVELOPER OVERRIDE: path to rmvpe.pt.")
    rvc.add_argument("--backend", default="infer_rvc_python",
                     help="RVC backend identifier (default: infer_rvc_python).")
    rvc.add_argument("--f0-method", default=None,
                     help="DEVELOPER OVERRIDE: F0 estimator. Profile default if omitted.")
    rvc.add_argument("--index-rate", type=float, default=None,
                     help="DEVELOPER OVERRIDE: retrieval index influence.")
    rvc.add_argument("--protect", type=float, default=None,
                     help="DEVELOPER OVERRIDE: consonant/breath protection.")
    rvc.add_argument("--filter-radius", type=int, default=None,
                     help="DEVELOPER OVERRIDE: F0 median-filter radius.")
    rvc.add_argument("--rms-mix-rate", type=float, default=None,
                     help="DEVELOPER OVERRIDE: source envelope mix rate.")
    rvc.add_argument("--pitch-shift", type=int, default=None,
                     help="DEVELOPER OVERRIDE: pitch shift in semitones.")
    rvc.add_argument("--chunk-ms", type=float, default=180.0,
                     help="RVC chunk size in milliseconds (default 180).")
    rvc.add_argument("--crossfade-ms", type=float, default=0.0,
                     help="Stitched crossfade at chunk boundaries in ms "
                          "(default 0 -- OFF). The current implementation "
                          "blends chunk N's tail with chunk N+1's head "
                          "purely on the OUTPUT side, with no INPUT-side "
                          "overlap. Audit measurement showed that smears "
                          "two temporally-disjoint regions together and "
                          "shifts the output timeline by one crossfade "
                          "length per chunk for zero faithfulness win. "
                          "Set >0 only to reproduce the legacy stitched-"
                          "blend behaviour. A future revision may add "
                          "true input-overlap crossfade.")
    rvc.add_argument("--device", default="auto",
                     choices=["auto", "cpu", "cuda", "directml_experimental"],
                     help="Inference device. 'auto' picks cuda if available, "
                          "otherwise cpu. 'cuda' fails if CUDA is unavailable. "
                          "directml_experimental is reserved for a future milestone.")
    rvc.add_argument("--force-cpu", action="store_true",
                     help="DEPRECATED: equivalent to --device cpu.")
    rvc.add_argument("--resample-sr", type=int, default=None,
                     help="Ask the RVC backend to resample its output to this "
                          "sample rate. Default for realtime is 0 (model's "
                          "native rate; the worker resamples to the stream "
                          "SR with a cheap linear interpolator). Override "
                          "to a positive SR to make the backend resample "
                          "internally (slower in our benchmarks).")
    rvc.add_argument("--warmup-rvc-count", type=int, default=2,
                     help="Number of dummy inference calls to run BEFORE "
                          "opening the audio stream. Avoids the cold-start "
                          "stall (~30 s first call). Set 0 to disable.")
    rvc.add_argument("--rvc-queue-ms", type=float, default=6000.0,
                     help="Per-direction queue capacity in milliseconds. "
                          "Identity-era default (640 ms) is too small for "
                          "1 s chunks; output blocks get dropped. Default "
                          "6000 ms = ~600 blocks at 48 kHz / 480-frame "
                          "blocks.")
    rvc.add_argument("--rvc-prebuffer-ms", type=float, default=None,
                     help="Silence prebuffered into the OutputStream before "
                          "real audio arrives. Default: 2 * chunk_ms. Adds "
                          "this much latency but hides startup underruns.")
    rvc.add_argument("--drop-stale-input",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="If inference falls behind, drop older queued "
                          "chunks and keep only the latest. Reduces "
                          "perceived latency drift. Default: on for RVC.")
    rvc.add_argument("--rvc-context-ms", type=float, default=200.0,
                     help="Stage 3 input-side LEFT context fed to the "
                          "model before each chunk. Default 200 ms. The "
                          "engine sees [previous_input_tail, chunk]; the "
                          "model's output for the context region is "
                          "trimmed away proportionally so the chunk's "
                          "EMITTED duration stays exactly chunk_ms (no "
                          "timeline drift). Eliminates per-chunk cold-"
                          "start in HuBERT / F0 / index without any "
                          "output-side blending. Set 0 to disable. "
                          "Engineering knob, not a voice tuning knob.")

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


_VOICE_IDENTITY_FIELDS = (
    # (cli_arg_name, profile_attr_name, default_when_neither)
    ("model_path",  "model_path",  None),
    ("index_path",  "index_path",  None),
    ("hubert_path", "hubert_path", None),
    ("rmvpe_path",  "rmvpe_path",  None),
    ("f0_method",   "f0_method",   "rmvpe"),
    ("index_rate",  "index_rate",  0.5),
    ("protect",     "protect",     0.33),
    ("filter_radius", "filter_radius", 3),
    ("rms_mix_rate",  "rms_mix_rate",  0.25),
    ("pitch_shift",   "pitch_shift",   0),
)


def _resolve_voice_identity(args: argparse.Namespace, profile) -> dict:
    """Resolve voice-identity params from (profile, CLI override) pairs.

    CLI override wins when set; a divergence from the profile prints a
    "developer override" warning to stderr. The point of the warning is
    to make tuning-by-flag visibly non-default in dev logs.
    """
    resolved: dict = {}
    for cli_name, prof_name, default in _VOICE_IDENTITY_FIELDS:
        cli_val = getattr(args, cli_name)
        prof_val = getattr(profile, prof_name) if profile is not None else None
        if cli_val is not None:
            if (
                profile is not None
                and prof_val is not None
                and cli_val != prof_val
            ):
                print(
                    f"WARNING: developer override: profile "
                    f"{prof_name}={prof_val!r} replaced by CLI "
                    f"--{cli_name.replace('_', '-')} = {cli_val!r}. "
                    "Model-faithful behaviour is to omit this flag.",
                    file=sys.stderr,
                )
            resolved[prof_name] = cli_val
        elif prof_val is not None:
            resolved[prof_name] = prof_val
        else:
            resolved[prof_name] = default
    return resolved


def _cmd_mode_rvc(args: argparse.Namespace) -> int:
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
        print(
            f"loaded model profile: name={profile.name!r} "
            f"model_path={profile.model_path!r}"
        )

    voice = _resolve_voice_identity(args, profile)

    if not voice["model_path"]:
        print(
            "error: --mode rvc requires --model-profile PATH or "
            "--model-path /path/to/model.pth. Place local model files "
            "under models/ (gitignored).",
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

    # Stage 2D default: resample_sr=0 (model's native SR) and let the
    # worker linear-resample to the stream SR. The Stage 2D benchmark
    # showed that asking infer_rvc_python to resample internally added
    # ~230 ms/chunk on kiki @ chunk_ms=1000, pushing rt > 1. Worker-side
    # linear resample costs ~1 ms.
    if args.resample_sr is not None:
        resample_sr = args.resample_sr
    elif profile is not None:
        resample_sr = profile.resample_sr
    else:
        resample_sr = 0

    rvc_config = RvcEngineConfig(
        model_path=voice["model_path"],
        index_path=voice["index_path"],
        backend=args.backend,
        f0_method=voice["f0_method"],
        index_rate=voice["index_rate"],
        protect=voice["protect"],
        filter_radius=voice["filter_radius"],
        rms_mix_rate=voice["rms_mix_rate"],
        pitch_shift=voice["pitch_shift"],
        sample_rate=config.sample_rate,
        resample_sr=resample_sr,
        device=args.device,
        force_cpu=args.force_cpu,
        hubert_path=voice["hubert_path"],
        rmvpe_path=voice["rmvpe_path"],
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

    if args.warmup_rvc_count > 0:
        warmup_chunk_samples = max(1, int(round(args.chunk_ms / 1000.0 * config.sample_rate)))
        print(
            f"warming up RVC ({args.warmup_rvc_count} calls, "
            f"chunk_samples={warmup_chunk_samples} at {config.sample_rate} Hz)..."
        )
        try:
            timings = engine.warmup(
                warmup_chunk_samples, config.sample_rate, args.warmup_rvc_count
            )
            for i, ms in enumerate(timings):
                print(f"  warmup #{i + 1}: {ms:.0f} ms")
        except (RvcInferenceError, ValueError) as exc:
            print(f"warning: warmup failed (continuing): {exc}", file=sys.stderr)

    try:
        run_rvc_stream(
            config,
            engine=engine,
            chunk_ms=args.chunk_ms,
            crossfade_ms=args.crossfade_ms,
            duration_seconds=args.duration_seconds,
            allow_virtual_cable_input=args.allow_virtual_cable_input,
            rvc_queue_ms=args.rvc_queue_ms,
            rvc_prebuffer_ms=args.rvc_prebuffer_ms,
            drop_stale_input=args.drop_stale_input,
            context_ms=args.rvc_context_ms,
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
