"""CLI entry point for the realtime RVC voice changer.

The one job: ``system default mic -> RVC inference -> CABLE Input`` so a
downstream app (Discord / OBS / recorder) selecting ``CABLE Output`` as
its microphone hears your voice converted to the model's voice.

Usage::

    # list devices (which mic is the system default, where CABLE Input is)
    python -m meloie.main --list-devices

    # validate a runtime config
    python -m meloie.main --check-config config/runtime.example.json

    # run: system default mic -> model A (v2) -> CABLE Input
    python -m meloie.main --config config/runtime.example.json \\
        --model-profile config/model_profiles/A.json \\
        --device cuda
    # (or just double-click run_A_direct.bat)

Voice identity comes entirely from the model profile (the trained model
defines the voice). The CLI only takes engineering knobs (device, block
size, queue/latency, which mic). There are no voice-shaping flags.

This is a v2-only build: the sole realtime engine is the 'direct' Applio
persistent-buffer engine (run in .venv-applio). A v1 / 256-dim model is
rejected at load.

Importing this module must NOT open audio devices and must NOT import
sounddevice / torch / the meloie.core inference stack. Those imports live
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
    voice.add_argument("--pitch", type=int, default=None, metavar="SEMITONES",
                       help="Transpose (变调) in semitones applied to the input "
                            "F0 before conversion -- THE main creative knob. A "
                            "female model typically needs about +12 for a male "
                            "voice. This "
                            "conditions the model's input pitch (not an output "
                            "pitch-shift). Overrides the profile's pitch_shift; "
                            "default: the profile's value.")
    voice.add_argument("--sid", type=int, default=0,
                       help="Speaker id for multi-speaker models (selects which "
                            "TRAINED voice the model uses -- the model defining the "
                            "voice, not runtime reshaping). Default 0; validated "
                            "against the model's speaker count.")

    eng = parser.add_argument_group("Engineering knobs (stability / latency)")
    # No engine selector: this is a v2-only build with a single realtime engine
    # (the 'direct' Applio persistent-buffer engine, run in .venv-applio). The
    # voice flows mic -> StreamingRvcEngine -> CABLE Input unconditionally.
    eng.add_argument("--direct-block-ms", type=float, default=250.0,
                     help="[engine=direct] output block size (ms). Applio default "
                          "250. Lower = lower latency, more seams.")
    eng.add_argument("--direct-context-ms", type=float, default=2500.0,
                     help="[engine=direct] REAL past audio fed to the encoders each "
                          "block (w-okada '额外推理时长'). Free latency-wise (it is "
                          "past); costs only compute + startup warm-up. Bigger = "
                          "steadier timbre + F0. Default 2500.")
    eng.add_argument("--direct-crossfade-ms", type=float, default=50.0,
                     help="[engine=direct] sin² seam crossfade overlap (ms). The one "
                          "sanctioned output blend; smooths block seams. Applio default 50.")
    eng.add_argument("--direct-embedder", default="contentvec",
                     choices=["contentvec"],
                     help="Embedder model (HuBERT/ContentVec). Only 'contentvec' is "
                          "staged + verified for v2/768-dim. Staged under "
                          "models/embedders.")
    eng.add_argument("--direct-f0", default=None,
                     choices=["rmvpe", "fcpe"],
                     help="Override the profile's F0 method. The realtime engine backs "
                          "'rmvpe' and 'fcpe'; fcpe is smoother + ~30%% faster (the "
                          "run_A_direct launcher bakes '--direct-f0 fcpe'). Default: "
                          "the profile's value.")
    eng.add_argument("--direct-denoise",
                     action=argparse.BooleanOptionalAction, default=False,
                     help="[engine=direct] INPUT-side noise reduction (Applio's "
                          "noisereduce TorchGate) BEFORE conversion, so ambient "
                          "noise is not converted into warbly voice. Input "
                          "conditioning (like pitch transpose), not output reshaping "
                          "-- the model still defines the voice. Default OFF (a clean "
                          "mic / soft speech is never silently degraded).")
    eng.add_argument("--direct-denoise-strength", type=float, default=0.5,
                     help="[engine=direct] denoise prop_decrease 0..1 (1 = most "
                          "aggressive). Higher cleans more but can muffle soft "
                          "speech. Start ~0.5 and tune to your mic.")
    eng.add_argument("--direct-denoise-nonstationary",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="[engine=direct] adapt to time-varying noise (recommended). "
                          "--no-direct-denoise-nonstationary = stationary gating.")
    eng.add_argument("--direct-formant",
                     action=argparse.BooleanOptionalAction, default=False,
                     help="INPUT-side FORMANT / gender shift (性别因子) before "
                          "conversion: moves the spectral envelope (vocal-tract / "
                          "gender) WITHOUT changing pitch -- input conditioning like "
                          "--pitch, not output reshaping. Auto-on if --direct-formant-"
                          "timbre/qfrency != 1.0. Default off.")
    eng.add_argument("--direct-formant-timbre", type=float, default=1.0,
                     help="Formant gender knob: >1 = formants up (brighter/feminine), "
                          "<1 = down (deeper/masculine), 1.0 = none. Tune by ear.")
    eng.add_argument("--direct-formant-qfrency", type=float, default=1.0,
                     help="Formant cepstral detail (Applio default 1.0).")
    eng.add_argument("--direct-autotune",
                     action=argparse.BooleanOptionalAction, default=False,
                     help="INPUT-side F0 autotune: snap the detected pitch to the "
                          "nearest semitone before conversion (creative/robotic "
                          "stylization). Input conditioning, default off.")
    eng.add_argument("--direct-autotune-strength", type=float, default=1.0,
                     help="Autotune blend 0..1 (1 = full snap).")
    eng.add_argument("--direct-auto-pitch",
                     action=argparse.BooleanOptionalAction, default=False,
                     help="INPUT-side auto pitch-shift: derive the transpose from the "
                          "input's median F0 toward --direct-auto-pitch-threshold "
                          "(a smart auto-version of --pitch, clamped ±12). Adds to "
                          "--pitch. Default off.")
    eng.add_argument("--direct-auto-pitch-threshold", type=float, default=155.0,
                     help="Target F0 (Hz) for --direct-auto-pitch (~155 male / "
                          "~255 female baseline).")
    eng.add_argument("--direct-protect", type=float, default=None,
                     help="Protect voiceless consonants / breath (0..0.5): higher "
                          "preserves more of the source's unvoiced detail (less "
                          "artifacting) at the cost of a touch less conversion. "
                          "Default: the profile's value (A: 0.33).")
    eng.add_argument("--direct-silence-dbfs", type=float, default=None,
                     help="Silence gate (响应阈值 / w-okada silentThreshold): below "
                          "this input level (dBFS) emit clean silence and skip GPU "
                          "inference. Input-side (decides WHETHER to convert; never "
                          "reshapes output). Default OFF; opt in with e.g. -50. Keep "
                          "it well below your soft-speech level.")
    eng.add_argument("--direct-silence-hangover-ms", type=float, default=250.0,
                     help="Keep converting this long after the last loud block so "
                          "soft trailing syllables are not clipped by the gate.")
    eng.add_argument("--device", default="auto",
                     choices=["auto", "cpu", "cuda"],
                     help="Inference device. 'auto' = cuda if available else cpu.")
    eng.add_argument("--rvc-queue-ms", type=float, default=6000.0,
                     help="Per-direction queue capacity in ms.")
    eng.add_argument("--rvc-prebuffer-ms", type=float, default=None,
                     help="Output silence prebuffer before first real audio = the "
                          "standing output latency. Default 800 ms (an absolute "
                          "cushion covering one inference spike + one output "
                          "burst). Lower = less latency but more underruns.")
    eng.add_argument("--drop-stale-input",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="If inference falls behind, drop oldest blocks to "
                          "keep latency bounded. Default: on.")

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
            "example: python -m meloie.main --config config/runtime.example.json "
            "--model-profile config/model_profiles/A.json --device cuda",
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

    # v2-only build: the direct (Applio persistent-buffer) engine is the sole path.
    return _run_direct(args, config, model_path, profile, pick)


def _run_direct(args, config, model_path, profile, pick) -> int:
    """Path-A direct engine: build + run the stateful StreamingRvcEngine."""
    from .engine.streaming_engine import (
        StreamingEngineConfig,
        StreamingEngineError,
        StreamingRvcEngine,
    )
    from .audio.devices import FeedbackLoopRisk
    from .audio.streaming_stream import run_streaming_stream
    from .engine.model_profile import find_default_index, models_root_for

    index_rate = float(pick("index_rate", 0.0))
    # Only load the index if it actually contributes (index_rate > 0). The path
    # is the explicit CLI/profile index, else the .pth's own default .index
    # (same-stem first, else first; recursive under models/).
    if index_rate > 0:
        index_path = (args.index_path or pick("index_path", "")
                      or find_default_index(model_path, models_root_for(model_path)))
    else:
        index_path = ""
    pitch_shift = args.pitch if args.pitch is not None else int(pick("pitch_shift", 0))

    # INPUT-side formant / gender: CLI overrides the profile; enable when either
    # the bool is set OR a non-1.0 timbre/qfrency is given (profile or CLI).
    f_timbre = (args.direct_formant_timbre if args.direct_formant_timbre != 1.0
                else float(pick("formant_timbre", 1.0)))
    f_qfrency = (args.direct_formant_qfrency if args.direct_formant_qfrency != 1.0
                 else float(pick("formant_qfrency", 1.0)))
    formant_on = bool(args.direct_formant) or f_timbre != 1.0 or f_qfrency != 1.0

    scfg = StreamingEngineConfig(
        model_path=model_path,
        index_path=index_path or "",
        f0_method=(args.direct_f0 or pick("f0_method", "rmvpe")),
        embedder=args.direct_embedder,
        pitch_shift=int(pitch_shift),
        index_rate=index_rate,
        protect=float(args.direct_protect if args.direct_protect is not None
                      else pick("protect", 0.33)),
        sid=int(args.sid),
        stream_sr=int(config.sample_rate),
        block_ms=float(args.direct_block_ms),
        context_ms=float(args.direct_context_ms),
        crossfade_ms=float(args.direct_crossfade_ms),
        denoise=bool(args.direct_denoise),
        denoise_strength=float(args.direct_denoise_strength),
        denoise_nonstationary=bool(args.direct_denoise_nonstationary),
        formant_shift=formant_on,
        formant_qfrency=float(f_qfrency),
        formant_timbre=float(f_timbre),
        f0_autotune=bool(args.direct_autotune),
        f0_autotune_strength=float(args.direct_autotune_strength),
        proposed_pitch=bool(args.direct_auto_pitch),
        proposed_pitch_threshold=float(args.direct_auto_pitch_threshold),
        silence_threshold_dbfs=args.direct_silence_dbfs,
        silence_hangover_ms=float(args.direct_silence_hangover_ms),
        device=args.device,
    )
    print(f"voice params (direct): pitch={scfg.pitch_shift:+d}  index_rate={scfg.index_rate}  "
          f"f0={scfg.f0_method}  protect={scfg.protect}  embedder={scfg.embedder}  sid={scfg.sid}")
    print(f"block_ms={scfg.block_ms:.0f}  context_ms={scfg.context_ms:.0f}  "
          f"crossfade_ms={scfg.crossfade_ms:.0f}  "
          f"denoise={('ON @ ' + format(scfg.denoise_strength, '.2f')) if scfg.denoise else 'OFF'}  "
          f"formant={('ON timbre=' + format(scfg.formant_timbre, '.2f')) if scfg.formant_shift else 'OFF'}  "
          f"autotune={'ON' if scfg.f0_autotune else 'OFF'}  "
          f"auto_pitch={'ON' if scfg.proposed_pitch else 'OFF'}  "
          f"silence_gate={(format(scfg.silence_threshold_dbfs, '.0f') + ' dBFS') if scfg.silence_threshold_dbfs is not None else 'OFF'}")
    print(f"loading direct (Applio persistent-buffer) engine, device={args.device} ...")
    engine = StreamingRvcEngine(scfg)
    try:
        engine.load()
    except StreamingEngineError as exc:
        print(f"error: direct engine load failed: {exc}", file=sys.stderr)
        return 11
    print(f"direct engine loaded. device={engine.resolved_device} "
          f"cuda={engine.cuda_device_name or '(n/a)'} precision={engine.resolved_precision} "
          f"block_frame={engine.block_frame} tgt_sr={engine.tgt_sr}")

    try:
        run_streaming_stream(
            config,
            engine=engine,
            duration_seconds=args.duration_seconds,
            allow_virtual_cable_input=args.allow_virtual_cable_input,
            rvc_queue_ms=args.rvc_queue_ms,
            rvc_prebuffer_ms=args.rvc_prebuffer_ms,
            drop_stale_input=args.drop_stale_input,
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
