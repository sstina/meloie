"""Offline RVC sanity CLI: read WAV -> RVC inference -> write WAV.

Purpose: verify a (.pth, .index, params) tuple end-to-end on a file
without any realtime audio stream. The realtime path uses the same
``RvcEngine``; if this command works on a known phrase, the realtime
``--mode rvc`` should work too.

Hard rules baked in:

* No torch / infer_rvc_python imports at module load. The engine
  loads them only when the user actually calls this CLI.
* The output WAV is written ONLY after a successful inference; a
  failed inference leaves no half-written artifact.
* Missing backend / model produces an actionable error with a clear
  hint to ``models/local/`` (which is gitignored).
* This CLI does NOT write any cache, log, or report unless the user
  explicitly chooses an output path under their own control.

Example::

    python -m tools.offline_infer \\
        --input-wav voices/sample.wav \\
        --output-wav voices/sample_rvc.wav \\
        --model-path models/local/MyVoice.pth \\
        --index-path models/local/MyVoice.index \\
        --f0-method rmvpe --index-rate 0.5 --protect 0.33
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tvoice-offline-infer",
        description="Run RVC inference on a WAV file (offline sanity check). "
                    "Use --model-profile for the recommended invocation; the "
                    "voice-identity flags below are developer overrides.",
    )
    p.add_argument("--input-wav", required=True)
    p.add_argument("--output-wav", required=True)
    p.add_argument("--model-profile", default=None,
                   help="Path to a model profile JSON (see "
                        "config/model_profiles/kiki.example.json). "
                        "Supplies the voice identity. Recommended.")
    p.add_argument("--model-path", default=None,
                   help="DEVELOPER OVERRIDE: path to RVC .pth model.")
    p.add_argument("--index-path", default=None,
                   help="DEVELOPER OVERRIDE: path to .index file.")
    p.add_argument("--hubert-path", default=None,
                   help="DEVELOPER OVERRIDE: path to hubert_base.pt.")
    p.add_argument("--rmvpe-path", default=None,
                   help="DEVELOPER OVERRIDE: path to rmvpe.pt.")
    p.add_argument("--backend", default="infer_rvc_python")
    p.add_argument("--f0-method", default=None,
                   help="DEVELOPER OVERRIDE.")
    p.add_argument("--index-rate", type=float, default=None,
                   help="DEVELOPER OVERRIDE.")
    p.add_argument("--protect", type=float, default=None,
                   help="DEVELOPER OVERRIDE.")
    p.add_argument("--filter-radius", type=int, default=None,
                   help="DEVELOPER OVERRIDE.")
    p.add_argument("--rms-mix-rate", type=float, default=None,
                   help="DEVELOPER OVERRIDE.")
    p.add_argument("--pitch-shift", type=int, default=None,
                   help="DEVELOPER OVERRIDE.")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "directml_experimental"],
                   help="Inference device. 'auto' picks cuda if available, "
                        "otherwise cpu.")
    p.add_argument("--force-cpu", action="store_true",
                   help="DEPRECATED: equivalent to --device cpu.")
    p.add_argument("--resample-sr", type=int, default=None,
                   help="Ask the RVC backend to resample its output to this "
                        "sample rate. Default: profile's resample_sr or 0 "
                        "(keep model's natural rate).")
    return p


_OFFLINE_VOICE_FIELDS = (
    ("model_path",  "model_path",  None),
    ("index_path",  "index_path",  None),
    ("hubert_path", "hubert_path", None),
    ("rmvpe_path",  "rmvpe_path",  None),
    ("f0_method",   "f0_method",   "rmvpe"),
    ("index_rate",  "index_rate",  0.5),
    ("protect",     "protect",     0.33),
    ("filter_radius", "filter_radius", 3),
    ("rms_mix_rate",  "rms_mix_rate",  1.0),
    ("pitch_shift",   "pitch_shift",   0),
)


def _resolve_offline_voice(args, profile) -> dict:
    """Resolve voice-identity params for offline inference.

    Same semantics as the realtime CLI: CLI override wins and a
    divergence prints a clear developer-override warning.
    """
    resolved: dict = {}
    for cli_name, prof_name, default in _OFFLINE_VOICE_FIELDS:
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


def main(argv: Optional[List[str]] = None) -> int:
    for _s in (sys.stdout, sys.stderr):  # tolerate non-GBK names on CN locale
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    args = _build_parser().parse_args(argv)

    in_path = Path(args.input_wav)
    out_path = Path(args.output_wav)

    if not in_path.exists():
        print(f"error: input WAV not found: {in_path}", file=sys.stderr)
        return 2

    # Lazy imports — RVC stack only touched here.
    from src.audio.wav_io import read_wav_mono_float32, write_wav_float32
    from src.engine.model_profile import ModelProfileError, load_model_profile
    from src.engine.rvc_engine import (
        DependencyMissingError,
        ModelLoadError,
        RvcEngine,
        RvcEngineConfig,
        RvcInferenceError,
    )
    from src.safety.guard import dbfs_peak, dbfs_rms, scrub_nan_inf

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

    voice = _resolve_offline_voice(args, profile)

    if not voice["model_path"]:
        print(
            "error: requires --model-profile PATH or --model-path "
            "/path/to/model.pth.",
            file=sys.stderr,
        )
        return 2

    model_path = Path(voice["model_path"])
    if not model_path.exists():
        print(
            f"error: model not found: {model_path}\n"
            "Tip: place local model files under models/ (gitignored) "
            "and pass the full path.",
            file=sys.stderr,
        )
        return 3
    if voice["index_path"] and not Path(voice["index_path"]).exists():
        print(f"error: index file not found: {voice['index_path']}", file=sys.stderr)
        return 3

    print(f"reading {in_path} ...")
    audio, in_sr = read_wav_mono_float32(str(in_path))
    duration = audio.size / float(in_sr)
    print(
        f"input : {audio.size} samples @ {in_sr} Hz  ({duration:.2f} s)  "
        f"peak={dbfs_peak(audio):.2f} dBFS  rms={dbfs_rms(audio):.2f} dBFS"
    )

    if args.resample_sr is not None:
        resample_sr = args.resample_sr
    elif profile is not None:
        resample_sr = profile.resample_sr
    else:
        resample_sr = 0

    cfg = RvcEngineConfig(
        model_path=str(model_path),
        index_path=voice["index_path"],
        backend=args.backend,
        f0_method=voice["f0_method"],
        index_rate=voice["index_rate"],
        protect=voice["protect"],
        filter_radius=voice["filter_radius"],
        rms_mix_rate=voice["rms_mix_rate"],
        pitch_shift=voice["pitch_shift"],
        resample_sr=resample_sr,
        device=args.device,
        force_cpu=args.force_cpu,
        hubert_path=voice["hubert_path"],
        rmvpe_path=voice["rmvpe_path"],
    )
    engine = RvcEngine(cfg)

    print(f"loading RVC backend={args.backend} device={args.device} ...")
    try:
        engine.load()
    except DependencyMissingError as exc:
        print(f"error: dependency missing: {exc}", file=sys.stderr)
        return 10
    except ModelLoadError as exc:
        print(f"error: model load failed: {exc}", file=sys.stderr)
        return 11
    print(
        f"engine loaded. resolved_device={engine.resolved_device} "
        f"cuda_device={engine.cuda_device_name or '(n/a)'}"
    )

    print(f"running RVC inference on {audio.size} samples ...")
    try:
        result, out_sr = engine.infer_array(audio, in_sr)
    except RvcInferenceError as exc:
        print(f"error: inference failed: {exc}", file=sys.stderr)
        return 12

    scrub = scrub_nan_inf(result)
    if scrub.replaced_count:
        print(f"warning: scrubbed {scrub.replaced_count} NaN/Inf samples")
        result = scrub.audio

    out_dur = result.size / float(out_sr)
    print(
        f"output: {result.size} samples @ {out_sr} Hz  ({out_dur:.2f} s)  "
        f"peak={dbfs_peak(result):.2f} dBFS  rms={dbfs_rms(result):.2f} dBFS"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav_float32(str(out_path), result, out_sr)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
