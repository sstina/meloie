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
        description="Run RVC inference on a WAV file (offline sanity check).",
    )
    p.add_argument("--input-wav", required=True)
    p.add_argument("--output-wav", required=True)
    p.add_argument("--model-path", required=True,
                   help="Path to RVC .pth (place local models under models/local/).")
    p.add_argument("--index-path", default=None,
                   help="Path to .index (optional).")
    p.add_argument("--backend", default="infer_rvc_python")
    p.add_argument("--f0-method", default="rmvpe")
    p.add_argument("--index-rate", type=float, default=0.5)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--filter-radius", type=int, default=3)
    p.add_argument("--rms-mix-rate", type=float, default=0.25)
    p.add_argument("--pitch-shift", type=int, default=0)
    p.add_argument("--force-cpu", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    in_path = Path(args.input_wav)
    out_path = Path(args.output_wav)
    model_path = Path(args.model_path)

    if not in_path.exists():
        print(f"error: input WAV not found: {in_path}", file=sys.stderr)
        return 2
    if not model_path.exists():
        print(
            f"error: model not found: {model_path}\n"
            "Tip: place local model files under models/local/ "
            "(gitignored) and pass the full path.",
            file=sys.stderr,
        )
        return 3
    if args.index_path and not Path(args.index_path).exists():
        print(f"error: index file not found: {args.index_path}", file=sys.stderr)
        return 3

    # Lazy imports — RVC stack only touched here.
    from src.audio.wav_io import read_wav_mono_float32, write_wav_float32
    from src.engine.rvc_engine import (
        DependencyMissingError,
        ModelLoadError,
        RvcEngine,
        RvcEngineConfig,
        RvcInferenceError,
    )
    from src.safety.guard import dbfs_peak, dbfs_rms, scrub_nan_inf

    print(f"reading {in_path} ...")
    audio, in_sr = read_wav_mono_float32(str(in_path))
    duration = audio.size / float(in_sr)
    print(
        f"input : {audio.size} samples @ {in_sr} Hz  ({duration:.2f} s)  "
        f"peak={dbfs_peak(audio):.2f} dBFS  rms={dbfs_rms(audio):.2f} dBFS"
    )

    cfg = RvcEngineConfig(
        model_path=str(model_path),
        index_path=args.index_path,
        backend=args.backend,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        protect=args.protect,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        pitch_shift=args.pitch_shift,
        force_cpu=args.force_cpu,
    )
    engine = RvcEngine(cfg)

    print(f"loading RVC backend={args.backend} ...")
    try:
        engine.load()
    except DependencyMissingError as exc:
        print(f"error: dependency missing: {exc}", file=sys.stderr)
        return 10
    except ModelLoadError as exc:
        print(f"error: model load failed: {exc}", file=sys.stderr)
        return 11

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
