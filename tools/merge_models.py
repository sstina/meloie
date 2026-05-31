"""Offline RVC model-merge CLI: blend 2+ v2 .pth models into one hybrid voice.

捏脸 / voice-morphing. Produces a STANDARD v2 .pth that ``StreamingRvcEngine`` loads
as-is (v2 guard, contentvec, NSF) -> contract-safe: the merged MODEL defines the
voice, the realtime runtime never reshapes output. Only identical-architecture v2
models merge (same sampling rate, f0 flag, vocoder, weight key-set + per-key shape).

No torch import at module load (kept lazy in :func:`main`) so ``import
tools.merge_models`` is cheap and side-effect free. Runs in .venv-applio. Example::

    python -m tools.merge_models \\
        --models models/A.pth models/C.pth --weights 0.6 0.4 \\
        --output models/AC_mix.pth --name "A+C" --write-profile --profile-pitch-shift 12

Exit codes: 0 ok / 2 usage+containerization / 3 incompatible models / 4 file not
found / 5 save or verify failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

_THIS = os.path.abspath(__file__)
RVC_ROOT = os.path.dirname(os.path.dirname(_THIS))   # .../RVC


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tvoice-merge-models",
        description="Blend 2+ identical-architecture v2 RVC .pth models into one "
                    "hybrid voice (weighted average of the model weights). The "
                    "result is a standard .pth the realtime engine loads as-is.",
    )
    p.add_argument("--models", nargs="+", required=True, metavar="PTH",
                   help="2+ v2 .pth model paths (relative to RVC/ or absolute).")
    p.add_argument("--weights", nargs="+", type=float, default=None, metavar="W",
                   help="Per-model blend strengths (normalized to sum 1). "
                        "Default: equal. Length must match --models.")
    p.add_argument("--output", required=True, metavar="PTH",
                   help="Output .pth path (MUST be under RVC/ -- no C: writes).")
    p.add_argument("--name", default=None, help="Display name (used by --write-profile).")
    p.add_argument("--write-profile", action="store_true",
                   help="Also write config/model_profiles/<out-stem>.json so the GUI lists it.")
    p.add_argument("--force", action="store_true",
                   help="Allow overwriting an existing output / profile.")
    p.add_argument("--profile-f0", default="rmvpe", choices=["rmvpe", "fcpe"],
                   help="f0_method written into the profile (default rmvpe).")
    p.add_argument("--profile-pitch-shift", type=int, default=0, metavar="SEMITONES",
                   help="pitch_shift written into the profile (default 0; tune 10-14 "
                        "for high/female voices).")
    p.add_argument("--verify-load", action="store_true",
                   help="After saving, load the merged model through the engine to "
                        "prove it passes the v2 guard (slower; needs the GPU stack).")
    return p


def _under_root(path: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), RVC_ROOT]) == RVC_ROOT
    except ValueError:
        return False   # e.g. a different drive on Windows


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8_stdio()
    args = _build_parser().parse_args(argv)

    # All relative paths resolve against RVC/ (like the engine / profiles).
    if os.path.abspath(os.getcwd()) != RVC_ROOT:
        os.chdir(RVC_ROOT)

    if len(args.models) < 2:
        print("error: need at least 2 --models to merge", file=sys.stderr)
        return 2
    if args.weights is not None and len(args.weights) != len(args.models):
        print(f"error: --weights ({len(args.weights)}) must match --models "
              f"({len(args.models)})", file=sys.stderr)
        return 2
    if not _under_root(args.output):
        print(f"error: --output must be under {RVC_ROOT} (no C: writes); got "
              f"{os.path.abspath(args.output)}", file=sys.stderr)
        return 2
    if os.path.exists(args.output) and not args.force:
        print(f"error: {args.output} exists; pass --force to overwrite", file=sys.stderr)
        return 2

    try:
        import torch
    except Exception as exc:
        print(f"error: torch unavailable (run in .venv-applio): {exc}", file=sys.stderr)
        return 5
    from src.engine.model_merge import MergeError, merge_checkpoints, weight_dict

    for m in args.models:
        if not os.path.isfile(m):
            print(f"error: model not found: {m}", file=sys.stderr)
            return 4

    try:
        merged_cpt, common, alphas = merge_checkpoints(
            args.models,
            args.weights if args.weights is not None else [1.0] * len(args.models),
        )
    except MergeError as exc:
        print(f"merge error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"error: failed to load/blend models: {exc}", file=sys.stderr)
        return 4

    recipe = " + ".join(
        f"{os.path.basename(m)}*{a:.3f}" for m, a in zip(args.models, alphas)
    )
    print(f"merging (v2, sr={common['sr']}, f0={common['f0']}, "
          f"{len(merged_cpt['weight'])} tensors): {recipe}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        torch.save(merged_cpt, args.output)
    except Exception as exc:
        print(f"error: failed to save {args.output}: {exc}", file=sys.stderr)
        return 5

    # lightweight integrity re-load (fast; not the full engine)
    try:
        back = torch.load(args.output, map_location="cpu", weights_only=True)
        w = weight_dict(back)
        assert str(back.get("version", "")).lower() == "v2", "merged version != v2"
        assert "emb_g.weight" in w, "merged weight missing emb_g.weight"
    except Exception as exc:
        print(f"error: merged file failed its integrity check: {exc}", file=sys.stderr)
        return 5
    print(f"wrote {args.output}")

    if args.verify_load:
        try:
            from src.engine.streaming_engine import (
                StreamingEngineConfig, StreamingRvcEngine,
            )
            eng = StreamingRvcEngine(StreamingEngineConfig(
                model_path=args.output, f0_method=args.profile_f0,
                embedder="contentvec", device="cuda",
            ))
            eng.load()
            print(f"verify-load OK: v2 guard passed, num_speakers={eng.num_speakers}")
        except Exception as exc:
            print(f"error: --verify-load failed: {exc}", file=sys.stderr)
            return 5

    if args.write_profile:
        rc = _write_profile(args)
        if rc != 0:
            return rc

    return 0


def _write_profile(args) -> int:
    import json

    stem = Path(args.output).stem
    prof_path = os.path.join(RVC_ROOT, "config", "model_profiles", f"{stem}.json")
    if os.path.exists(prof_path) and not args.force:
        print(f"error: profile {prof_path} exists; pass --force to overwrite",
              file=sys.stderr)
        return 2
    rel = os.path.relpath(os.path.abspath(args.output), RVC_ROOT).replace("\\", "/")
    profile = {
        "name": args.name or stem,
        "model_path": rel,
        "f0_method": args.profile_f0,
        "index_rate": 0.0,
        "pitch_shift": int(args.profile_pitch_shift),
        "notes": "merged model; index_rate 0 (no shared index). Tune pitch_shift by ear.",
    }
    try:
        os.makedirs(os.path.dirname(prof_path), exist_ok=True)
        with open(prof_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"error: failed to write profile {prof_path}: {exc}", file=sys.stderr)
        return 5
    print(f"wrote profile {prof_path}")
    if int(args.profile_pitch_shift) == 0:
        print("  reminder: pitch_shift=0 -> tune it (10-14 for high/female voices) "
              "or the voice may sound neutral / 电音.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
