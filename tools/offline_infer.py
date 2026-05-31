"""Offline RVC sanity CLI (v2 / direct engine): WAV -> StreamingRvcEngine -> WAV.

Purpose: verify a (.pth, .index, params) tuple end-to-end on a file without
opening any realtime audio device. It drives the SAME persistent-buffer engine
the realtime path uses (``StreamingRvcEngine.process_block`` in a loop), so if an
offline render of a phrase sounds right, the realtime run will too. v2-only: a
v1 / 256-dim model is rejected at load (the engine's version guard).

Hard rules baked in:

* No torch / Applio-stack imports at module load. The engine loads them only
  when the user actually calls this CLI (keeps `import tools.offline_infer` cheap
  and side-effect-free).
* The output WAV is written ONLY after a successful inference; a failed
  inference leaves no half-written artifact.
* This CLI writes no cache/log/report — only the output WAV the user names.

Runs in .venv-applio (the v2 runtime). Example::

    python -m tools.offline_infer \\
        --input-wav voices/sample.wav \\
        --output-wav voices/sample_A.wav \\
        --model-profile config/model_profiles/A.json \\
        --f0-method fcpe --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

STREAM_SR = 48000  # the engine runs at this stream SR (mirrors the realtime path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tvoice-offline-infer",
        description="Run v2 RVC inference on a WAV file (offline sanity check) "
                    "via the same direct/persistent-buffer engine as realtime. "
                    "Use --model-profile for the recommended invocation; the "
                    "voice-identity flags below are developer overrides.",
    )
    p.add_argument("--input-wav", required=True)
    p.add_argument("--output-wav", required=True)
    p.add_argument("--model-profile", default=None,
                   help="Path to a model profile JSON (see "
                        "config/model_profiles/A.json). Supplies the voice "
                        "identity. Recommended.")
    p.add_argument("--model-path", default=None,
                   help="DEVELOPER OVERRIDE: path to RVC v2 .pth model.")
    p.add_argument("--index-path", default=None,
                   help="DEVELOPER OVERRIDE: path to .index file (used only when "
                        "index-rate > 0).")
    p.add_argument("--f0-method", default=None,
                   choices=["rmvpe", "fcpe"],
                   help="F0 estimator. Default: the profile's value. The engine "
                        "backs only rmvpe / fcpe; fcpe is the smoother + faster "
                        "choice on this stack.")
    p.add_argument("--embedder", default="contentvec",
                   help="Embedder model name (contentvec for v2 / 768-dim).")
    p.add_argument("--index-rate", type=float, default=None,
                   help="DEVELOPER OVERRIDE (0..1). Default: the profile's value. "
                        "The index is loaded only when this is > 0.")
    p.add_argument("--protect", type=float, default=None,
                   help="DEVELOPER OVERRIDE (0..0.5).")
    p.add_argument("--pitch-shift", type=int, default=None, metavar="SEMITONES",
                   help="Transpose applied to the input F0 before conversion. "
                        "Default: the profile's value.")
    p.add_argument("--block-ms", type=float, default=250.0,
                   help="Engine output block size (ms).")
    p.add_argument("--context-ms", type=float, default=2500.0,
                   help="Real past audio fed to the encoders each block (ms). "
                        "Matches the realtime default (2500) so an offline render "
                        "predicts the live result.")
    p.add_argument("--crossfade-ms", type=float, default=50.0,
                   help="sin² seam crossfade overlap (ms).")
    p.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=False,
                   help="Input-side noise reduction before conversion (default off).")
    p.add_argument("--denoise-strength", type=float, default=0.5,
                   help="Denoise prop_decrease 0..1 (1 = most aggressive).")
    p.add_argument("--formant", action=argparse.BooleanOptionalAction, default=False,
                   help="Input-side formant/gender shift (性别因子); pitch untouched. "
                        "Default off (auto-on if --formant-timbre/qfrency != 1.0).")
    p.add_argument("--formant-timbre", type=float, default=1.0,
                   help="Gender knob: >1 brighter/feminine, <1 deeper/masculine, 1.0 = off.")
    p.add_argument("--formant-qfrency", type=float, default=1.0,
                   help="Formant cepstral detail (default 1.0).")
    p.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=False,
                   help="Input-side F0 autotune (snap to nearest semitone).")
    p.add_argument("--auto-pitch", action=argparse.BooleanOptionalAction, default=False,
                   help="Input-side auto pitch-shift from median F0.")
    p.add_argument("--sid", type=int, default=0,
                   help="Speaker id for multi-speaker models (default 0).")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Inference device. 'auto' picks cuda if available, else cpu.")
    p.add_argument("--trim-warmup", action=argparse.BooleanOptionalAction, default=True,
                   help="Drop the leading warm-up region (the engine emits zeros "
                        "while its context buffer fills). Default: on.")
    return p


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

    # Lazy imports — the heavy v2 / Applio stack is only touched here.
    import numpy as np
    import librosa
    from src.audio.wav_io import read_wav_mono_float32, write_wav_float32
    from src.engine.model_profile import ModelProfileError, load_model_profile
    from src.engine.streaming_engine import (
        StreamingEngineConfig,
        StreamingEngineError,
        StreamingRvcEngine,
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

    def pick(attr, default):
        return getattr(profile, attr) if profile is not None else default

    model_path = args.model_path or (profile.model_path if profile else None)
    if not model_path:
        print(
            "error: requires --model-profile PATH or --model-path "
            "/path/to/model.pth.",
            file=sys.stderr,
        )
        return 2
    if not Path(model_path).exists():
        print(
            f"error: model not found: {model_path}\n"
            "Tip: place local model files under models/ (gitignored).",
            file=sys.stderr,
        )
        return 3

    # Index is loaded only when it contributes (index_rate > 0) — same as realtime.
    index_rate = float(args.index_rate if args.index_rate is not None
                       else pick("index_rate", 0.0))
    index_path = (args.index_path or pick("index_path", "")) if index_rate > 0 else ""
    if index_path and not Path(index_path).exists():
        print(f"error: index file not found: {index_path}", file=sys.stderr)
        return 3
    pitch_shift = (args.pitch_shift if args.pitch_shift is not None
                   else int(pick("pitch_shift", 0)))

    scfg = StreamingEngineConfig(
        model_path=str(model_path),
        index_path=index_path or "",
        f0_method=(args.f0_method or pick("f0_method", "rmvpe")),
        embedder=args.embedder,
        pitch_shift=int(pitch_shift),
        index_rate=index_rate,
        protect=float(args.protect if args.protect is not None else pick("protect", 0.33)),
        sid=int(args.sid),
        stream_sr=STREAM_SR,
        block_ms=float(args.block_ms),
        context_ms=float(args.context_ms),
        crossfade_ms=float(args.crossfade_ms),
        denoise=bool(args.denoise),
        denoise_strength=float(args.denoise_strength),
        formant_shift=(bool(args.formant) or args.formant_timbre != 1.0
                       or args.formant_qfrency != 1.0),
        formant_qfrency=float(args.formant_qfrency),
        formant_timbre=float(args.formant_timbre),
        f0_autotune=bool(args.autotune),
        proposed_pitch=bool(args.auto_pitch),
        device=("cpu" if args.device == "cpu" else "cuda"),
    )
    print(
        f"voice params: pitch={scfg.pitch_shift:+d}  index_rate={scfg.index_rate}  "
        f"f0={scfg.f0_method}  protect={scfg.protect}  embedder={scfg.embedder}  sid={scfg.sid}  "
        f"denoise={('ON @ ' + format(scfg.denoise_strength, '.2f')) if scfg.denoise else 'OFF'}  "
        f"formant={('ON timbre=' + format(scfg.formant_timbre, '.2f')) if scfg.formant_shift else 'OFF'}"
    )

    print(f"reading {in_path} ...")
    audio, in_sr = read_wav_mono_float32(str(in_path))
    if in_sr != STREAM_SR:
        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=STREAM_SR)
    duration = audio.size / float(STREAM_SR)
    print(
        f"input : {audio.size} samples @ {STREAM_SR} Hz  ({duration:.2f} s)  "
        f"peak={dbfs_peak(audio):.2f} dBFS  rms={dbfs_rms(audio):.2f} dBFS"
    )

    print(f"loading direct (Applio persistent-buffer) engine, device={args.device} ...")
    engine = StreamingRvcEngine(scfg)
    try:
        engine.load()
    except StreamingEngineError as exc:
        print(f"error: engine load failed: {exc}", file=sys.stderr)
        return 11
    print(
        f"engine loaded. device={engine.resolved_device} "
        f"cuda={engine.cuda_device_name or '(n/a)'} "
        f"precision={engine.resolved_precision} tgt_sr={engine.tgt_sr}"
    )

    bf = engine.block_frame
    warmup_samples = int(engine._warmup) * bf  # zeros emitted while the buffer fills
    n_blocks = audio.size // bf
    if n_blocks == 0:
        print(f"error: input too short ({audio.size} samples < one "
              f"{bf}-sample block at block-ms={args.block_ms}).", file=sys.stderr)
        return 2

    print(f"running RVC inference: {n_blocks} blocks of {bf} samples ...")
    out_chunks = []
    for i in range(n_blocks):
        block = audio[i * bf:(i + 1) * bf]
        out_chunks.append(engine.process_block(block, STREAM_SR))
    result = np.concatenate(out_chunks) if out_chunks else np.zeros(0, np.float32)

    if args.trim_warmup and warmup_samples > 0:
        result = result[warmup_samples:]
        print(f"trimmed {warmup_samples} warm-up samples "
              f"(~{warmup_samples * 1000.0 / STREAM_SR:.0f} ms)")

    scrub = scrub_nan_inf(result)
    if scrub.replaced_count:
        print(f"warning: scrubbed {scrub.replaced_count} NaN/Inf samples")
        result = scrub.audio

    out_dur = result.size / float(STREAM_SR)
    print(
        f"output: {result.size} samples @ {STREAM_SR} Hz  ({out_dur:.2f} s)  "
        f"peak={dbfs_peak(result):.2f} dBFS  rms={dbfs_rms(result):.2f} dBFS"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav_float32(str(out_path), result, STREAM_SR)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
