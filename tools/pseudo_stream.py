"""Offline 'pseudo-stream' RVC driver — the audit's load-bearing tool.

The realtime ``--mode rvc`` worker does several things between
``engine.infer_array`` returning a model chunk and the audio reaching
``CABLE Input``:

1. per-chunk inference (no input overlap)
2. SR conversion if the model returns its native rate
3. stitched crossfade at chunk boundaries
4. block split + queue plumbing
5. (optional) drop-stale-input policy

A user-reported listening regression vs the offline whole-file pass
needs an apples-to-apples comparison that exercises **only** items
(1)-(3) without any audio hardware, queue scheduling, or RVC fallback
noise. That is what this tool does:

    python -m tools.pseudo_stream \\
        --input-wav test.wav \\
        --output-wav audit_pseudo_stream.wav \\
        --model-profile config/model_profiles/kiki.example.json \\
        --device cuda \\
        --chunk-ms 1000 \\
        --crossfade-ms 0 \\
        --resampler polyphase \\
        --stream-sr 48000

It loads the same model profile the realtime path uses, runs
``engine.infer_array`` chunk-by-chunk in pure-Python (single thread, no
queues), applies the configurable post-model chain, and writes a WAV
that should sound the same as the live ``CABLE Output`` *would*.

What this tool deliberately does NOT do:

* open any audio device
* spawn a worker thread
* apply identity fallback on inference errors (errors propagate so the
  audit sees them; the realtime path's fallback is a safety feature
  irrelevant to faithfulness analysis)
* run for any longer than the input file

Output WAVs are gitignored under ``*.wav`` — they are diagnostic
artifacts, not source.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tvoice-pseudo-stream",
        description="Drive an RVC model in pseudo-stream chunks for "
                    "model-faithfulness comparison vs the offline whole-file "
                    "pass. Writes a WAV. Does not touch audio devices.",
    )
    p.add_argument("--input-wav", required=True,
                   help="Input WAV (any sample width / channels).")
    p.add_argument("--output-wav", required=True,
                   help="Output WAV (gitignored — see *.wav rule).")
    p.add_argument("--model-profile", default=None,
                   help="Model profile JSON. Recommended: supplies voice "
                        "identity verbatim, matching --mode rvc.")
    p.add_argument("--model-path", default=None,
                   help="DEVELOPER OVERRIDE: bypass profile model_path.")
    p.add_argument("--index-path", default=None,
                   help="DEVELOPER OVERRIDE.")
    p.add_argument("--hubert-path", default=None,
                   help="DEVELOPER OVERRIDE.")
    p.add_argument("--rmvpe-path", default=None,
                   help="DEVELOPER OVERRIDE.")
    p.add_argument("--backend", default="infer_rvc_python")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "directml_experimental"])
    p.add_argument("--stream-sr", type=int, default=48000,
                   help="Sample rate the pseudo-stream renders at (the "
                        "realtime equivalent is the OutputStream SR). The "
                        "engine input is resampled to this; the engine "
                        "output is resampled from its native SR to this.")
    p.add_argument("--chunk-ms", type=float, default=1000.0,
                   help="Pseudo-stream chunk size in ms (default 1000 — "
                        "matches the recommended realtime chunk).")
    p.add_argument("--crossfade-ms", type=float, default=0.0,
                   help="Stitched crossfade at chunk boundaries in ms. "
                        "Default 0 (model-faithful). Set >0 to reproduce "
                        "the legacy stitched-blend behaviour.")
    p.add_argument("--resampler", default="polyphase",
                   choices=["polyphase", "linear", "torchaudio"],
                   help="Resampler used between engine native SR and "
                        "stream SR. 'polyphase' uses scipy.signal.resample_"
                        "poly (sinc-quality, recommended). 'linear' uses "
                        "np.interp (the historical default — what the "
                        "current worker does). 'torchaudio' uses "
                        "torchaudio.functional.resample.")
    p.add_argument("--resample-sr", type=int, default=0,
                   help="Ask the backend to resample its output internally "
                        "to this SR (the offline path's old default was "
                        "0 = model native, --stream-sr converts here; "
                        "set this to --stream-sr to mirror the backend-"
                        "resamples-internally A/B reference).")
    p.add_argument("--warmup-count", type=int, default=2,
                   help="Warmup inferences before the timed run.")
    p.add_argument("--limit-seconds", type=float, default=None,
                   help="Optionally cap input duration to first N seconds.")
    p.add_argument("--report-json", default=None,
                   help="If set, write per-chunk timing + amplitude metrics "
                        "to this JSON (also gitignored).")
    return p


_VOICE_FIELDS = (
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


def _resolve(args, profile) -> dict:
    out: dict = {}
    for cli_name, prof_name, default in _VOICE_FIELDS:
        cli_val = getattr(args, cli_name, None)
        prof_val = getattr(profile, prof_name) if profile is not None else None
        if cli_val is not None:
            out[prof_name] = cli_val
        elif prof_val is not None:
            out[prof_name] = prof_val
        else:
            out[prof_name] = default
    return out


def _resample_polyphase(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """scipy.signal.resample_poly with rational up/down factors.

    sinc-windowed polyphase resampling — the right tool for arbitrary
    SR pairs that share a small GCD (e.g. 40000:48000 = 5:6).
    """
    from math import gcd
    from scipy.signal import resample_poly
    if from_sr == to_sr:
        return audio.astype(np.float32, copy=True)
    g = gcd(int(from_sr), int(to_sr))
    up = int(to_sr) // g
    down = int(from_sr) // g
    out = resample_poly(audio.astype(np.float64, copy=False), up, down)
    return out.astype(np.float32, copy=False)


def _resample_torchaudio(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """torchaudio.functional.resample — Kaiser-windowed sinc, CPU."""
    import torch
    import torchaudio.functional as F
    if from_sr == to_sr:
        return audio.astype(np.float32, copy=True)
    t = torch.from_numpy(audio.astype(np.float32, copy=False))
    y = F.resample(t, int(from_sr), int(to_sr))
    return y.numpy().astype(np.float32, copy=False)


def _resample_linear(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """np.interp — what the current worker does. Aliased."""
    from src.audio.chunker import linear_resample
    return linear_resample(audio, int(from_sr), int(to_sr))


_RESAMPLERS = {
    "polyphase":  _resample_polyphase,
    "torchaudio": _resample_torchaudio,
    "linear":     _resample_linear,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    from src.audio.wav_io import read_wav_mono_float32, write_wav_float32
    from src.engine.crossfade import linear_crossfade
    from src.engine.model_profile import ModelProfileError, load_model_profile
    from src.engine.rvc_engine import (
        DependencyMissingError,
        ModelLoadError,
        RvcEngine,
        RvcEngineConfig,
        RvcInferenceError,
    )
    from src.safety.guard import dbfs_peak, dbfs_rms, scrub_nan_inf

    in_path = Path(args.input_wav)
    if not in_path.exists():
        print(f"error: input WAV not found: {in_path}", file=sys.stderr)
        return 2

    profile = None
    if args.model_profile:
        try:
            profile = load_model_profile(args.model_profile)
        except ModelProfileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 7
        print(f"loaded model profile: name={profile.name!r}")

    voice = _resolve(args, profile)
    if not voice["model_path"]:
        print("error: provide --model-profile or --model-path", file=sys.stderr)
        return 2
    if not Path(voice["model_path"]).exists():
        print(f"error: model not found: {voice['model_path']}", file=sys.stderr)
        return 3

    if args.resampler not in _RESAMPLERS:
        print(f"error: unknown resampler {args.resampler!r}", file=sys.stderr)
        return 2

    print(f"reading {in_path} ...")
    audio_raw, in_sr = read_wav_mono_float32(str(in_path))
    print(
        f"input : {audio_raw.size} samples @ {in_sr} Hz  "
        f"({audio_raw.size / float(in_sr):.2f} s)  "
        f"peak={dbfs_peak(audio_raw):.2f} dBFS  "
        f"rms={dbfs_rms(audio_raw):.2f} dBFS"
    )

    stream_sr = int(args.stream_sr)
    if in_sr != stream_sr:
        print(
            f"resampling input {in_sr} -> {stream_sr} Hz via {args.resampler}"
        )
        audio = _RESAMPLERS[args.resampler](audio_raw, in_sr, stream_sr)
    else:
        audio = audio_raw.astype(np.float32, copy=False)

    if args.limit_seconds is not None:
        cap = max(1, int(round(args.limit_seconds * stream_sr)))
        if audio.size > cap:
            audio = audio[:cap]
            print(f"capped input to first {args.limit_seconds:.2f} s = {audio.size} samples")

    cfg = RvcEngineConfig(
        model_path=voice["model_path"],
        index_path=voice["index_path"],
        backend=args.backend,
        f0_method=voice["f0_method"],
        index_rate=voice["index_rate"],
        protect=voice["protect"],
        filter_radius=voice["filter_radius"],
        rms_mix_rate=voice["rms_mix_rate"],
        pitch_shift=voice["pitch_shift"],
        sample_rate=stream_sr,
        resample_sr=int(args.resample_sr or 0),
        device=args.device,
        hubert_path=voice["hubert_path"],
        rmvpe_path=voice["rmvpe_path"],
    )
    engine = RvcEngine(cfg)
    print(f"loading RVC backend={cfg.backend} device={args.device} ...")
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
        f"cuda={engine.cuda_device_name or '(n/a)'}  resample_sr={cfg.resample_sr}"
    )

    chunk_size = max(1, int(round(args.chunk_ms / 1000.0 * stream_sr)))
    crossfade_size = max(0, int(round(args.crossfade_ms / 1000.0 * stream_sr)))
    print(
        f"chunk_size={chunk_size} samples ({args.chunk_ms:.1f} ms @ {stream_sr} Hz)  "
        f"crossfade_size={crossfade_size} samples  resampler={args.resampler}"
    )

    if args.warmup_count > 0:
        print(f"warming up {args.warmup_count} call(s)...")
        try:
            for ms in engine.warmup(chunk_size, stream_sr, args.warmup_count):
                print(f"  warmup: {ms:.0f} ms")
        except RvcInferenceError as exc:
            print(f"warning: warmup failed (continuing): {exc}", file=sys.stderr)

    # Split into chunks; pad the final partial chunk with zeros so the
    # engine sees consistent shapes (mimics what realtime would do when
    # the mic feeds steadily — the model never sees a half-chunk).
    n_full = audio.size // chunk_size
    remainder = audio.size - n_full * chunk_size
    chunks = [audio[i * chunk_size:(i + 1) * chunk_size] for i in range(n_full)]
    if remainder > 0:
        last = np.zeros(chunk_size, dtype=np.float32)
        last[:remainder] = audio[n_full * chunk_size:]
        chunks.append(last)
        print(f"padded final chunk: {remainder} real samples + {chunk_size - remainder} silence")

    print(f"running pseudo-stream: {len(chunks)} chunk(s) ...")

    output_pieces: List[np.ndarray] = []
    pending_tail: Optional[np.ndarray] = None
    per_chunk_ms: List[float] = []
    per_chunk_peak: List[float] = []
    native_sr_seen: Optional[int] = None

    for i, chunk in enumerate(chunks):
        t0 = time.perf_counter()
        processed, result_sr = engine.infer_array(chunk, stream_sr)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        per_chunk_ms.append(dur_ms)
        result_sr = int(result_sr)
        if native_sr_seen is None:
            native_sr_seen = result_sr

        processed = np.asarray(processed, dtype=np.float32).reshape(-1)
        scrub = scrub_nan_inf(processed)
        if scrub.replaced_count:
            print(f"  chunk {i}: scrubbed {scrub.replaced_count} NaN/Inf")
            processed = scrub.audio

        if processed.size == 0 or result_sr <= 0:
            print(f"  chunk {i}: backend returned invalid audio "
                  f"(size={processed.size} sr={result_sr})", file=sys.stderr)
            continue

        if result_sr != stream_sr:
            processed = _RESAMPLERS[args.resampler](processed, result_sr, stream_sr)

        per_chunk_peak.append(float(np.max(np.abs(processed))) if processed.size else 0.0)

        # Stitched crossfade (matches the worker's current behaviour).
        if crossfade_size == 0 or processed.size < 2 * crossfade_size:
            if pending_tail is not None:
                output_pieces.append(pending_tail)
                pending_tail = None
            output_pieces.append(processed)
        elif pending_tail is None:
            output_pieces.append(processed[:-crossfade_size])
            pending_tail = processed[-crossfade_size:].astype(np.float32, copy=True)
        else:
            head = processed[:crossfade_size]
            blended = linear_crossfade(pending_tail, head)
            output_pieces.append(blended)
            output_pieces.append(processed[crossfade_size:-crossfade_size])
            pending_tail = processed[-crossfade_size:].astype(np.float32, copy=True)

        print(
            f"  chunk {i + 1}/{len(chunks)}: in={chunk.size}@{stream_sr} "
            f"-> out={processed.size}@{stream_sr}  "
            f"infer={dur_ms:5.0f} ms  peak={per_chunk_peak[-1]:.3f}"
        )

    if pending_tail is not None:
        output_pieces.append(pending_tail)

    result = (
        np.concatenate(output_pieces).astype(np.float32, copy=False)
        if output_pieces else np.zeros(0, dtype=np.float32)
    )
    print(
        f"output: {result.size} samples @ {stream_sr} Hz  "
        f"({result.size / float(stream_sr):.2f} s)  "
        f"peak={dbfs_peak(result):.2f} dBFS  "
        f"rms={dbfs_rms(result):.2f} dBFS"
    )

    out_path = Path(args.output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav_float32(str(out_path), result, stream_sr)
    print(f"wrote {out_path}")

    if args.report_json:
        import json
        rep = {
            "input_wav": str(in_path),
            "output_wav": str(out_path),
            "stream_sr": stream_sr,
            "engine_native_sr": native_sr_seen,
            "chunk_ms": float(args.chunk_ms),
            "crossfade_ms": float(args.crossfade_ms),
            "resampler": args.resampler,
            "resample_sr": int(args.resample_sr or 0),
            "n_chunks": len(per_chunk_ms),
            "infer_ms": per_chunk_ms,
            "chunk_peak_amplitude": per_chunk_peak,
            "result_samples": int(result.size),
            "result_peak_dbfs": float(dbfs_peak(result)),
            "result_rms_dbfs":  float(dbfs_rms(result)),
        }
        Path(args.report_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"wrote report {args.report_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
