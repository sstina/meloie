"""Microbenchmark for ``RvcEngine.infer_array`` outside the realtime stream.

Why this exists: the Stage 2C realtime smoke ran without errors but at
~3.4 s mean inference per 180 ms chunk on a 4080. That cannot be
diagnosed from inside the realtime loop. This tool calls the engine
directly, with proper CUDA synchronisation around each call, so we can
see per-call timings and figure out which knob actually moves them.

Hard rules:
* No audio devices opened — this is offline microbenchmarking.
* No torch / infer_rvc_python import at module load — both are pulled
  in lazily inside ``main()`` so the unit tests don't need the RVC
  stack installed.
* No artifacts written unless ``--save-report`` is given and a
  ``--report-dir`` (kept under the gitignored
  ``eval_corpus/reports/rvc_perf/``) is supplied.

Example::

    .\\.venv310\\Scripts\\python.exe -m tools.benchmark_rvc_infer \\
        --input-wav test.wav \\
        --model-path models\\kiki\\kikiV1.pth \\
        --index-path models\\kiki\\kikiV1.index \\
        --hubert-path models\\kiki\\hubert_base.pt \\
        --rmvpe-path models\\kiki\\rmvpe.pt \\
        --device cuda --f0-method rmvpe \\
        --chunk-ms 180,360,500,750,1000,1500 \\
        --warmup-count 2 --repeat-count 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tvoice-benchmark-rvc-infer",
        description="Microbenchmark RvcEngine.infer_array per chunk size.",
    )
    p.add_argument("--input-wav", default=None,
                   help="Path to a mono WAV used as the source audio. "
                        "If omitted, a synthetic mixed-tone signal is generated.")
    p.add_argument("--sample-rate", type=int, default=48000,
                   help="Sample rate to feed into the engine (synthetic + "
                        "after resample). Default 48000.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--index-path", default=None)
    p.add_argument("--hubert-path", default=None)
    p.add_argument("--rmvpe-path", default=None)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "directml_experimental"])
    p.add_argument("--f0-method", default="rmvpe")
    p.add_argument("--index-rate", type=float, default=0.5)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--filter-radius", type=int, default=3)
    p.add_argument("--rms-mix-rate", type=float, default=0.25)
    p.add_argument("--pitch-shift", type=int, default=0)
    p.add_argument("--resample-sr", type=int, default=48000)
    p.add_argument("--chunk-ms", default="180,360,500,750,1000,1500",
                   help="Comma-separated list of chunk sizes in ms.")
    p.add_argument("--warmup-count", type=int, default=2,
                   help="Per-chunk-size warmup calls (not timed).")
    p.add_argument("--repeat-count", type=int, default=5,
                   help="Per-chunk-size timed calls.")
    p.add_argument("--no-side-tests", action="store_true",
                   help="Skip the index_rate=0 and resample_sr=0 side sweeps.")
    p.add_argument("--report-dir", default=None,
                   help="Directory for JSON report (use "
                        "eval_corpus/reports/rvc_perf/ so gitignore covers).")
    p.add_argument("--save-report", action="store_true",
                   help="Persist benchmark results to JSON in --report-dir.")
    return p


def _cuda_sync() -> None:
    try:
        import torch  # noqa: WPS433
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _synthesize_test_audio(sample_rate: int, seconds: float = 3.0) -> np.ndarray:
    """Mixed-tone speech-shaped signal for benchmarking when no WAV given."""
    n = int(round(seconds * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    sig = (
        0.30 * np.sin(2 * np.pi * 200.0 * t)
        + 0.18 * np.sin(2 * np.pi * 400.0 * t)
        + 0.10 * np.sin(2 * np.pi * 800.0 * t)
        + 0.05 * np.sin(2 * np.pi * 1600.0 * t)
    )
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (sig * envelope).astype(np.float32)


def _read_input_audio(path: Optional[str], target_sr: int) -> np.ndarray:
    """Load audio at ``target_sr``. Reads WAV via stdlib; resamples via numpy
    linear interpolation (good enough — we are timing inference, not
    measuring audio quality)."""
    if path is None:
        return _synthesize_test_audio(target_sr)
    from src.audio.wav_io import read_wav_mono_float32
    audio, sr = read_wav_mono_float32(path)
    if sr == target_sr:
        return audio
    # Simple linear resample. Suitable only for benchmark warm-up audio.
    ratio = float(target_sr) / float(sr)
    new_len = int(round(audio.size * ratio))
    if new_len <= 0:
        raise ValueError("resampled audio would be empty")
    x_old = np.arange(audio.size, dtype=np.float64)
    x_new = np.linspace(0.0, audio.size - 1, new_len, dtype=np.float64)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _introspect_backend(engine) -> Dict[str, Any]:
    """Best-effort inspection of where the model's tensors actually live.

    The returned dict reports what we could observe. Anything that
    raises gets caught and recorded as ``"?"`` — the benchmark must
    keep running even if backend internals change shape.
    """
    out: Dict[str, Any] = {
        "backend_name": engine.backend_name,
        "resolved_device": engine.resolved_device,
        "cuda_device_name": engine.cuda_device_name,
    }
    try:
        import torch  # noqa: WPS433
        out["torch_version"] = torch.__version__
        out["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        out["torch_version"] = "?"
        out["torch_cuda_available"] = False

    converter = getattr(engine._backend, "_converter", None)
    if converter is None:
        return out

    out["converter_type"] = type(converter).__name__
    out["converter_attrs"] = sorted(
        a for a in dir(converter) if not a.startswith("_")
    )

    # Defensive deep-dive — different infer_rvc_python versions expose
    # different attribute names.
    for attr_name in (
        "hubert_model", "model_voice_path", "model_pitch", "rmvpe_model",
        "model_config", "cache_model", "config", "model_voice",
        "model", "loaded_models",
    ):
        try:
            val = getattr(converter, attr_name, None)
        except Exception:
            continue
        if val is None:
            continue
        info: Dict[str, Any] = {"type": type(val).__name__}
        # If it's a torch nn.Module, get device/dtype of first param.
        try:
            params = list(val.parameters())  # type: ignore[attr-defined]
            if params:
                info["first_param_device"] = str(params[0].device)
                info["first_param_dtype"] = str(params[0].dtype)
                info["param_count"] = int(sum(p.numel() for p in params))
        except Exception:
            pass
        # If dict-like, capture top-level keys.
        try:
            if hasattr(val, "keys"):
                info["keys"] = list(val.keys())  # type: ignore[arg-type]
        except Exception:
            pass
        out[f"converter.{attr_name}"] = info

    return out


def _bench_one(
    engine, audio: np.ndarray, sample_rate: int, chunk_ms: float,
    warmup_count: int, repeat_count: int, label: str,
) -> Dict[str, Any]:
    from src.audio.benchmark import (
        fits_realtime, realtime_factor, slice_repeating, summarize_timings,
    )
    from src.safety.guard import dbfs_peak, dbfs_rms

    chunk_size = max(1, int(round(chunk_ms * sample_rate / 1000.0)))
    chunk = slice_repeating(audio, chunk_size)

    warmup_ms: List[float] = []
    for _ in range(int(warmup_count)):
        t0 = time.perf_counter()
        _result, _sr = engine.infer_array(chunk, sample_rate)
        _cuda_sync()
        warmup_ms.append((time.perf_counter() - t0) * 1000.0)

    timed_ms: List[float] = []
    last_result: Optional[np.ndarray] = None
    last_sr = sample_rate
    for _ in range(int(repeat_count)):
        t0 = time.perf_counter()
        last_result, last_sr = engine.infer_array(chunk, sample_rate)
        _cuda_sync()
        timed_ms.append((time.perf_counter() - t0) * 1000.0)

    summary = summarize_timings(timed_ms)
    rt = realtime_factor(summary["mean_ms"], chunk_ms)

    if last_result is None:
        peak_dbfs = -200.0
        rms_dbfs = -200.0
        clip_pct = 0.0
        output_samples = 0
    else:
        arr = np.asarray(last_result, dtype=np.float32).reshape(-1)
        peak_dbfs = float(dbfs_peak(arr))
        rms_dbfs = float(dbfs_rms(arr))
        clip_pct = float(np.mean(np.abs(arr) >= 0.999)) * 100.0
        output_samples = int(arr.size)

    return {
        "label": label,
        "chunk_ms": float(chunk_ms),
        "chunk_samples": int(chunk_size),
        "input_sample_rate": int(sample_rate),
        "output_samples": output_samples,
        "output_sample_rate": int(last_sr),
        "warmup_ms": warmup_ms,
        "timed_ms": timed_ms,
        "summary": summary,
        "realtime_factor": float(rt),
        "fits_realtime": bool(fits_realtime(rt)),
        "output_peak_dbfs": peak_dbfs,
        "output_rms_dbfs": rms_dbfs,
        "output_clip_pct": clip_pct,
    }


def _print_result_row(r: Dict[str, Any]) -> None:
    s = r["summary"]
    print(
        f"  [{r['label']:>10s}] chunk_ms={r['chunk_ms']:>5.0f}  "
        f"mean={s['mean_ms']:>7.1f}  median={s['median_ms']:>7.1f}  "
        f"p95={s['p95_ms']:>7.1f}  max={s['max_ms']:>7.1f}  "
        f"rt={r['realtime_factor']:>5.2f}  "
        f"fits={'Y' if r['fits_realtime'] else 'N'}  "
        f"out_sr={r['output_sample_rate']}  "
        f"warmup={['%.0f' % w for w in r['warmup_ms']]}  "
        f"out_pk={r['output_peak_dbfs']:>6.1f}dB  "
        f"clip%={r['output_clip_pct']:>4.1f}",
        flush=True,
    )


def _build_engine(args: argparse.Namespace, *, index_rate: float,
                  resample_sr: int, label: str):
    from src.engine.rvc_engine import RvcEngine, RvcEngineConfig

    cfg = RvcEngineConfig(
        model_path=args.model_path,
        index_path=args.index_path,
        backend="infer_rvc_python",
        f0_method=args.f0_method,
        index_rate=index_rate,
        protect=args.protect,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        pitch_shift=args.pitch_shift,
        resample_sr=resample_sr,
        device=args.device,
        hubert_path=args.hubert_path,
        rmvpe_path=args.rmvpe_path,
        backend_tag=f"bench_{label}",
    )
    return RvcEngine(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    chunk_ms_list = [float(s) for s in str(args.chunk_ms).split(",") if s.strip()]
    if not chunk_ms_list:
        print("error: --chunk-ms must list at least one value", file=sys.stderr)
        return 2

    if not Path(args.model_path).exists():
        print(f"error: model not found: {args.model_path}", file=sys.stderr)
        return 3

    print(f"reading input audio (target_sr={args.sample_rate}) ...")
    audio = _read_input_audio(args.input_wav, int(args.sample_rate))
    print(f"  audio: {audio.size} samples at {args.sample_rate} Hz "
          f"({audio.size / float(args.sample_rate):.2f} s)")

    all_results: List[Dict[str, Any]] = []
    introspection: Dict[str, Any] = {}

    # -- main sweep -------------------------------------------------------
    print(
        f"\n=== main sweep  index_rate={args.index_rate}  "
        f"resample_sr={args.resample_sr}  f0={args.f0_method} ==="
    )
    engine_main = _build_engine(
        args, index_rate=args.index_rate, resample_sr=args.resample_sr, label="main"
    )
    try:
        engine_main.load()
    except Exception as exc:
        print(f"error: engine load failed: {exc}", file=sys.stderr)
        return 10
    print(
        f"  engine loaded. resolved_device={engine_main.resolved_device} "
        f"cuda_device={engine_main.cuda_device_name or '(n/a)'}"
    )
    introspection = _introspect_backend(engine_main)

    for chunk_ms in chunk_ms_list:
        r = _bench_one(
            engine_main, audio, int(args.sample_rate), chunk_ms,
            args.warmup_count, args.repeat_count, label="main",
        )
        all_results.append(r)
        _print_result_row(r)

    # Pick the chunk size with the lowest mean inference for the side sweeps.
    best = min(all_results, key=lambda r: r["summary"]["mean_ms"])
    best_chunk_ms = best["chunk_ms"]
    print(f"\nbest by mean_ms in main sweep: chunk_ms={best_chunk_ms}")

    # -- side sweep A: index_rate=0 ---------------------------------------
    if not args.no_side_tests:
        print(
            f"\n=== side A: index_rate=0.0  resample_sr={args.resample_sr} "
            f"(retains index file but down-weights it) ==="
        )
        engine_noidx = _build_engine(
            args, index_rate=0.0, resample_sr=args.resample_sr, label="noidx"
        )
        try:
            engine_noidx.load()
            for chunk_ms in (180.0, best_chunk_ms):
                r = _bench_one(
                    engine_noidx, audio, int(args.sample_rate), chunk_ms,
                    args.warmup_count, args.repeat_count, label="noidx",
                )
                all_results.append(r)
                _print_result_row(r)
        except Exception as exc:
            print(f"warning: side A failed: {exc}", file=sys.stderr)

        # -- side sweep B: resample_sr=0 (native model SR) ----------------
        print(
            f"\n=== side B: index_rate={args.index_rate}  resample_sr=0 "
            "(backend's native output SR; no internal resampling) ==="
        )
        engine_native = _build_engine(
            args, index_rate=args.index_rate, resample_sr=0, label="native"
        )
        try:
            engine_native.load()
            for chunk_ms in (best_chunk_ms,):
                r = _bench_one(
                    engine_native, audio, int(args.sample_rate), chunk_ms,
                    args.warmup_count, args.repeat_count, label="native",
                )
                all_results.append(r)
                _print_result_row(r)
        except Exception as exc:
            print(f"warning: side B failed: {exc}", file=sys.stderr)

    # -- final summary ---------------------------------------------------
    print("\n=== summary ===")
    for r in all_results:
        _print_result_row(r)

    fastest = min(all_results, key=lambda r: r["realtime_factor"])
    print(
        f"\nfastest by realtime_factor: label={fastest['label']} "
        f"chunk_ms={fastest['chunk_ms']:.0f}  "
        f"rt={fastest['realtime_factor']:.2f}  "
        f"mean={fastest['summary']['mean_ms']:.1f}ms  "
        f"fits_realtime={fastest['fits_realtime']}"
    )

    print("\n=== introspection ===")
    for k, v in introspection.items():
        print(f"  {k} = {v}")

    if args.save_report:
        if not args.report_dir:
            print("error: --save-report requires --report-dir", file=sys.stderr)
            return 4
        out_dir = Path(args.report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        path = out_dir / f"benchmark_rvc_infer_{ts}.json"
        payload = {
            "timestamp": ts,
            "args": vars(args),
            "introspection": introspection,
            "results": all_results,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved: {path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
