"""Stage 4-E2 investigation probe — diagnostic only (NOT staged).

Answers three questions the Stage 4-E2 task requires before any code
change:

  1. Is the per-chunk output-length deficit deterministic, and exactly
     how large is it (in backend SR and stream SR terms)? Repeated
     trials over several input lengths and content types.

  2. WHERE does the deficit live -- at the front, the tail, or split?
     Decided by two cross-correlation experiments:
        (a) tail test  : infer(X) vs infer(X + extra_tail). If the
            front of both renderings aligns at lag 0 and stays
            correlated until near the end, the model's output is
            stable at the front and the change is at the tail ->
            deficit (and look-ahead need) is at the TAIL.
        (b) front test : infer(X) vs infer(context + X). The lag at
            which X's rendering begins in the second output tells us
            how the front maps (and whether prepending context costs
            any output at the start).

  3. Does tail padding (silence vs real look-ahead) make the usable
     current-chunk region fully present so a sample-accurate trim can
     replace the Stage 4-E polyphase stretch?

Run (from RVC/, env vars redirected to RVC/.cache, .venv310 active):

  python -m tools.probe_frame_deficit \
      --model-profile config/model_profiles/kiki.example.json \
      --device cuda --stream-sr 48000 --input-wav test.wav
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def _xcorr_best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple:
    """Return (best_lag, normalised_corr_at_best_lag).

    Positive lag means ``b`` is delayed relative to ``a`` (b[lag:] aligns
    with a[:]). Brute force over [-max_lag, max_lag] on an overlap window.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    n = min(a.size, b.size)
    if n == 0:
        return 0, 0.0
    win = min(n, 20000)
    best_lag = 0
    best_corr = -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa = a[: win]
            bb = b[lag: lag + win]
        else:
            aa = a[-lag: -lag + win]
            bb = b[: win]
        m = min(aa.size, bb.size)
        if m < 1000:
            continue
        aa = aa[:m]
        bb = bb[:m]
        da = aa - aa.mean()
        db = bb - bb.mean()
        denom = np.sqrt((da * da).sum() * (db * db).sum())
        if denom <= 0:
            continue
        c = float((da * db).sum() / denom)
        if c > best_corr:
            best_corr = c
            best_lag = lag
    return best_lag, best_corr


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="probe-frame-deficit")
    p.add_argument("--model-profile", required=True)
    p.add_argument("--device", default="cuda",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--stream-sr", type=int, default=48000)
    p.add_argument("--input-wav", default=None,
                   help="Real speech wav for content-dependence + xcorr tests.")
    p.add_argument("--trials", type=int, default=8,
                   help="Trials per (length,content) cell for the stats pass.")
    args = p.parse_args(argv)

    from src.audio.wav_io import read_wav_mono_float32
    from src.engine.model_profile import load_model_profile
    from src.engine.rvc_engine import RvcEngine, RvcEngineConfig

    profile = load_model_profile(args.model_profile)
    print(f"profile: {profile.name}")
    cfg = RvcEngineConfig(
        model_path=profile.model_path,
        index_path=profile.index_path,
        f0_method=profile.f0_method,
        index_rate=profile.index_rate,
        protect=profile.protect,
        filter_radius=profile.filter_radius,
        rms_mix_rate=profile.rms_mix_rate,
        pitch_shift=profile.pitch_shift,
        sample_rate=args.stream_sr,
        resample_sr=0,
        device=args.device,
        hubert_path=profile.hubert_path,
        rmvpe_path=profile.rmvpe_path,
    )
    engine = RvcEngine(cfg)
    print("loading engine ...")
    engine.load()
    print(f"loaded. device={engine.resolved_device} "
          f"cuda={engine.cuda_device_name}")

    sr = int(args.stream_sr)

    # Source content generators.
    def sine(n: int) -> np.ndarray:
        t = np.arange(n, dtype=np.float64) / sr
        return (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)

    def noise(n: int) -> np.ndarray:
        # deterministic pseudo-noise (no Math.random equivalent needed)
        idx = np.arange(n, dtype=np.float64)
        return (0.1 * np.sin(idx * 0.7) * np.cos(idx * 0.013)).astype(np.float32)

    speech_full: Optional[np.ndarray] = None
    if args.input_wav and Path(args.input_wav).exists():
        raw, in_sr = read_wav_mono_float32(args.input_wav)
        if in_sr != sr:
            from src.audio.chunker import resample_audio
            raw = resample_audio(raw, in_sr, sr)
        speech_full = raw
        print(f"speech wav: {speech_full.size} samples @ {sr} "
              f"({speech_full.size / sr:.2f}s)")

    def speech(n: int, off: int = 0) -> np.ndarray:
        if speech_full is None or speech_full.size < n + off:
            return sine(n)
        return speech_full[off: off + n].astype(np.float32, copy=True)

    print("warming up ...")
    engine.warmup(48000, sr, 2)

    # ----- Pass 1: deterministic deficit over lengths & content -----
    print("\n=== PASS 1: output length deficit (native SR) ===")
    lengths = [24000, 48000, 52800, 57600, 96000]
    contents = {"sine": sine, "noise": noise}
    if speech_full is not None:
        contents["speech"] = speech
    native_sr_seen = None
    rows = []
    for n in lengths:
        for cname, gen in contents.items():
            outs = []
            for k in range(args.trials):
                x = gen(n) if cname != "speech" else speech(n, off=k * 4000)
                y, rsr = engine.infer_array(x, sr)
                native_sr_seen = int(rsr)
                outs.append(int(np.asarray(y).reshape(-1).size))
            mean = statistics.mean(outs)
            mn, mx = min(outs), max(outs)
            std = statistics.pstdev(outs) if len(outs) > 1 else 0.0
            # Expected output if NO deficit, at native SR.
            exp_native = round(n * native_sr_seen / sr)
            deficit_native = exp_native - mean
            deficit_stream = deficit_native * sr / native_sr_seen
            rows.append((n, cname, mean, mn, mx, std,
                         deficit_native, deficit_stream))
            print(f"  in={n:>6} {cname:<7} out_mean={mean:>9.1f} "
                  f"min={mn} max={mx} std={std:5.2f} "
                  f"| deficit_native={deficit_native:7.1f} "
                  f"deficit_stream={deficit_stream:7.1f}")
    print(f"native_sr={native_sr_seen}")

    # ----- Pass 2: TAIL localization -----
    # infer(X) vs infer(X + extra_tail). Front-aligned (lag 0) + high corr
    # over the front => the model's rendering is stable at the front and
    # the extra output appears at the tail => the lost frame is at the TAIL.
    print("\n=== PASS 2: tail localization (xcorr) ===")
    base_n = 48000
    extra = 9600  # 200 ms
    X = speech(base_n, off=0) if speech_full is not None else sine(base_n)
    tail_real = speech(extra, off=base_n) if speech_full is not None else sine(extra)
    tail_zero = np.zeros(extra, dtype=np.float32)
    outA, _ = engine.infer_array(X, sr)
    outB_real, _ = engine.infer_array(np.concatenate([X, tail_real]), sr)
    outB_zero, _ = engine.infer_array(np.concatenate([X, tail_zero]), sr)
    outA = np.asarray(outA).reshape(-1)
    outB_real = np.asarray(outB_real).reshape(-1)
    outB_zero = np.asarray(outB_zero).reshape(-1)
    lag_r, corr_r = _xcorr_best_lag(outA, outB_real, max_lag=2000)
    lag_z, corr_z = _xcorr_best_lag(outA, outB_zero, max_lag=2000)
    print(f"  len(outA)={outA.size}  len(outB_real)={outB_real.size}  "
          f"len(outB_zero)={outB_zero.size}")
    print(f"  X vs X+real_tail : best_lag={lag_r} corr={corr_r:.4f}  "
          f"(lag~0 & high corr => front stable, change at tail)")
    print(f"  X vs X+zero_tail : best_lag={lag_z} corr={corr_z:.4f}")
    # How long does the front stay (near-)identical? Compare prefix RMS err.
    for label, outB in (("real", outB_real), ("zero", outB_zero)):
        k = min(outA.size, outB.size)
        # find first index (from the end, scanning back) where they diverge
        diff = np.abs(outA[:k] - outB[:k])
        thr = 0.02 * (np.abs(outA[:k]).max() + 1e-9)
        diverge = np.where(diff > thr)[0]
        first_div = int(diverge[0]) if diverge.size else k
        last_same = int(k - (np.where(diff[::-1] > thr)[0][0] if
                             (diff > thr).any() else k))
        print(f"    tail={label}: front identical up to ~{first_div} / {k} "
              f"samples (native SR)")

    # ----- Pass 3: FRONT localization -----
    # infer(X) vs infer(context + X). Find the lag at which X's rendering
    # begins in the second output. Expect ~ context*native/stream if the
    # front carries no extra deficit.
    print("\n=== PASS 3: front localization (xcorr) ===")
    ctx = speech(extra, off=0) if speech_full is not None else sine(extra)
    Xf = speech(base_n, off=extra) if speech_full is not None else sine(base_n)
    outX, _ = engine.infer_array(Xf, sr)
    outCX, _ = engine.infer_array(np.concatenate([ctx, Xf]), sr)
    outX = np.asarray(outX).reshape(-1)
    outCX = np.asarray(outCX).reshape(-1)
    lag_f, corr_f = _xcorr_best_lag(outX, outCX, max_lag=12000)
    exp_ctx_native = round(extra * (native_sr_seen / sr))
    print(f"  len(outX)={outX.size}  len(out[ctx+X])={outCX.size}")
    print(f"  best_lag={lag_f} corr={corr_f:.4f}  "
          f"expected_ctx_native={exp_ctx_native}  "
          f"(lag close to expected => front maps cleanly, no start deficit)")

    print("\n=== SUMMARY HINTS ===")
    print("  If PASS2 lag~0 with high corr and front identical for most of "
          "the window, the lost frame is at the TAIL: a small real/zero "
          "tail pad + sample-accurate trim restores a full current chunk.")
    print("  If PASS3 lag ~= expected_ctx_native, prepended context costs "
          "no extra start frames: front trim = round(context*native/stream).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
