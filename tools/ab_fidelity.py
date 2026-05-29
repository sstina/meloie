"""Stage 4-E2 offline A/B fidelity — diagnostic only (NOT staged).

Loads the kiki engine ONCE and compares, against the offline whole-file
ground truth (full context everywhere = the faithfulness gold standard):

  * stretch   — Stage 4-E output-side polyphase stretch (pitch-shifted)
  * silence   — input-side silence tail pad + exact slice
  * lookahead — input-side real look-ahead tail pad + exact slice

For each, the chunked pipeline mirrors src.engine.worker.rvc_worker_loop.
Reports, after integer-lag cross-correlation alignment vs GT:
  - alignment shift (samples / ms)
  - RMS-error SNR (dB) over the aligned overlap
  - high-band (>8 kHz) error-energy fraction
  - timeline drift (output duration vs input duration)
  - whether every emitted chunk == chunk_size (no drift, no stretch len)

Run from RVC/ (.venv310, caches redirected, PYTHONIOENCODING=utf-8):
  python -m tools.ab_fidelity --model-profile config/model_profiles/kiki.example.json \
      --input-wav test.wav --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def _xcorr_lag(ref: np.ndarray, cand: np.ndarray, max_lag: int = 4000) -> tuple:
    a = ref.astype(np.float64)
    b = cand.astype(np.float64)
    win = min(a.size, b.size, 60000)
    best_lag, best_c = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa, bb = a[:win], b[lag: lag + win]
        else:
            aa, bb = a[-lag: -lag + win], b[:win]
        m = min(aa.size, bb.size)
        if m < 2000:
            continue
        aa, bb = aa[:m] - aa[:m].mean(), bb[:m] - bb[:m].mean()
        denom = np.sqrt((aa * aa).sum() * (bb * bb).sum())
        if denom <= 0:
            continue
        c = float((aa * bb).sum() / denom)
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag, best_c


def _snr_db(ref: np.ndarray, cand: np.ndarray, lag: int) -> tuple:
    if lag >= 0:
        a, b = ref, cand[lag:]
    else:
        a, b = ref[-lag:], cand
    m = min(a.size, b.size)
    a, b = a[:m].astype(np.float64), b[:m].astype(np.float64)
    # scale cand to best match (gain-invariant SNR)
    g = float((a * b).sum() / ((b * b).sum() + 1e-12))
    err = a - g * b
    sig_p = float((a * a).sum()) + 1e-12
    err_p = float((err * err).sum()) + 1e-12
    snr = 10.0 * np.log10(sig_p / err_p)
    # HF (>8 kHz) error-energy fraction
    nfft = 1 << int(np.ceil(np.log2(max(2, m))))
    E = np.abs(np.fft.rfft(err, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / 48000.0)
    hf = float(E[freqs >= 8000].sum() / (E.sum() + 1e-12))
    return snr, hf, m


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="ab-fidelity")
    p.add_argument("--model-profile", required=True)
    p.add_argument("--input-wav", required=True)
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--stream-sr", type=int, default=48000)
    p.add_argument("--chunk-ms", type=float, default=1000.0)
    p.add_argument("--context-ms", type=float, default=200.0)
    p.add_argument("--tail-pad-ms", type=float, default=30.0)
    args = p.parse_args(argv)

    from src.audio.chunker import (
        reconcile_to_length,
        resample_audio,
        trim_to_region,
    )
    from src.audio.wav_io import read_wav_mono_float32
    from src.engine.model_profile import load_model_profile
    from src.engine.rvc_engine import RvcEngine, RvcEngineConfig
    from src.safety.guard import scrub_nan_inf

    sr = int(args.stream_sr)
    profile = load_model_profile(args.model_profile)
    cfg = RvcEngineConfig(
        model_path=profile.model_path, index_path=profile.index_path,
        f0_method=profile.f0_method, index_rate=profile.index_rate,
        protect=profile.protect, filter_radius=profile.filter_radius,
        rms_mix_rate=profile.rms_mix_rate, pitch_shift=profile.pitch_shift,
        sample_rate=sr, resample_sr=0, device=args.device,
        hubert_path=profile.hubert_path, rmvpe_path=profile.rmvpe_path,
    )
    engine = RvcEngine(cfg)
    print("loading engine ...")
    engine.load()
    print(f"loaded. device={engine.resolved_device}")

    raw, in_sr = read_wav_mono_float32(args.input_wav)
    audio = resample_audio(raw, in_sr, sr) if in_sr != sr else raw.astype(np.float32)
    print(f"input: {audio.size} samples ({audio.size/sr:.3f}s)")
    engine.warmup(int(args.chunk_ms / 1000.0 * sr), sr, 2)

    chunk_size = max(1, int(round(args.chunk_ms / 1000.0 * sr)))
    context_size = max(0, int(round(args.context_ms / 1000.0 * sr)))
    tail_pad_size = max(0, int(round(args.tail_pad_ms / 1000.0 * sr)))

    def infer(x):
        y, rsr = engine.infer_array(x.astype(np.float32), sr)
        y = scrub_nan_inf(np.asarray(y, dtype=np.float32).reshape(-1)).audio
        rsr = int(rsr)
        if rsr != sr:
            y = resample_audio(y, rsr, sr)
        return y

    # Ground truth: whole-file single pass (full context everywhere).
    print("rendering ground truth (whole file) ...")
    gt = infer(audio)
    print(f"  GT: {gt.size} samples ({gt.size/sr:.3f}s)")

    # Build chunk list (pad final partial chunk with zeros).
    n_full = audio.size // chunk_size
    rem = audio.size - n_full * chunk_size
    chunks = [audio[i*chunk_size:(i+1)*chunk_size] for i in range(n_full)]
    if rem > 0:
        last = np.zeros(chunk_size, dtype=np.float32)
        last[:rem] = audio[n_full*chunk_size:]
        chunks.append(last)

    def run_method(method):
        ctx = np.zeros(context_size, dtype=np.float32) if context_size > 0 else None
        input_side = method in ("lookahead", "silence")
        out_pieces = []
        emit_sizes = []
        for i, ch in enumerate(chunks):
            parts = []
            if ctx is not None:
                parts.append(ctx)
            parts.append(ch)
            if input_side and tail_pad_size > 0:
                if method == "lookahead":
                    nxt = chunks[i+1] if i+1 < len(chunks) else None
                    tail = (nxt[:tail_pad_size].copy() if nxt is not None
                            else np.zeros(tail_pad_size, dtype=np.float32))
                else:
                    tail = np.zeros(tail_pad_size, dtype=np.float32)
                parts.append(tail)
            x = parts[0] if len(parts) == 1 else np.concatenate(parts)
            proc = infer(x)  # already at stream sr
            if input_side:
                emit, _short = trim_to_region(proc, context_size, chunk_size)
            else:
                # legacy: proportional native-equivalent trim already folded
                # into stream-sr terms here (proc is at stream sr) then stretch.
                if context_size > 0 and proc.size > 0:
                    # proc corresponds to [ctx][chunk] at stream sr -> drop ctx.
                    trim = min(context_size, proc.size - 1)
                    proc = proc[trim:]
                if method == "stretch" and proc.size != chunk_size and proc.size > 0:
                    proc = reconcile_to_length(proc, chunk_size, method="polyphase")
                emit = proc
            emit_sizes.append(int(emit.size))
            out_pieces.append(emit)
            if ctx is not None:
                if ch.size >= context_size:
                    ctx = ch[-context_size:].copy()
                else:
                    nc = np.zeros(context_size, dtype=np.float32)
                    nc[-ch.size:] = ch
                    ctx = nc
        return np.concatenate(out_pieces), emit_sizes

    print(f"\n{'method':<10} {'lag(ms)':>8} {'SNR(dB)':>9} {'HFerr%':>7} "
          f"{'dur(s)':>8} {'drift(ms)':>10} {'emit==chunk':>12}")
    input_dur = audio.size / sr
    for method in ("stretch", "silence", "lookahead"):
        cand, emit_sizes = run_method(method)
        lag, corr = _xcorr_lag(gt, cand)
        snr, hf, _m = _snr_db(gt, cand, lag)
        dur = cand.size / sr
        drift = (dur - input_dur) * 1000.0
        all_exact = all(s == chunk_size for s in emit_sizes)
        print(f"{method:<10} {lag/sr*1000:>8.2f} {snr:>9.2f} {hf*100:>7.3f} "
              f"{dur:>8.3f} {drift:>+10.2f} {str(all_exact):>12} "
              f"(corr={corr:.3f}, emit_sizes={sorted(set(emit_sizes))})")

    print(f"\ninput_duration={input_dur:.3f}s  GT_duration={gt.size/sr:.3f}s "
          f"(GT is ~20 ms short — the single-pass tail loss)")
    print("Higher SNR vs GT = closer to the offline reference. stretch is "
          "pitch-shifted (time-warped) so integer-lag alignment penalises it; "
          "lookahead/silence are not pitch-shifted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
