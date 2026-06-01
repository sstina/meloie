"""A1 (read-only): probe whether a model's learned pitch embedding (enc_p.emb_pitch)
marks out its trained F0 range / comfort band, WITHOUT any reference audio or runtime.

Method (see docs/f0_remap_plan.md):
  - emb_pitch[k] is one learned row per coarse-F0 bucket k in 1..255 (256-wide
    Embedding). Buckets the model was trained on get pushed around (sparse grad);
    unused buckets only shrink under weight-decay. So the *active band* of rows
    approximates the model's comfort F0 range.
  - We stay BASE-FREE (no external pretrained base needed for A): per-model row-norm
    profile ||emb_M[k]||, plus cross-model differencing emb_A - emb_mean(others)
    (the shared pretrain base cancels to first order). Boundaries via Delta-weighted
    2/98 percentiles (robust to a stray creak/octave-error bucket); centroid via the
    Delta-weighted mean bucket (base-immune).
  - bucket -> Hz by inverting the RVC mel quantization (f0_min/f0_max default 50/1100).

go/no-go (NOT the circular '+12 match'):
  (a) is a model's Delta-vs-bucket curve a clean unimodal active band?
  (b) cross-model octave: a high/female model's centroid Hz ~= 2x a low/male one's?

Pure analysis: reads model weights only, touches no audio device / runtime / network.
Run in .venv-applio:  python tools\\analyze_model_f0.py
"""
from __future__ import annotations

import math
import os
import sys

# RVC root = parent of this tools/ dir.
RVC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# RVC mel quantization constants (vendored get_f0 uses f0_min=50, f0_max=1100).
F0_MIN_HZ = 50.0
F0_MAX_HZ = 1100.0
_SPARK = " .:-=+*#%@"   # 10 levels, low->high


def _mel(hz: float) -> float:
    return 1127.0 * math.log(1.0 + hz / 700.0)


def _bucket_to_hz(bucket: float) -> float:
    """Invert the RVC coarse-F0 quantization: bucket in [1,255] -> Hz."""
    mel_min, mel_max = _mel(F0_MIN_HZ), _mel(F0_MAX_HZ)
    mel = (bucket - 1.0) * (mel_max - mel_min) / 254.0 + mel_min
    return 700.0 * (math.exp(mel / 1127.0) - 1.0)


def _sparkline(vals, width: int = 72) -> str:
    """Downsample a 1-D array to `width` columns and render as an ASCII sparkline."""
    import numpy as np

    v = np.asarray(vals, dtype=float)
    n = len(v)
    cols = []
    for c in range(width):
        a = int(round(c * n / width))
        b = int(round((c + 1) * n / width))
        b = max(b, a + 1)
        cols.append(float(v[a:b].mean()))
    cols = np.asarray(cols)
    lo, hi = float(cols.min()), float(cols.max())
    if hi - lo < 1e-12:
        return _SPARK[0] * width
    idx = np.clip(((cols - lo) / (hi - lo) * (len(_SPARK) - 1)).round().astype(int), 0, len(_SPARK) - 1)
    return "".join(_SPARK[i] for i in idx)


def _weighted_percentile(positions, weights, pct):
    """Delta-weighted percentile of bucket positions (pct in [0,1])."""
    import numpy as np

    pos = np.asarray(positions, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(pos)
    pos, w = pos[order], w[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(pos[len(pos) // 2])
    cw = cw / cw[-1]
    return float(np.interp(pct, cw, pos))


def _find_emb_pitch(weight_dict):
    for key in weight_dict:
        if "emb_pitch" in key and key.endswith(".weight"):
            return key
    return None


def _load_models():
    """Return list of dicts {name, emb (np [256,h]), version, sr, f0, key}."""
    import numpy as np
    import torch

    models_dir = os.path.join(RVC_ROOT, "models")
    out = []
    for fn in sorted(os.listdir(models_dir)):
        if not fn.endswith(".pth"):
            continue
        path = os.path.join(models_dir, fn)
        try:
            cpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            cpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(cpt, dict) or "weight" not in cpt:
            print(f"  [skip] {fn}: not an RVC checkpoint dict")
            continue
        wd = cpt["weight"]
        key = _find_emb_pitch(wd)
        if key is None:
            print(f"  [skip] {fn}: no emb_pitch (not F0-conditioned?)")
            continue
        emb = wd[key].detach().to(torch.float32).cpu().numpy()
        cfg = cpt.get("config")
        sr = cfg[-1] if isinstance(cfg, (list, tuple)) and cfg else cpt.get("sr")
        out.append({
            "name": os.path.splitext(fn)[0],
            "emb": emb,
            "version": str(cpt.get("version", "?")),
            "sr": sr,
            "f0": cpt.get("f0", "?"),
            "key": key,
        })
    return out


def _band_stats(activity):
    """Given per-bucket activity over buckets 1..255, return centroid/2-98 boundaries
    (bucket + Hz) and a cleanliness metric (mass fraction in the contiguous main band)."""
    import numpy as np

    a = np.asarray(activity, dtype=float)            # index 0 == bucket1
    buckets = np.arange(1, len(a) + 1, dtype=float)
    w = np.clip(a - np.median(a), 0.0, None)         # lift floor so untrained buckets ~0
    if w.sum() <= 0:
        w = a.copy()
    centroid_b = float((buckets * w).sum() / w.sum())
    lo_b = _weighted_percentile(buckets, w, 0.02)
    hi_b = _weighted_percentile(buckets, w, 0.98)
    # cleanliness: fraction of weight within +-1 octave-ish window (35 buckets) of centroid
    win = np.abs(buckets - centroid_b) <= 35
    main_frac = float(w[win].sum() / w.sum())
    # unimodality proxy: peak / (median of nonzero) ratio
    nz = w[w > 0]
    peak_ratio = float(w.max() / (np.median(nz) + 1e-9)) if len(nz) else 0.0
    return {
        "centroid_b": centroid_b, "centroid_hz": _bucket_to_hz(centroid_b),
        "lo_b": lo_b, "lo_hz": _bucket_to_hz(lo_b),
        "hi_b": hi_b, "hi_hz": _bucket_to_hz(hi_b),
        "main_frac": main_frac, "peak_ratio": peak_ratio,
    }


def run() -> int:
    import numpy as np

    print("=" * 78)
    print("A1 emb_pitch comfort-range probe (read-only)")
    print("RVC mel quant: f0_min=%.0f f0_max=%.0f Hz | bucket1=%.1fHz bucket255=%.1fHz"
          % (F0_MIN_HZ, F0_MAX_HZ, _bucket_to_hz(1), _bucket_to_hz(255)))
    print("=" * 78)

    models = _load_models()
    if not models:
        print("no F0-conditioned models found under models/")
        return 1

    # compatibility for cross-model differencing (same emb shape)
    shapes = {m["name"]: m["emb"].shape for m in models}
    base_shape = models[0]["emb"].shape
    comparable = [m for m in models if m["emb"].shape == base_shape]

    print("\nloaded models:")
    for m in models:
        print(f"  {m['name']:>6}  emb{m['emb'].shape}  ver={m['version']} sr={m['sr']} f0={m['f0']}  key={m['key']}")
    if len(comparable) < len(models):
        print("  (note: only same-shape models are cross-compared)")

    # per-bucket signals: row norm (single-model) and cross-model diff vs mean-of-others
    embs = {m["name"]: m["emb"] for m in comparable}
    names = [m["name"] for m in comparable]
    stack = np.stack([embs[n] for n in names], axis=0)        # [M,256,h]
    rownorm = {n: np.linalg.norm(embs[n][1:], axis=1) for n in names}   # buckets 1..255

    # speaking-range buckets ~ 80..300 Hz (where real voice F0 lives)
    def _bucket_of(hz):
        return (_mel(hz) - _mel(F0_MIN_HZ)) * 254.0 / (_mel(F0_MAX_HZ) - _mel(F0_MIN_HZ)) + 1.0
    sp_lo, sp_hi = _bucket_of(80.0), _bucket_of(300.0)

    def _peak(sig):
        i = int(np.argmax(sig))
        return i + 1, _bucket_to_hz(i + 1)

    def _speaking_frac(sig):
        a = np.asarray(sig, float)
        w = np.clip(a - np.median(a), 0.0, None)
        if w.sum() <= 0:
            return 0.0
        bk = np.arange(1, len(a) + 1)
        return float(w[(bk >= sp_lo) & (bk <= sp_hi)].sum() / w.sum())

    print(f"\n(speaking-range gate: ~80..300 Hz == buckets {sp_lo:.0f}..{sp_hi:.0f})")
    print("\n--- per-model row-norm  ||emb[k]||  (buckets 1..255, low Hz -> high Hz) ---")
    for n in names:
        s = _band_stats(rownorm[n])
        pb, ph = _peak(rownorm[n])
        print(f"  {n:>6} |{_sparkline(rownorm[n])}|")
        print(f"         peak b{pb}~{ph:.0f}Hz | centroid {s['centroid_hz']:.0f}Hz | "
              f"speakingFrac {_speaking_frac(rownorm[n]):.2f}")

    # CORRECT cross-diff: only WITHIN a shared-base group (same sr -> same f0G base),
    # so the shared pretrain base cancels. With 2 members/group this is the pair diff.
    sr_of = {m["name"]: m["sr"] for m in comparable}
    by_sr = {}
    for n in names:
        by_sr.setdefault(sr_of[n], []).append(n)

    print("\n--- WITHIN-base pairwise diff  ||emb_i - emb_j||  (same sr; base cancels) ---")
    pair_results = []
    for sr, grp in sorted(by_sr.items(), key=lambda kv: str(kv[0])):
        if len(grp) < 2:
            print(f"  sr={sr}: only {grp} -> cannot pair within base")
            continue
        for ii in range(len(grp)):
            for jj in range(ii + 1, len(grp)):
                a, b = grp[ii], grp[jj]
                d = np.linalg.norm((embs[a] - embs[b])[1:], axis=1)
                s = _band_stats(d)
                pb, ph = _peak(d)
                spf = _speaking_frac(d)
                pair_results.append((f"{a}-{b}", spf, ph))
                print(f"  {a}-{b} (sr={sr}) |{_sparkline(d)}|")
                print(f"         peak b{pb}~{ph:.0f}Hz | centroid {s['centroid_hz']:.0f}Hz | "
                      f"speakingFrac {spf:.2f}  (>=0.5 & peak in 80-300Hz = real usage signal)")

    print("\n--- VERDICT ---")
    rn_peaks = [_peak(rownorm[n])[1] for n in names]
    rn_speak = [_speaking_frac(rownorm[n]) for n in names]
    rownorm_is_base = all(p > 400 for p in rn_peaks)
    pair_ok = any(spf >= 0.5 and 80 <= ph <= 300 for _, spf, ph in pair_results)
    print(f"  row-norm peaks (Hz): {[round(p) for p in rn_peaks]}  speakingFrac: {[round(x,2) for x in rn_speak]}")
    if rownorm_is_base:
        print("  -> row-norm peaks are all >400Hz (non-speech) and ~identical across models:")
        print("     it tracks the BASE emb_pitch magnitude structure, NOT per-model training usage.")
    if pair_results and not pair_ok:
        print("  -> within-base pairwise diffs show NO clean band in the 80-300Hz speaking range.")
    if rownorm_is_base and (not pair_results or not pair_ok):
        print("  NO-GO (base-free): emb_pitch weights do not recover the trained F0 range for these")
        print("  checkpoints (fine-tune barely moved emb_pitch / base dominates). Per plan: fall back")
        print("  to ear/seller-seeded target_f0_median; optionally try A1b (true f0G40k/48k base, norm-")
        print("  alized delta) but expect weak signal for short fine-tunes.")
    elif pair_ok:
        print("  PARTIAL GO: a within-base pair shows a speaking-range band -> emb_pitch carries usage")
        print("  signal; worth pursuing A1b with the true base for absolute boundaries.")
    print("\nNOTE: Hz assumes training f0_min/f0_max = 50/1100. Cross-sr (40k vs 48k) diffs are NOT")
    print("comparable (different base) and are intentionally excluded.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
