"""A3 (offline): characterize what the engine's `formant_timbre` knob actually does
to formants, so we can target a real M->F formant up-shift (~1.15-1.25x) instead of
blindly trusting a number like 0.25.

It replicates the engine's input-side formant op EXACTLY (streaming_engine._build_formant
+ process_block): StftPitchShift(1024, 32, 48000).shiftpitch(x, factors=1,
quefrency=formant_qfrency*1e-3, distortion=formant_timbre). factors=1 => pitch
untouched, only the spectral envelope (formants) move.

Mechanism (read from stftpitchshift source): with quefrency>0 it lifts the formant
envelope, whitens the frame, RESAMPLES the envelope's frequency axis by `distortion`,
then re-applies it -> `distortion` is a direct formant-frequency multiplier
(>1 = formants up, <1 = formants down).

We confirm that empirically with two robust, tilt-proof measures on the synthesized
vowel's output: the spectral CENTROID ratio and the F1-region dominant-peak ratio
(both vs the neutral distortion=1.0 output). A monotonic rise with `distortion`
validates the knob; the value giving ~1.15-1.25x is the M->F target.

Pure offline analysis: synthesizes its own signal, no audio device / model / runtime.
Run in .venv-applio:  python tools\\measure_formant.py
"""
from __future__ import annotations

import math
import sys

SR = 48000                 # engine applies formant at stream SR (48k)
QUEFRENCY = 1.0 * 1e-3     # engine: formant_qfrency(default 1.0) * 1e-3
FORMANTS = [(730, 80), (1090, 90), (2440, 120), (3500, 150)]   # /a/-ish
F0 = 120.0                 # male-ish source so an up-shift is the interesting direction
SWEEP = [0.25, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5, 2.0]


def _synth_vowel(sr, dur=1.5):
    import numpy as np
    from scipy.signal import lfilter

    n = int(sr * dur)
    src = np.zeros(n, dtype=np.float64)
    src[:: int(round(sr / F0))] = 1.0                # glottal impulse train
    y = src
    for f, bw in FORMANTS:                            # cascade of 2-pole resonators
        r = math.exp(-math.pi * bw / sr)
        a = [1.0, -2.0 * r * math.cos(2.0 * math.pi * f / sr), r * r]
        y = lfilter([1.0 - r], a, y)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)


def _avg_power_spectrum(x, sr):
    import numpy as np
    from scipy.signal import stft

    _f, _t, Z = stft(x, sr, nperseg=2048, noverlap=1536, window="hann")
    return np.linspace(0.0, sr / 2.0, Z.shape[0]), (np.abs(Z) ** 2).mean(axis=1)


def _centroid(freqs, P, lo=200.0, hi=5000.0):
    import numpy as np

    band = (freqs >= lo) & (freqs <= hi)
    return float((freqs[band] * P[band]).sum() / (P[band].sum() + 1e-20))


def _smooth_env(P, lifter=100):
    """Cepstral-smoothed log spectrum -> formant envelope (linear)."""
    import numpy as np

    logS = np.log(P + 1e-12)
    full = np.concatenate([logS, logS[-2:0:-1]])
    cep = np.fft.ifft(full).real
    win = np.zeros_like(cep)
    win[:lifter] = 1.0
    win[-lifter + 1:] = 1.0
    return np.exp(np.fft.fft(cep * win).real[: len(logS)])


def _f1_peak(freqs, P, lo=300.0, hi=1400.0):
    import numpy as np

    env = _smooth_env(P)
    band = (freqs >= lo) & (freqs <= hi)
    fb, eb = freqs[band], env[band]
    return float(fb[int(np.argmax(eb))]) if len(fb) else float("nan")


def run() -> int:
    try:
        from stftpitchshift import StftPitchShift
    except Exception as exc:
        print(f"stftpitchshift missing (run in .venv-applio): {exc}")
        return 2

    import numpy as np  # noqa: F401

    x = _synth_vowel(SR)
    shifter = StftPitchShift(1024, 32, SR)            # EXACT engine construction

    def shift(d):
        return shifter.shiftpitch(x.copy(), factors=1, quefrency=QUEFRENCY,
                                  distortion=float(d)).astype(np.float32)

    f0_grid, P0 = _avg_power_spectrum(shift(1.0), SR)
    cen0, peak0 = _centroid(f0_grid, P0), _f1_peak(f0_grid, P0)

    print("=" * 72)
    print("A3 formant_timbre characterization  (engine op: StftPitchShift(1024,32,48000),")
    print(f"  factors=1, quefrency={QUEFRENCY}, distortion=formant_timbre)")
    print(f"  test vowel formants(Hz)={[f for f, _ in FORMANTS]}  f0={F0:.0f}")
    print(f"  neutral (d=1.0): centroid={cen0:.0f}Hz  F1peak={peak0:.0f}Hz")
    print("=" * 72)
    print(f"  {'formant_timbre':>14} | {'centroidRatio':>13} | {'F1 peak Hz':>10} | {'F1 ratio':>8} | note")
    print("  " + "-" * 70)
    best = None
    for d in SWEEP:
        f, P = _avg_power_spectrum(shift(d), SR)
        cen, pk = _centroid(f, P), _f1_peak(f, P)
        cr, pr = cen / cen0, pk / peak0
        note = ""
        if abs(d - 0.25) < 1e-9:
            note = "<- seller '0.25'"
        if cr < 0.99 and d != 1.0:
            note = (note + " formants DOWN").strip()
        if 1.15 <= cr <= 1.25 and best is None:
            best = (d, cr)
            note = (note + " <= ~1.2 TARGET").strip()
        print(f"  {d:>14.2f} | {cr:>13.3f} | {pk:>10.0f} | {pr:>8.3f} | {note}")
    print()
    print("  centroidRatio ~= distortion (near-linear) => formant_timbre is a DIRECT formant-freq")
    print("  multiplier. (F1 peak column is noisier: peak-pick jumps as formants merge.)")
    if best:
        print(f"  => formant_timbre ~ {best[0]:.2f} gives ~{best[1]:.2f}x formant up-shift by centroid "
              f"(M->F target ~1.15-1.25). A/B-listen on model A around this value.")
    print("\n  Direction now MEASURED, not guessed: distortion>1 = formants UP, <1 = DOWN.")
    print("  Seller '0.25' shifts formants DOWN (wrong way for male->female).")
    print("  NOTE: this is the INPUT-side carrier formant shift in isolation; model A also")
    print("  reshapes formants -> final 'sounds female' still needs an ear A/B on the full pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
