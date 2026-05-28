"""Pure measurement helpers for Stage 1 validation tools.

Used by ``tools/click_test.py`` (identity-path latency) and
``tools/verify_cable_route.py`` (non-silence check on the cable route).

These helpers are deliberately free of any sounddevice / hardware
dependency so they can be unit tested with synthetic numpy arrays.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..safety.guard import dbfs_peak, dbfs_rms


# ---------------------------------------------------------------------------
# Click / pulse generation
# ---------------------------------------------------------------------------

def generate_click_pulse(
    sample_rate: int,
    pulse_amplitude: float = 0.5,
    pre_silence_seconds: float = 0.3,
    post_silence_seconds: float = 1.5,
    click_samples: int = 16,
) -> np.ndarray:
    """Generate a [pre_silence | windowed click | post_silence] mono signal.

    The click is a short triangular pulse with ``click_samples`` total
    length, peaking at ``pulse_amplitude``. The temporal narrowness
    gives a sharp cross-correlation peak; the small window avoids the
    DC step that a single-sample impulse would create.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if click_samples <= 0:
        raise ValueError("click_samples must be > 0")
    if pre_silence_seconds < 0 or post_silence_seconds < 0:
        raise ValueError("silence durations must be >= 0")
    if not (0.0 < pulse_amplitude <= 1.0):
        raise ValueError("pulse_amplitude must be in (0, 1]")

    pre_n = int(round(pre_silence_seconds * sample_rate))
    post_n = int(round(post_silence_seconds * sample_rate))
    n = pre_n + click_samples + post_n
    signal = np.zeros(n, dtype=np.float32)

    half = click_samples // 2
    if half == 0:
        signal[pre_n] = pulse_amplitude
        return signal

    ramp_up = np.linspace(0.0, pulse_amplitude, half, endpoint=False, dtype=np.float32)
    ramp_down = np.linspace(
        pulse_amplitude, 0.0, click_samples - half, endpoint=False, dtype=np.float32
    )
    pulse = np.concatenate([ramp_up, ramp_down])
    signal[pre_n:pre_n + click_samples] = pulse
    return signal


def generate_sine_tone(
    sample_rate: int,
    duration_seconds: float,
    frequency_hz: float = 440.0,
    amplitude: float = 0.25,
) -> np.ndarray:
    """Generate a mono sine tone for cable-route verification."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0")
    if not (0.0 < amplitude <= 1.0):
        raise ValueError("amplitude must be in (0, 1]")

    n = int(round(duration_seconds * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    signal = (amplitude * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)
    return signal


# ---------------------------------------------------------------------------
# Cross-correlation latency estimator
# ---------------------------------------------------------------------------

def estimate_latency_samples(
    reference: np.ndarray,
    captured: np.ndarray,
    max_lag_samples: Optional[int] = None,
) -> Tuple[int, float]:
    """Estimate the lag of ``captured`` relative to ``reference`` in samples.

    Uses FFT-based cross-correlation with zero padding so it is fast
    and not subject to circular wrap-around aliasing. Only positive
    lags are searched (the captured signal cannot lead the reference
    in a real loopback test).

    Returns ``(lag_samples, normalised_peak)`` where
    ``normalised_peak`` is the absolute correlation peak divided by
    ``sqrt(sum(ref^2) * sum(cap^2))`` — i.e. roughly the cosine
    similarity at the best alignment, in [0, 1] for clean signals.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cap = np.asarray(captured, dtype=np.float64).ravel()
    if ref.size == 0 or cap.size == 0:
        raise ValueError("reference and captured must be non-empty")

    # Zero-pad to >= len(ref) + len(cap) to avoid circular aliasing,
    # then to the next power of two for FFT speed.
    needed = ref.size + cap.size
    nfft = 1
    while nfft < needed:
        nfft <<= 1

    R = np.fft.rfft(ref, nfft)
    C = np.fft.rfft(cap, nfft)
    xc = np.fft.irfft(np.conj(R) * C, nfft)

    # Positive-lag region: xc[0 .. cap.size-1].
    upper = cap.size - 1
    if max_lag_samples is not None:
        upper = min(upper, int(max_lag_samples))
    upper = max(0, min(upper, xc.size - 1))

    search = xc[: upper + 1]
    lag = int(np.argmax(np.abs(search)))
    peak = float(np.abs(search[lag]))

    norm = float(np.sqrt(np.sum(ref * ref) * np.sum(cap * cap)))
    normalised = peak / norm if norm > 0.0 else 0.0
    return lag, normalised


# ---------------------------------------------------------------------------
# Non-silence detection + capture summary
# ---------------------------------------------------------------------------

DEFAULT_NON_SILENCE_THRESHOLD_DBFS = -60.0


def is_non_silent(
    audio: np.ndarray,
    threshold_dbfs: float = DEFAULT_NON_SILENCE_THRESHOLD_DBFS,
) -> bool:
    """True if the audio's peak level is above ``threshold_dbfs``.

    The threshold defaults to -60 dBFS — well above the noise floor of
    a healthy capture path but below any deliberately rendered signal.
    """
    return bool(dbfs_peak(audio) > float(threshold_dbfs))


def summarize_capture(
    audio: np.ndarray,
    threshold_dbfs: float = DEFAULT_NON_SILENCE_THRESHOLD_DBFS,
) -> Dict[str, Any]:
    """Return a JSON-serialisable summary of a captured buffer."""
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a numpy array, got {type(audio).__name__}"
        )
    peak = float(dbfs_peak(audio))
    rms = float(dbfs_rms(audio))
    return {
        "n_samples": int(audio.size),
        "peak_dbfs": peak,
        "rms_dbfs": rms,
        "non_silent": bool(peak > float(threshold_dbfs)),
        "threshold_dbfs": float(threshold_dbfs),
    }
