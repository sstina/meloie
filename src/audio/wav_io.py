"""WAV file I/O helpers, stdlib only.

Reads mono float32 from any standard PCM WAV (8/16/24/32-bit); writes
mono 16-bit PCM (the most compatible format for downstream tooling).

These helpers are pure: no audio hardware, no sounddevice, no heavy
dependencies. ``wave`` (stdlib) handles the file format; numpy does
the bit-width conversions.
"""

from __future__ import annotations

import wave
from typing import Tuple

import numpy as np


def read_wav_mono_float32(path: str) -> Tuple[np.ndarray, int]:
    """Read a WAV file as a 1-D float32 numpy array in [-1, 1].

    Multichannel WAVs are downmixed to mono by averaging channels.
    Supports 8-, 16-, 24-, and 32-bit signed PCM (the formats stdlib
    ``wave`` understands).
    """
    with wave.open(str(path), "rb") as wav:
        nchannels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        sample_rate = wav.getframerate()
        nframes = wav.getnframes()
        raw = wav.readframes(nframes)

    if sampwidth == 1:
        # 8-bit WAV is unsigned in the stdlib.
        ints = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        data = (ints.astype(np.float32)) / 128.0
    elif sampwidth == 2:
        ints = np.frombuffer(raw, dtype="<i2")
        data = ints.astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit signed PCM, little-endian. Vectorised unpack into int32.
        n_samples_total = len(raw) // 3
        b = np.frombuffer(raw, dtype=np.uint8).reshape(n_samples_total, 3).astype(np.int32)
        ints = (b[:, 0]) | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        ints = np.frombuffer(raw, dtype="<i4")
        data = ints.astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

    if nchannels > 1:
        data = data.reshape(-1, nchannels).mean(axis=1)
    data = data.astype(np.float32, copy=False)
    return data, int(sample_rate)


def write_wav_float32(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write a 1-D mono float32 array as a 16-bit PCM WAV.

    Values are clipped to [-1, 1] before conversion. ``sample_rate``
    is recorded verbatim — pass the value the audio is meant to be
    played back at.
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a numpy array, got {type(audio).__name__}"
        )
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D mono, got shape {audio.shape}")
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be > 0")

    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm16.tobytes())
