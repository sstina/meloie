"""Tests for stdlib-only WAV I/O helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.wav_io import read_wav_mono_float32, write_wav_float32


def test_roundtrip_mono_pcm16(tmp_path):
    sr = 48000
    t = np.arange(sr // 4, dtype=np.float64) / sr
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    write_wav_float32(str(path), audio, sr)

    loaded, loaded_sr = read_wav_mono_float32(str(path))
    assert loaded_sr == sr
    assert loaded.dtype == np.float32
    assert loaded.shape == audio.shape
    # 16-bit quantisation noise ~ 1/32767 ~ 3e-5
    np.testing.assert_allclose(loaded, audio, atol=2e-4)


def test_silence_roundtrip(tmp_path):
    silence = np.zeros(4800, dtype=np.float32)
    path = tmp_path / "silence.wav"
    write_wav_float32(str(path), silence, 48000)
    loaded, sr = read_wav_mono_float32(str(path))
    assert sr == 48000
    np.testing.assert_array_equal(loaded, silence)


def test_clip_above_full_scale(tmp_path):
    audio = np.array([1.5, -1.5, 0.5, -0.5], dtype=np.float32)
    path = tmp_path / "loud.wav"
    write_wav_float32(str(path), audio, 8000)
    loaded, _ = read_wav_mono_float32(str(path))
    assert float(np.max(loaded)) <= 1.0
    assert float(np.min(loaded)) >= -1.0


def test_write_rejects_stereo(tmp_path):
    stereo = np.zeros((100, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        write_wav_float32(str(tmp_path / "x.wav"), stereo, 48000)


def test_write_rejects_zero_sample_rate(tmp_path):
    with pytest.raises(ValueError):
        write_wav_float32(str(tmp_path / "x.wav"), np.zeros(10, dtype=np.float32), 0)


def test_write_rejects_non_array(tmp_path):
    with pytest.raises(TypeError):
        write_wav_float32(str(tmp_path / "x.wav"), [0.0, 0.1], 48000)  # type: ignore[arg-type]


def test_read_downmixes_stereo_to_mono(tmp_path):
    """Write a stereo WAV by hand and confirm the reader averages channels."""
    import wave

    sr = 16000
    left = np.full(800, 0.4, dtype=np.float32)
    right = np.full(800, -0.4, dtype=np.float32)
    inter = np.empty(2 * left.size, dtype=np.float32)
    inter[0::2] = left
    inter[1::2] = right
    pcm = np.clip(inter, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())

    loaded, loaded_sr = read_wav_mono_float32(str(path))
    assert loaded_sr == sr
    assert loaded.shape == (800,)
    # mean of (0.4, -0.4) ~= 0
    assert float(np.max(np.abs(loaded))) < 1e-3
