"""Tests for the pure device-resolution helpers in ``audio.streams``.

These tests work entirely against fake device dictionaries shaped like
``sounddevice.query_devices()`` records, so they require no audio
hardware, no VB-CABLE install, and no ``sounddevice`` package.

Importantly: ``run_identity_stream`` is NOT called from these tests —
that function opens real audio streams. We only exercise the pure
resolution helpers it uses.
"""

from __future__ import annotations

import pytest

from src.audio.devices import FeedbackLoopRisk, select_default_input_device
from src.audio.streams import (
    AudioRuntimeConfig,
    queue_blocks_from_ms,
    resolve_input_device,
    resolve_output_device,
)


def _dev(name, in_ch=0, out_ch=0):
    return {
        "name": name,
        "max_input_channels": in_ch,
        "max_output_channels": out_ch,
        "default_samplerate": 48000.0,
        "hostapi": 0,
    }


# Some virtual-cable drivers register both directions on each endpoint.
# Cover that here so the output guard cannot be silently bypassed.
FAKE_DEVICES = [
    _dev("Microphone (USB Audio Device)", in_ch=1),
    _dev("CABLE Output (VB-Audio Virtual Cable)", in_ch=2, out_ch=2),
    _dev("Speakers (Realtek High Definition Audio)", out_ch=2),
    _dev("CABLE Input (VB-Audio Virtual Cable)", in_ch=2, out_ch=2),
]


# ---------------------------------------------------------------------------
# resolve_input_device
# ---------------------------------------------------------------------------

def test_resolve_input_picks_physical_microphone():
    info = resolve_input_device(FAKE_DEVICES, "Microphone")
    assert info.name == "Microphone (USB Audio Device)"


def test_resolve_input_refuses_cable_output_by_default():
    with pytest.raises(FeedbackLoopRisk):
        resolve_input_device(FAKE_DEVICES, "CABLE Output")


def test_resolve_input_diagnostic_override_allows_cable_output():
    info = resolve_input_device(
        FAKE_DEVICES, "CABLE Output", allow_virtual_cable=True
    )
    assert "CABLE Output" in info.name


def test_resolve_input_missing_raises_lookup_error():
    with pytest.raises(LookupError):
        resolve_input_device(FAKE_DEVICES, "Nonexistent")


def test_resolve_input_missing_with_override_raises_lookup_error():
    with pytest.raises(LookupError):
        resolve_input_device(
            FAKE_DEVICES, "Nonexistent", allow_virtual_cable=True
        )


# ---------------------------------------------------------------------------
# resolve_output_device
# ---------------------------------------------------------------------------

def test_resolve_output_picks_cable_input():
    info = resolve_output_device(FAKE_DEVICES, "CABLE Input")
    assert info.name == "CABLE Input (VB-Audio Virtual Cable)"


def test_resolve_output_refuses_cable_output_even_if_output_capable():
    # CABLE Output here has output channels (some driver versions do),
    # but the helper must still refuse it: rendering Discord's outbound
    # audio back into the cable would loop.
    with pytest.raises(FeedbackLoopRisk):
        resolve_output_device(FAKE_DEVICES, "CABLE Output")


def test_resolve_output_picks_physical_speakers():
    info = resolve_output_device(FAKE_DEVICES, "Speakers")
    assert info.name == "Speakers (Realtek High Definition Audio)"


def test_resolve_output_missing_raises_lookup_error():
    with pytest.raises(LookupError):
        resolve_output_device(FAKE_DEVICES, "Nonexistent")


# ---------------------------------------------------------------------------
# AudioRuntimeConfig validation
# ---------------------------------------------------------------------------

def test_audio_runtime_config_defaults_validate():
    AudioRuntimeConfig().validate()


def test_audio_runtime_config_rejects_zero_sample_rate():
    with pytest.raises(ValueError):
        AudioRuntimeConfig(sample_rate=0).validate()


def test_audio_runtime_config_rejects_non_mono():
    # The route is mono end-to-end; stereo would drop/garble a channel.
    with pytest.raises(ValueError):
        AudioRuntimeConfig(channels=2).validate()
    with pytest.raises(ValueError):
        AudioRuntimeConfig(channels=3).validate()


def test_audio_runtime_config_default_input_is_system_default():
    # None means "follow the Windows default recording device".
    assert AudioRuntimeConfig().input_device_substring is None


# ---------------------------------------------------------------------------
# select_default_input_device (the "系统默认 mic" path)
# ---------------------------------------------------------------------------

def test_default_input_picks_device_at_default_index():
    info = select_default_input_device(FAKE_DEVICES, 0)
    assert info.name == "Microphone (USB Audio Device)"


def test_default_input_refuses_cable_output():
    # index 1 in FAKE_DEVICES is the VB-CABLE capture side.
    with pytest.raises(FeedbackLoopRisk):
        select_default_input_device(FAKE_DEVICES, 1)


def test_default_input_rejects_output_only_device():
    # index 2 (Speakers) has no input channels.
    with pytest.raises(LookupError):
        select_default_input_device(FAKE_DEVICES, 2)


def test_default_input_rejects_invalid_index():
    with pytest.raises(LookupError):
        select_default_input_device(FAKE_DEVICES, -1)
    with pytest.raises(LookupError):
        select_default_input_device(FAKE_DEVICES, 999)


# ---------------------------------------------------------------------------
# Stage 2E: queue_blocks_from_ms
# ---------------------------------------------------------------------------

def test_queue_blocks_from_ms_basic_48k_480():
    # 1000 ms at 48 kHz / 480 sample blocks = 1000 / 10 = 100 blocks
    assert queue_blocks_from_ms(1000.0, 480, 48000, minimum=1) == 100


def test_queue_blocks_from_ms_default_minimum_clamps_small_input():
    # tiny ms request should still give at least the minimum
    assert queue_blocks_from_ms(50.0, 480, 48000, minimum=64) == 64


def test_queue_blocks_from_ms_six_seconds_at_48k_480():
    # 6000 ms = 600 blocks (this is the new RVC default)
    assert queue_blocks_from_ms(6000.0, 480, 48000, minimum=64) == 600


def test_queue_blocks_from_ms_zero_yields_minimum():
    assert queue_blocks_from_ms(0.0, 480, 48000, minimum=64) == 64


def test_queue_blocks_from_ms_rejects_zero_block_size():
    with pytest.raises(ValueError):
        queue_blocks_from_ms(1000.0, 0, 48000)


def test_queue_blocks_from_ms_rejects_zero_sample_rate():
    with pytest.raises(ValueError):
        queue_blocks_from_ms(1000.0, 480, 0)
