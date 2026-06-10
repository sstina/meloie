"""Tests for the pure device-selection helpers.

These tests use plain dictionaries shaped like ``sounddevice.query_devices()``
records, so they require no audio hardware, no VB-CABLE install, and
no ``sounddevice`` package.
"""

from __future__ import annotations

import pytest

from meloie.audio.devices import (
    FeedbackLoopRisk,
    is_probable_cable_input,
    is_probable_cable_output,
    is_probable_virtual_cable,
    normalize_device_name,
    select_device_by_substring,
)


def _dev(name, in_ch=0, out_ch=0, sr=48000.0, hostapi=0):
    return {
        "name": name,
        "max_input_channels": in_ch,
        "max_output_channels": out_ch,
        "default_samplerate": sr,
        "hostapi": hostapi,
    }


FAKE_DEVICES = [
    _dev("Microsoft Sound Mapper - Input", in_ch=2),
    _dev("Microphone (USB Audio Device)", in_ch=1),
    _dev("CABLE Output (VB-Audio Virtual Cable)", in_ch=2),
    _dev("Microsoft Sound Mapper - Output", out_ch=2),
    _dev("Speakers (Realtek High Definition Audio)", out_ch=2),
    _dev("CABLE Input (VB-Audio Virtual Cable)", out_ch=2),
]


# ---------------------------------------------------------------------------
# Name classification
# ---------------------------------------------------------------------------

def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize_device_name("  CABLE    Input ") == "cable input"
    assert normalize_device_name("Microphone") == "microphone"


def test_cable_input_detection():
    assert is_probable_cable_input("CABLE Input (VB-Audio Virtual Cable)")
    assert not is_probable_cable_input("CABLE Output (VB-Audio Virtual Cable)")
    assert not is_probable_cable_input("Microphone (USB)")


def test_cable_output_detection():
    assert is_probable_cable_output("CABLE Output (VB-Audio Virtual Cable)")
    assert not is_probable_cable_output("CABLE Input (VB-Audio Virtual Cable)")
    assert not is_probable_cable_output("Speakers (Realtek)")


def test_virtual_cable_detection_covers_both_sides():
    assert is_probable_virtual_cable("CABLE Input (VB-Audio Virtual Cable)")
    assert is_probable_virtual_cable("CABLE Output (VB-Audio Virtual Cable)")
    assert not is_probable_virtual_cable("Microphone (USB Audio Device)")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_select_physical_microphone_as_input():
    info = select_device_by_substring(FAKE_DEVICES, "Microphone", kind="input")
    assert info.name == "Microphone (USB Audio Device)"
    assert info.is_input_capable
    assert info.index == 1


def test_app_output_selects_cable_input_not_cable_output():
    info = select_device_by_substring(FAKE_DEVICES, "CABLE Input", kind="output")
    assert info.name == "CABLE Input (VB-Audio Virtual Cable)"
    assert info.is_output_capable
    # Ensure we did not accidentally pick up the capture side
    assert not is_probable_cable_output(info.name)


def test_feedback_loop_input_selection_is_refused():
    # Asking for "CABLE Output" as input would feed the cable's capture
    # side back into the app; the helper must refuse this.
    with pytest.raises(FeedbackLoopRisk):
        select_device_by_substring(FAKE_DEVICES, "CABLE Output", kind="input")


def test_missing_device_raises_lookup_error():
    with pytest.raises(LookupError):
        select_device_by_substring(FAKE_DEVICES, "NonexistentDeviceXYZ", kind="input")


def test_kind_must_be_input_or_output():
    with pytest.raises(ValueError):
        select_device_by_substring(FAKE_DEVICES, "Microphone", kind="midi")


def test_input_kind_skips_output_only_device():
    # "Speakers" has no input channels — must not be returned as input.
    with pytest.raises(LookupError):
        select_device_by_substring(FAKE_DEVICES, "Speakers", kind="input")


def test_output_kind_skips_input_only_device():
    with pytest.raises(LookupError):
        select_device_by_substring(FAKE_DEVICES, "Microphone", kind="output")
