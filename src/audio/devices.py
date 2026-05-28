"""Audio device selection helpers — pure, hardware-free.

The helpers in this module work on plain dictionaries shaped like the
records returned by ``sounddevice.query_devices()``. They never import
``sounddevice`` and never touch hardware, so they are safe to unit test
without VB-CABLE, a microphone, or any audio backend installed.

VB-CABLE routing rule (do not invert):
    * The app renders audio to ``CABLE Input``  (virtual cable render side).
    * Discord / OBS / Zoom select ``CABLE Output`` as their microphone
      (virtual cable capture side).
    * The app's own input device must be a physical microphone — never
      the virtual cable's capture side, or a feedback loop is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence


DeviceKind = str  # "input" or "output"


@dataclass(frozen=True)
class AudioDeviceInfo:
    """A normalised view of one sounddevice device record."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi: int

    @property
    def is_input_capable(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output_capable(self) -> bool:
        return self.max_output_channels > 0


def normalize_device_name(name: str) -> str:
    """Lowercase + collapse runs of whitespace. Used for substring matching."""
    if name is None:
        return ""
    return " ".join(str(name).split()).lower()


def is_probable_cable_input(name: str) -> bool:
    """True if the device name looks like the VB-CABLE *render* endpoint.

    ``CABLE Input`` is what the app should select as its **output** device.
    """
    n = normalize_device_name(name)
    return "cable" in n and "input" in n


def is_probable_cable_output(name: str) -> bool:
    """True if the device name looks like the VB-CABLE *capture* endpoint.

    ``CABLE Output`` is what Discord / OBS / Zoom select as their
    microphone. The app should normally **not** use this as its input.
    """
    n = normalize_device_name(name)
    return "cable" in n and "output" in n


def is_probable_virtual_cable(name: str) -> bool:
    """True for any virtual-cable side (either render or capture)."""
    return is_probable_cable_input(name) or is_probable_cable_output(name)


def _as_info(raw: Mapping, index: int) -> AudioDeviceInfo:
    return AudioDeviceInfo(
        index=index,
        name=str(raw.get("name", "")),
        max_input_channels=int(raw.get("max_input_channels", 0) or 0),
        max_output_channels=int(raw.get("max_output_channels", 0) or 0),
        default_samplerate=float(raw.get("default_samplerate", 0.0) or 0.0),
        hostapi=int(raw.get("hostapi", 0) or 0),
    )


def iter_device_infos(devices: Iterable[Mapping]) -> Sequence[AudioDeviceInfo]:
    """Wrap raw sounddevice records into ``AudioDeviceInfo``s, preserving index."""
    return [_as_info(d, i) for i, d in enumerate(devices)]


class FeedbackLoopRisk(ValueError):
    """Raised when a selection would route the app's input through the
    same virtual cable it is rendering to — i.e. a feedback loop."""


def select_device_by_substring(
    devices: Iterable[Mapping],
    substring: str,
    kind: DeviceKind,
) -> AudioDeviceInfo:
    """Select the first device whose name contains ``substring`` and that
    supports ``kind`` (``"input"`` or ``"output"``).

    Refuses to return ``CABLE Output`` as an input device — that would
    feed the virtual cable's capture side back into the app and form a
    feedback loop with the cable's render side.
    """
    if kind not in ("input", "output"):
        raise ValueError(f"kind must be 'input' or 'output', got {kind!r}")

    needle = normalize_device_name(substring)
    if not needle:
        raise ValueError("substring must be a non-empty string")

    infos = iter_device_infos(devices)
    for info in infos:
        if needle not in normalize_device_name(info.name):
            continue
        if kind == "input" and not info.is_input_capable:
            continue
        if kind == "output" and not info.is_output_capable:
            continue
        if kind == "input" and is_probable_cable_output(info.name):
            raise FeedbackLoopRisk(
                f"refusing to use {info.name!r} as input: it is the "
                "VB-CABLE capture side; using it as the app's mic creates "
                "a feedback loop. Pick a physical microphone instead."
            )
        return info

    raise LookupError(
        f"no {kind} device matched substring {substring!r} "
        f"(searched {len(infos)} devices)"
    )
