"""Realtime audio stream scaffolding.

Stage 1 placeholder: no streams are opened here yet. The realtime
identity stream is intentionally a guarded ``NotImplementedError`` so
that import-time code and tests can never accidentally open hardware.

The ``sounddevice`` import is lazy: it only happens inside the explicit
``list_audio_devices()`` / ``describe_devices()`` calls, so unit tests
that exercise the rest of the module do not require it to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .devices import AudioDeviceInfo, iter_device_infos


@dataclass(frozen=True)
class AudioRuntimeConfig:
    """Runtime config for the realtime audio loop.

    These values map 1-to-1 to fields in ``config/runtime.example.json``.
    """

    sample_rate: int = 48000
    block_size: int = 480           # 10 ms at 48 kHz
    channels: int = 1
    input_device_substring: str = "Microphone"
    output_device_substring: str = "CABLE Input"
    queue_blocks: int = 64
    mode: str = "identity"

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if self.queue_blocks <= 0:
            raise ValueError("queue_blocks must be > 0")
        if self.mode not in ("identity", "rvc_not_implemented"):
            raise ValueError(
                f"mode must be 'identity' or 'rvc_not_implemented', "
                f"got {self.mode!r}"
            )


@dataclass
class StreamStatusSnapshot:
    """Lightweight status snapshot, used once realtime is wired up.

    Kept here so the field shape is fixed across stages.
    """

    running: bool = False
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    notes: List[str] = field(default_factory=list)


def list_audio_devices() -> List[AudioDeviceInfo]:
    """Return wrapped ``AudioDeviceInfo`` records for the host.

    Imports ``sounddevice`` lazily. If it is not installed, raises
    ``RuntimeError`` with a clear message — the rest of the package
    (and the unit tests) continues to work without it.
    """
    try:
        import sounddevice as sd  # noqa: WPS433  (deliberately lazy)
    except ImportError as exc:  # pragma: no cover  - env-dependent
        raise RuntimeError(
            "sounddevice is not installed; cannot enumerate devices. "
            "Install it with `pip install sounddevice` when ready."
        ) from exc

    raw = sd.query_devices()
    return list(iter_device_infos(raw))


def describe_devices() -> str:
    """Human-readable device listing. Lazy-imports ``sounddevice``."""
    infos = list_audio_devices()
    lines = ["index  in  out  name"]
    for info in infos:
        lines.append(
            f"{info.index:>5}  "
            f"{info.max_input_channels:>2}  "
            f"{info.max_output_channels:>3}  "
            f"{info.name}"
        )
    return "\n".join(lines)


def run_identity_stream(config: AudioRuntimeConfig) -> None:
    """Stage 1 realtime identity loop — intentionally not implemented.

    The Stage 1 design (mic -> in_queue -> identity worker -> out_queue
    -> CABLE Input) is documented in ``rvc.md`` and will be implemented
    in a follow-up commit. This skeleton refuses to open any audio
    stream so that nothing here can accidentally start recording.
    """
    config.validate()
    raise NotImplementedError(
        "Stage 1 realtime identity streaming is intentionally not "
        "implemented in this skeleton. The async-queue audio loop "
        "(input callback -> in_queue -> worker -> out_queue -> output "
        "callback) will be wired up in the next commit. No audio "
        "streams are opened by this skeleton."
    )
