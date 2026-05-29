"""Record a named input device to a WAV — Stage 4-F downstream consumer.

A minimal "downstream app" that uses CABLE Output (the VB-CABLE capture
endpoint) as its input and writes what it receives to a WAV, so the
operator can A/B the live converted route against the offline reference.

It does NOT touch the realtime runtime. It only *reads* CABLE Output
(the downstream side of the cable) — there is no feedback loop because
it never renders back to the mic or to CABLE Input.

Run it concurrently with `python -m src.main --mode rvc ...` (which
renders to CABLE Input):

  python -m tools.record_device --device-substring "CABLE Output" \
      --duration-seconds 75 --output-wav audit_s4f_downstream_60s.wav

Output WAVs are gitignored (*.wav) — diagnostic artifacts, not source.
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from typing import List, Optional

import numpy as np


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="record-device")
    p.add_argument("--device-substring", default="CABLE Output",
                   help="Input device name substring to capture from.")
    p.add_argument("--output-wav", required=True,
                   help="Output WAV path (gitignored).")
    p.add_argument("--duration-seconds", type=float, required=True)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--block-size", type=int, default=480)
    p.add_argument("--channels", type=int, default=1)
    args = p.parse_args(argv)

    import sounddevice as sd
    from src.audio.devices import iter_device_infos, normalize_device_name
    from src.audio.wav_io import write_wav_float32
    from src.safety.guard import dbfs_peak, dbfs_rms

    raw = list(sd.query_devices())
    needle = normalize_device_name(args.device_substring)
    chosen = None
    for info in iter_device_infos(raw):
        if needle in normalize_device_name(info.name) and info.is_input_capable:
            chosen = info
            break
    if chosen is None:
        print(f"error: no input device matched {args.device_substring!r}",
              file=sys.stderr)
        return 2
    print(f"recording from [{chosen.index}] {chosen.name} "
          f"for {args.duration_seconds:.0f}s @ {args.sample_rate} Hz")

    q: "queue.Queue" = queue.Queue()
    captured = 0
    flags = 0

    def cb(indata, frames, time_info, status):  # noqa: ANN001
        nonlocal flags
        if status:
            flags += 1
        q.put_nowait(indata[:, 0].astype(np.float32, copy=True))

    pieces: List[np.ndarray] = []
    start = time.monotonic()
    with sd.InputStream(device=chosen.index, callback=cb,
                        samplerate=args.sample_rate, channels=args.channels,
                        blocksize=args.block_size, dtype="float32",
                        latency="low"):
        print("recording... ", flush=True)
        while time.monotonic() - start < args.duration_seconds:
            try:
                blk = q.get(timeout=0.2)
                pieces.append(blk)
                captured += blk.size
            except queue.Empty:
                pass
    # Drain remaining
    while True:
        try:
            pieces.append(q.get_nowait())
        except queue.Empty:
            break

    audio = (np.concatenate(pieces).astype(np.float32, copy=False)
             if pieces else np.zeros(0, dtype=np.float32))
    write_wav_float32(args.output_wav, audio, args.sample_rate)
    dur = audio.size / float(args.sample_rate)
    print(f"wrote {args.output_wav}: {audio.size} samples ({dur:.2f}s)  "
          f"peak={dbfs_peak(audio):.2f} dBFS  rms={dbfs_rms(audio):.2f} dBFS  "
          f"status_flags={flags}")
    if dbfs_peak(audio) <= -60.0:
        print("WARNING: captured audio is near-silent — CABLE Output may not "
              "be carrying the runtime's render (check routing).",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
