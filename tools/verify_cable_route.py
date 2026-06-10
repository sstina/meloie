"""Stage 1 cable-route verification tool.

Renders a 440 Hz sine tone to the selected output device (default
``CABLE Input``) while simultaneously capturing from the selected
input device (default ``CABLE Output``). Confirms that the cable
route is alive: the captured buffer must be non-silent.

This is the through-cable transport check from the legacy validation
ladder. It does NOT exercise the realtime identity worker — that is
what ``python -m meloie.main --mode identity`` is for. It exercises the
VB-CABLE pipe itself so that, when the identity worker is also alive,
we know both halves are good.

Run with::

    python -m tools.verify_cable_route \\
        --output-device-substring "CABLE Input" \\
        --input-device-substring "CABLE Output" \\
        --duration-seconds 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np

from meloie.audio.measurement import (
    DEFAULT_NON_SILENCE_THRESHOLD_DBFS,
    generate_sine_tone,
    summarize_capture,
)
from meloie.audio.streams import resolve_input_device, resolve_output_device


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tvoice-verify-cable-route",
        description="Verify Python -> CABLE Input -> CABLE Output is non-silent.",
    )
    p.add_argument("--output-device-substring", default="CABLE Input")
    p.add_argument("--input-device-substring", default="CABLE Output")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument(
        "--duration-seconds", type=float, default=2.0,
        help="How long to render and capture the tone.",
    )
    p.add_argument("--frequency-hz", type=float, default=440.0)
    p.add_argument("--amplitude", type=float, default=0.25)
    p.add_argument(
        "--threshold-dbfs", type=float,
        default=DEFAULT_NON_SILENCE_THRESHOLD_DBFS,
        help="Capture peak must exceed this for the route to be 'alive'.",
    )
    p.add_argument(
        "--report-dir", default=None,
        help="Directory for saved report. Use a path under "
             "eval_corpus/reports/ so the .gitignore covers it.",
    )
    p.add_argument(
        "--save", action="store_true",
        help="Save captured audio (.wav) + sidecar JSON to --report-dir. "
             "Default behaviour does NOT write any artifact.",
    )
    return p


def _write_wav(path: Path, audio_f32: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm16.tobytes())


def main(argv: Optional[List[str]] = None) -> int:
    for _s in (sys.stdout, sys.stderr):  # tolerate non-GBK device names
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    args = _build_parser().parse_args(argv)

    try:
        import sounddevice as sd  # noqa: WPS433  - deliberately lazy
    except ImportError as exc:
        print(
            f"error: sounddevice is not installed ({exc}). "
            "Install with `pip install sounddevice` and retry.",
            file=sys.stderr,
        )
        return 2

    raw_devices = list(sd.query_devices())

    try:
        in_info = resolve_input_device(
            raw_devices, args.input_device_substring, allow_virtual_cable=True
        )
        out_info = resolve_output_device(
            raw_devices, args.output_device_substring
        )
    except LookupError as exc:
        print(f"error: device resolution failed: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # FeedbackLoopRisk on output side, etc.
        print(f"error: device resolution refused: {exc}", file=sys.stderr)
        return 3

    sr = int(args.sample_rate)
    tone = generate_sine_tone(
        sample_rate=sr,
        duration_seconds=args.duration_seconds,
        frequency_hz=args.frequency_hz,
        amplitude=args.amplitude,
    )
    play = tone.reshape(-1, 1).astype(np.float32)

    print(f"output device  [{out_info.index:>3}]: {out_info.name}")
    print(f"input device   [{in_info.index:>3}]: {in_info.name}")
    print(
        f"sample_rate={sr} frequency_hz={args.frequency_hz:.1f} "
        f"amplitude={args.amplitude:.3f} duration={args.duration_seconds:.2f}s"
    )
    print("rendering + capturing ...", flush=True)

    captured_2d = sd.playrec(
        play,
        samplerate=sr,
        channels=1,
        device=(in_info.index, out_info.index),
        dtype="float32",
    )
    sd.wait()
    captured = np.asarray(captured_2d).reshape(-1)

    summary = summarize_capture(captured, threshold_dbfs=args.threshold_dbfs)

    print("--- result ---")
    print(f"  captured_n_samples  = {summary['n_samples']}")
    print(f"  captured_peak_dbfs  = {summary['peak_dbfs']:.2f}")
    print(f"  captured_rms_dbfs   = {summary['rms_dbfs']:.2f}")
    print(f"  threshold_dbfs      = {summary['threshold_dbfs']:.2f}")
    print(f"  non_silent          = {summary['non_silent']}")

    if args.save:
        if not args.report_dir:
            print(
                "error: --save requires --report-dir (use a path under "
                "eval_corpus/reports/ so the gitignore covers it)",
                file=sys.stderr,
            )
            return 4
        out_dir = Path(args.report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        wav_path = out_dir / f"verify_cable_{ts}.wav"
        json_path = out_dir / f"verify_cable_{ts}.json"
        _write_wav(wav_path, captured, sr)
        payload = {
            "timestamp": ts,
            "sample_rate": sr,
            "captured": summary,
            "input_device": {"index": in_info.index, "name": in_info.name},
            "output_device": {"index": out_info.index, "name": out_info.name},
            "config": {
                "frequency_hz": args.frequency_hz,
                "amplitude": args.amplitude,
                "duration_seconds": args.duration_seconds,
                "threshold_dbfs": args.threshold_dbfs,
            },
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved: {wav_path}")
        print(f"saved: {json_path}")

    return 0 if summary["non_silent"] else 6


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
