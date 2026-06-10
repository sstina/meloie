"""Offline smoke for the Phase 0 live-control API (NOT a pytest test).

Proves, on the real model A, that the engine live setters propagate during the
process_block loop and that formant/denoise/silence/index toggles do not crash —
all WITHOUT audio devices. Run in .venv-applio:

    . .\setup_env_applio.ps1
    python tests\smoke_session_applio.py

Named smoke_*.py with no test_* functions so the default pytest run never
collects it (keeps the pure suite count stable). Skips cleanly (exit 0) if
models/A.pth is absent.
"""

import os
import sys

RVC = r"D:\Users\Palovil\Desktop\Tvoice\RVC"
sys.path.insert(0, RVC)
os.chdir(RVC)

import numpy as np  # noqa: E402


def _sine(n, sr, f=150.0, a=0.2):
    t = np.arange(n, dtype=np.float32) / float(sr)
    return (a * np.sin(2.0 * np.pi * f * t)).astype(np.float32)


def run_smoke() -> int:
    from meloie.engine.streaming_engine import StreamingEngineConfig, StreamingRvcEngine
    from meloie.control import RealtimeSession, SessionState

    if not os.path.exists(os.path.join(RVC, "models", "A.pth")):
        print("SKIP: models/A.pth not present — smoke needs the real model.")
        return 0

    SR = 48000
    has_index = os.path.exists(os.path.join(RVC, "models", "V2.index"))
    cfg = StreamingEngineConfig(
        model_path="models/A.pth",
        index_path="models/V2.index" if has_index else "",
        f0_method="fcpe", pitch_shift=12, index_rate=0.0, protect=0.33,
        stream_sr=SR, block_ms=250, context_ms=2500, crossfade_ms=50, device="cuda",
    )
    e = StreamingRvcEngine(cfg)
    e.load()
    bf = e.block_frame
    print(f"loaded: device={e.resolved_device} block_frame={bf} warmup={e._warmup}")

    # warm up
    for _ in range(int(e._warmup) + 2):
        out = e.process_block(_sine(bf, SR), SR)
        assert out.shape[0] == bf and out.dtype == np.float32

    # live pitch change propagates
    e.set_pitch_shift(5)
    assert e.pitch_shift == 5
    out = e.process_block(_sine(bf, SR), SR)
    assert out.shape[0] == bf and np.all(np.isfinite(out))
    print("ok: set_pitch_shift live, output finite")

    # formant toggle (builds stftpitchshift lazily) — no crash
    e.set_formant(True, timbre=0.25)
    assert e._formant_on and e._formant is not None
    assert e.process_block(_sine(bf, SR), SR).shape[0] == bf
    e.set_formant(False)
    print("ok: formant on/off live")

    # denoise toggle (builds TorchGate lazily) — no crash
    e.set_denoise(True, strength=0.5)
    assert e._denoise_on and e._denoiser is not None
    assert e.process_block(_sine(bf, SR), SR).shape[0] == bf
    e.set_denoise(False)
    print("ok: denoise on/off live")

    # index live (only if V2.index loaded)
    if e._index_loaded():
        e.set_index_rate(0.4)
        assert e.index_rate == 0.4
        assert e.process_block(_sine(bf, SR), SR).shape[0] == bf
        e.set_index_rate(0.0)
        print("ok: index_rate live")
    else:
        print("note: no index loaded; index_rate live path skipped")

    # sid (声线) live: A's emb_g has many rows (inherited pretrained table)
    print(f"num_speakers={e.num_speakers}")
    if e.num_speakers > 1:
        e.set_sid(1)
        assert e._pipeline.sid == 1 and int(e._pipeline.torch_sid.item()) == 1
        assert e.process_block(_sine(bf, SR), SR).shape[0] == bf
        e.set_sid(0)
        print("ok: set_sid live")
    else:
        print("note: single-speaker model; set_sid live path skipped")

    # silence gate: after hangover, a quiet block is muted to zeros
    e.set_silence_gate(-50.0, 250.0)
    e.process_block(_sine(bf, SR), SR)            # loud -> arms hangover
    quiet = np.full(bf, 1e-5, dtype=np.float32)
    last = None
    for _ in range(4):
        last = e.process_block(quiet, SR)
    assert np.allclose(last, 0.0), "silence gate did not mute a quiet block"
    e.set_silence_gate(None)
    print("ok: silence gate mutes after hangover")

    # RealtimeSession delegation (reuse the loaded engine; no start -> no devices)
    s = RealtimeSession(engine_factory=lambda c: e)
    s.load(cfg)
    assert s.state is SessionState.LOADED
    s.set_pitch_shift(7)
    assert s.engine.pitch_shift == 7
    print("ok: RealtimeSession load + set delegation")

    print("PASS: Phase 0 live-control smoke")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke())
