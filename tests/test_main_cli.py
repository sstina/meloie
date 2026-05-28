"""Tests for the src.main CLI argument parser.

We avoid actually starting any audio stream — only parsing logic and
error paths are exercised here.
"""

from __future__ import annotations

from src.main import _build_parser


def test_parser_accepts_rvc_mode_with_dev_override_flags():
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-path", "models/local/example.pth",
        "--index-path", "models/local/example.index",
        "--chunk-ms", "180",
        "--crossfade-ms", "20",
    ])
    assert args.mode == "rvc"
    assert args.model_path == "models/local/example.pth"
    assert args.index_path == "models/local/example.index"
    assert args.chunk_ms == 180.0
    assert args.crossfade_ms == 20.0   # explicit value still parsed
    # Voice-identity defaults are now None so we can detect explicit
    # CLI overrides vs profile values vs hard-coded fallbacks.
    assert args.f0_method is None
    assert args.index_rate is None
    assert args.protect is None
    assert args.device == "auto"
    assert args.resample_sr is None


def test_parser_accepts_cuda_device_and_explicit_resample_sr():
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-path", "models/local/x.pth",
        "--device", "cuda",
        "--resample-sr", "48000",
    ])
    assert args.device == "cuda"
    assert args.resample_sr == 48000


def test_parser_context_default_is_200ms():
    """Stage 3: input-side left-context is ON by default at 200 ms.

    It is an *engineering* knob (continuity behaviour), not a voice
    tuning knob — the model's identity parameters are still profile-
    owned. The audit (tools/pseudo_stream) confirms this default
    preserves chunk duration exactly while reducing per-chunk cold-
    start in HuBERT / F0 / index."""
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-path", "models/local/x.pth",
    ])
    assert args.rvc_context_ms == 200.0


def test_parser_context_can_be_disabled():
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-path", "models/local/x.pth",
        "--rvc-context-ms", "0",
    ])
    assert args.rvc_context_ms == 0.0


def test_parser_crossfade_default_is_zero():
    """Model-faithful default: stitched OUTPUT-side crossfade is OFF.

    The previous default (20 ms) inserted a fixed one-crossfade-length
    timeline shift per chunk and blended two temporally-disjoint
    regions with no input-overlap. The audit (tools/pseudo_stream)
    showed this delivered no measurable faithfulness gain vs the
    offline whole-file reference. Set to >0 only as a debug override.
    """
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-path", "models/local/x.pth",
    ])
    assert args.crossfade_ms == 0.0


def test_parser_warmup_default_is_two():
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-path", "models/local/x.pth",
    ])
    assert args.warmup_rvc_count == 2


def test_parser_warmup_disable_via_zero():
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-path", "models/local/x.pth",
        "--warmup-rvc-count", "0",
    ])
    assert args.warmup_rvc_count == 0


def test_parser_rvc_queue_and_prebuffer_defaults():
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-path", "models/local/x.pth",
    ])
    assert args.rvc_queue_ms == 6000.0
    assert args.rvc_prebuffer_ms is None  # main computes 2 * chunk_ms
    assert args.drop_stale_input is True


def test_parser_can_disable_stale_drop():
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-path", "models/local/x.pth",
        "--no-drop-stale-input",
    ])
    assert args.drop_stale_input is False


def test_parser_accepts_explicit_queue_and_prebuffer():
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-path", "models/local/x.pth",
        "--rvc-queue-ms", "8000",
        "--rvc-prebuffer-ms", "1500",
    ])
    assert args.rvc_queue_ms == 8000.0
    assert args.rvc_prebuffer_ms == 1500.0


# ---------------------------------------------------------------------------
# Stage 2F: --model-profile + voice-identity resolution
# ---------------------------------------------------------------------------

def test_parser_accepts_model_profile_flag():
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "rvc",
        "--config", "config/runtime.example.json",
        "--model-profile", "config/model_profiles/kiki.example.json",
    ])
    assert args.model_profile == "config/model_profiles/kiki.example.json"
    assert args.model_path is None  # not required when a profile is given


def test_resolve_voice_identity_uses_profile_when_cli_unset():
    from argparse import Namespace
    from src.engine.model_profile import ModelProfile
    from src.main import _resolve_voice_identity

    profile = ModelProfile(
        name="kiki",
        model_path="models/kiki/kikiV1.pth",
        index_path="models/kiki/kikiV1.index",
        hubert_path="models/kiki/hubert_base.pt",
        rmvpe_path="models/kiki/rmvpe.pt",
        f0_method="rmvpe", index_rate=0.5, protect=0.33,
        filter_radius=3, rms_mix_rate=0.25, pitch_shift=0,
    )
    args = Namespace(
        model_path=None, index_path=None, hubert_path=None, rmvpe_path=None,
        f0_method=None, index_rate=None, protect=None, filter_radius=None,
        rms_mix_rate=None, pitch_shift=None,
    )
    out = _resolve_voice_identity(args, profile)
    assert out["model_path"] == "models/kiki/kikiV1.pth"
    assert out["f0_method"] == "rmvpe"
    assert out["index_rate"] == 0.5
    assert out["pitch_shift"] == 0


def test_resolve_voice_identity_cli_override_wins_and_warns(capsys):
    from argparse import Namespace
    from src.engine.model_profile import ModelProfile
    from src.main import _resolve_voice_identity

    profile = ModelProfile(model_path="m.pth", pitch_shift=0, f0_method="rmvpe")
    args = Namespace(
        model_path=None, index_path=None, hubert_path=None, rmvpe_path=None,
        f0_method=None, index_rate=None, protect=None, filter_radius=None,
        rms_mix_rate=None,
        pitch_shift=6,  # explicit override
    )
    out = _resolve_voice_identity(args, profile)
    captured = capsys.readouterr()
    assert out["pitch_shift"] == 6
    assert "developer override" in captured.err.lower()
    assert "pitch_shift" in captured.err
    # No warning for non-overridden fields.
    assert "f0_method" not in captured.err


def test_resolve_voice_identity_no_profile_uses_defaults():
    from argparse import Namespace
    from src.main import _resolve_voice_identity

    args = Namespace(
        model_path="x.pth", index_path=None, hubert_path=None, rmvpe_path=None,
        f0_method=None, index_rate=None, protect=None, filter_radius=None,
        rms_mix_rate=None, pitch_shift=None,
    )
    out = _resolve_voice_identity(args, None)
    assert out["model_path"] == "x.pth"
    assert out["f0_method"] == "rmvpe"
    assert out["index_rate"] == 0.5
    assert out["protect"] == 0.33
    assert out["pitch_shift"] == 0


def test_resolve_voice_identity_matching_override_does_not_warn(capsys):
    from argparse import Namespace
    from src.engine.model_profile import ModelProfile
    from src.main import _resolve_voice_identity

    profile = ModelProfile(model_path="m.pth", pitch_shift=3)
    args = Namespace(
        model_path=None, index_path=None, hubert_path=None, rmvpe_path=None,
        f0_method=None, index_rate=None, protect=None, filter_radius=None,
        rms_mix_rate=None,
        pitch_shift=3,  # matches profile
    )
    _resolve_voice_identity(args, profile)
    captured = capsys.readouterr()
    assert "developer override" not in captured.err.lower()


def test_parser_rejects_unknown_device():
    parser = _build_parser()
    import pytest
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--mode", "rvc",
            "--config", "x",
            "--model-path", "y",
            "--device", "rocm",
        ])


def test_parser_accepts_identity_mode():
    parser = _build_parser()
    args = parser.parse_args([
        "--mode", "identity",
        "--config", "config/runtime.example.json",
        "--duration-seconds", "30",
    ])
    assert args.mode == "identity"
    assert args.duration_seconds == 30.0
    assert args.allow_virtual_cable_input is False


def test_parser_rejects_unknown_mode(capsys):
    parser = _build_parser()
    import pytest
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "rvc_not_implemented"])
