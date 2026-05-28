"""Tests for the src.main CLI argument parser.

We avoid actually starting any audio stream — only parsing logic and
error paths are exercised here.
"""

from __future__ import annotations

from src.main import _build_parser


def test_parser_accepts_rvc_mode_with_required_flags():
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
    assert args.crossfade_ms == 20.0
    assert args.f0_method == "rmvpe"
    assert args.index_rate == 0.5
    assert args.protect == 0.33
    # Stage 2C defaults
    assert args.device == "auto"
    assert args.resample_sr is None  # _cmd_mode_rvc defaults to stream SR


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
