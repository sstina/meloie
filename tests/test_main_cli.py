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
