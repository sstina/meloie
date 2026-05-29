"""Tests for the src.main CLI parser + pure config helpers.

No audio stream is started — only argument parsing, config loading, and
device-override merging are exercised.
"""

from __future__ import annotations

import pytest

from src.main import _apply_device_overrides, _build_parser, _load_config
from src.audio.streams import AudioRuntimeConfig


CONFIG = "config/runtime.example.json"


# ---------------------------------------------------------------------------
# Parser: voice source
# ---------------------------------------------------------------------------

def test_parser_accepts_model_profile():
    args = _build_parser().parse_args(["--config", CONFIG,
                                       "--model-profile", "p.json"])
    assert args.model_profile == "p.json"
    assert args.model_path is None


def test_parser_accepts_model_and_index_path():
    args = _build_parser().parse_args([
        "--config", CONFIG,
        "--model-path", "models/local/x.pth",
        "--index-path", "models/local/x.index",
    ])
    assert args.model_path == "models/local/x.pth"
    assert args.index_path == "models/local/x.index"


# ---------------------------------------------------------------------------
# Parser: engineering-knob defaults (no voice-shaping flags exist)
# ---------------------------------------------------------------------------

def test_parser_engineering_defaults():
    args = _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth"])
    assert args.device == "auto"
    assert args.chunk_ms == 1000.0
    assert args.rvc_context_ms == 500.0
    assert args.tail_pad_ms == 30.0
    assert args.sola_search_ms == 10.0
    assert args.rvc_queue_ms == 6000.0
    assert args.rvc_prebuffer_ms is None
    assert args.warmup_rvc_count == 2
    assert args.drop_stale_input is True
    assert args.resample_sr is None
    assert args.input_device is None     # follow system default
    assert args.output_device is None
    assert args.allow_virtual_cable_input is False


def test_parser_can_disable_stale_drop():
    args = _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth",
                                       "--no-drop-stale-input"])
    assert args.drop_stale_input is False


def test_parser_explicit_device_and_resample_sr():
    args = _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth",
                                       "--device", "cuda", "--resample-sr", "48000"])
    assert args.device == "cuda"
    assert args.resample_sr == 48000


def test_parser_rejects_unknown_device():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth",
                                    "--device", "rocm"])


def test_parser_input_device_override_and_legacy_alias():
    a = _build_parser().parse_args(["--config", CONFIG, "--input-device", "Realtek"])
    assert a.input_device == "Realtek"
    # legacy long form still works (muscle memory from the old README)
    b = _build_parser().parse_args(["--config", CONFIG,
                                    "--input-device-substring", "WO Mic"])
    assert b.input_device == "WO Mic"


def test_parser_has_no_voice_shaping_flags():
    """The runtime is a faithful carrier — there must be no --mode,
    --crossfade-ms, --frame-restore-method, --pitch-shift, etc."""
    parser = _build_parser()
    for bad in ("--mode", "--crossfade-ms", "--frame-restore-method",
                "--pitch-shift", "--index-rate", "--f0-method"):
        with pytest.raises(SystemExit):
            parser.parse_args([bad, "x"])


# ---------------------------------------------------------------------------
# Config loading + device overrides
# ---------------------------------------------------------------------------

def test_load_config_input_defaults_to_system_default():
    cfg = _load_config(CONFIG)
    assert cfg.input_device_substring is None     # null in JSON -> system default
    assert cfg.output_device_substring == "CABLE Input"
    assert cfg.sample_rate == 48000


def test_apply_device_overrides_input_pins_a_mic():
    base = AudioRuntimeConfig()  # input None = system default
    args = _build_parser().parse_args(["--config", CONFIG, "--input-device", "Realtek"])
    merged = _apply_device_overrides(base, args)
    assert merged.input_device_substring == "Realtek"
    assert merged.output_device_substring == "CABLE Input"


def test_apply_device_overrides_noop_returns_same_object():
    base = AudioRuntimeConfig()
    args = _build_parser().parse_args(["--config", CONFIG])
    merged = _apply_device_overrides(base, args)
    assert merged is base
