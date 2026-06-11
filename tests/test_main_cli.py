"""Tests for the meloie.main CLI parser + pure config helpers.

No audio stream is started — only argument parsing, config loading, and
device-override merging are exercised.
"""

from __future__ import annotations

import pytest

from meloie.main import _apply_device_overrides, _build_parser, _load_config
from meloie.audio.streams import AudioRuntimeConfig


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


def test_parser_accepts_index_rate():
    """--index-rate makes --index-path actually usable in quick-run mode
    (without it the effective rate defaulted to 0 and the path was inert)."""
    args = _build_parser().parse_args([
        "--config", CONFIG, "--model-path", "x.pth", "--index-rate", "0.3",
    ])
    assert args.index_rate == pytest.approx(0.3)
    # default: None -> fall back to the profile's value
    args = _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth"])
    assert args.index_rate is None


# ---------------------------------------------------------------------------
# Parser: engineering-knob defaults (no voice-shaping flags exist)
# ---------------------------------------------------------------------------

def test_parser_engineering_defaults():
    args = _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth"])
    assert args.device == "auto"
    assert args.rvc_queue_ms == 6000.0
    assert args.rvc_prebuffer_ms is None
    assert args.drop_stale_input is True
    assert args.input_device is None     # follow system default
    assert args.output_device is None
    assert args.allow_virtual_cable_input is False
    assert args.pitch is None            # transpose: default to the profile's value


def test_parser_direct_engine_defaults():
    """The direct (v2 Applio) engine knobs and their defaults."""
    args = _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth"])
    assert args.direct_block_ms == 250.0
    assert args.direct_context_ms == 2500.0
    assert args.direct_crossfade_ms == 50.0
    assert args.direct_embedder == "contentvec"
    assert args.direct_f0 is None        # default to the profile's f0_method
    assert args.direct_denoise is False  # input denoise opt-in


def test_parser_has_no_engine_selector():
    """v2-only build: there is a single realtime engine, so the --engine
    selector was removed entirely. There is no args.engine, and passing any
    --engine value (incl. the retired v1 'cache' path) is rejected."""
    args = _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth"])
    assert not hasattr(args, "engine")
    for value in ("direct", "cache"):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth",
                                        "--engine", value])


def test_parser_can_disable_stale_drop():
    args = _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth",
                                       "--no-drop-stale-input"])
    assert args.drop_stale_input is False


def test_parser_explicit_device_and_direct_f0():
    args = _build_parser().parse_args(["--config", CONFIG, "--model-path", "x.pth",
                                       "--device", "cuda", "--direct-f0", "fcpe"])
    assert args.device == "cuda"
    assert args.direct_f0 == "fcpe"


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


def test_parser_accepts_pitch():
    """--pitch (变调/transpose) is THE creative knob: it conditions the model's
    input F0 (a female model needs ~+12 for a male voice). Allowed."""
    args = _build_parser().parse_args(["--config", CONFIG, "--pitch", "12"])
    assert args.pitch == 12
    args = _build_parser().parse_args(["--config", CONFIG, "--pitch", "-5"])
    assert args.pitch == -5


def test_parser_has_no_output_shaping_flags():
    """Still a faithful carrier on the OUTPUT — no --mode, --crossfade-ms,
    --frame-restore-method, --f0-method, --rms-mix-rate. (--pitch and
    --index-rate are allowed: input-side conditioning — F0 transpose and the
    FAISS feature blend — not output shaping; offline_infer has the same pair.)"""
    parser = _build_parser()
    for bad in ("--mode", "--crossfade-ms", "--frame-restore-method",
                "--f0-method", "--rms-mix-rate"):
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


def test_apply_device_overrides_preserves_other_fields():
    """dataclasses.replace semantics: fields the override doesn't name (e.g.
    monitor_device_substring) must survive the merge."""
    base = AudioRuntimeConfig(monitor_device_substring="Headphones", queue_blocks=99)
    args = _build_parser().parse_args(["--config", CONFIG, "--input-device", "Realtek"])
    merged = _apply_device_overrides(base, args)
    assert merged.input_device_substring == "Realtek"
    assert merged.monitor_device_substring == "Headphones"
    assert merged.queue_blocks == 99


def test_apply_device_overrides_empty_strings_are_symmetric():
    """--input-device '' and --output-device '' both mean 'unset/keep default'
    (the old code treated them asymmetrically)."""
    base = AudioRuntimeConfig(input_device_substring="OldMic")
    args = _build_parser().parse_args(["--config", CONFIG,
                                       "--input-device", "", "--output-device", ""])
    merged = _apply_device_overrides(base, args)
    assert merged.input_device_substring is None         # "" -> unset
    assert merged.output_device_substring == "CABLE Input"
