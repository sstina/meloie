"""Qt-free config assembly + enumeration for the GUI.

Turns a selected model (``.pth`` discovered in ``models/``) + device choices into
the dataclasses the engine and stream need (``StreamingEngineConfig`` /
``AudioRuntimeConfig``), and lists the discovered models + audio devices for the
QML dropdowns. A model's tuned recipe (pitch/index/...) comes from a matching
``<stem>.json`` profile if one exists. Intentionally has NO Qt import so it stays
cheap and unit-testable; the Qt ``Backend`` calls into it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from ..audio.devices import is_probable_cable_input, is_probable_cable_output
from ..audio.streams import AudioRuntimeConfig, list_audio_devices
from ..engine.model_profile import (
    ModelProfileError,
    find_default_index,
    load_model_profile,
)
from ..engine.streaming_engine import StreamingEngineConfig

from ..app_paths import app_base_dir
# External data root: next to the .exe when frozen, else the source root (matches
# models_dir()'s frozen branch). config/ is read AND written at runtime, so it must
# live next to the exe (writable), never inside the read-only bundle.
RVC_ROOT = app_base_dir()
PROFILES_DIR = os.path.join(RVC_ROOT, "config", "model_profiles")

DEFAULT_F0 = "fcpe"            # launcher parity (run_A_direct bakes --direct-f0 fcpe)
STREAM_SR = 48000


# ---------------------------------------------------------------- game modes
# Game mode trades precision/latency for low (ideally zero) dGPU usage while gaming.
# Each mode overrides ONLY load-time engine fields (device / block_ms / context_ms /
# cpu_threads) -- those need a reload anyway. The "sacrifice precision" LIVE knobs
# (index_rate -> 0, silence gate) are deliberately NOT here: the Backend toggles
# those through its existing setters so _desired / engine / UI stay consistent and
# returning to "off" can restore them. Feasibility measured (model A, fcpe, index 0):
# CPU inference barely scales past ~8 threads and runs ~270 ms/block at block_ms=500
# (< 500 ms budget, ~50% headroom) -> zero-dGPU realtime, latency ~1.3 s.
GAME_MODES: Dict[str, Dict[str, Any]] = {
    # off: normal default path (full dGPU, best quality). No overrides.
    "off": {},
    # dgpu_light: stay on the dGPU but halve the inference rate (block 500 -> 2 Hz
    # bursts) and shrink context -> far less GPU contention with the game, still low
    # latency. The best-quality of the two active modes.
    "dgpu_light": {"device": "cuda", "block_ms": 500.0, "context_ms": 1500.0, "cpu_threads": 0},
    # cpu_zero: zero dGPU. Inference runs entirely on the CPU pinned to ~8 cores; the
    # whole dGPU and the rest of the CPU stay free for the game. Latency ~1.3 s.
    "cpu_zero": {"device": "cpu", "block_ms": 500.0, "context_ms": 1000.0, "cpu_threads": 8},
}


def apply_game_mode(scfg: StreamingEngineConfig, mode: Optional[str]) -> StreamingEngineConfig:
    """Apply the game-mode load-time overrides for ``mode`` to ``scfg``.

    Returns a NEW config with the overrides (the input is left untouched) so the
    mapping stays a pure transform; ``None`` / ``"off"`` / unknown returns ``scfg``
    unchanged. Only device / block_ms / context_ms / cpu_threads are touched —
    every other field (recipe, f0, formant, ...) is preserved."""
    overrides = GAME_MODES.get(mode or "off") or {}
    if not overrides:
        return scfg
    return replace(scfg, **overrides)


# ---------------------------------------------------------------- devices
def list_device_dicts() -> List[Dict[str, Any]]:
    """Audio devices as plain dicts for QML ComboBoxes. Returns [] (not raise)
    if sounddevice/enumeration fails — the Backend surfaces that separately."""
    try:
        infos = list_audio_devices()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for d in infos:
        out.append({
            "index": int(d.index),
            "name": d.name,
            "maxIn": int(d.max_input_channels),
            "maxOut": int(d.max_output_channels),
            "isCableInput": is_probable_cable_input(d.name),
            "isCableOutput": is_probable_cable_output(d.name),
        })
    return out


# -------------------------------------------------- model discovery (models/ dir)
def models_dir() -> str:
    """The folder we discover ``.pth`` models in: a ``models`` subdir (matched
    case-insensitively) next to ``run_gui.bat`` / the frozen ``.exe``. Defaults to
    ``RVC_ROOT/models`` when no such folder is found."""
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else RVC_ROOT
    try:
        for entry in os.scandir(base):
            if entry.is_dir() and entry.name.lower() == "models":
                return entry.path
    except OSError:
        pass
    return os.path.join(base, "models")


def list_model_files() -> List[Dict[str, Any]]:
    """Every ``.pth`` model found RECURSIVELY under :func:`models_dir` (any depth of
    subfolder), as ``{name, path}`` dicts for a QML ComboBox. ``name`` is the path
    relative to ``models_dir`` without the ``.pth`` extension, forward-slashed — the
    bare stem for a top-level model, or ``subdir/stem`` when nested, so two models
    with the same filename in different folders stay distinguishable. Sorted by
    name; ``[]`` if the folder is missing (``os.walk`` of a missing dir yields
    nothing). The path IS the identity (the engine loads ``path``); ``name`` is
    display-only and the ``<stem>.json`` profile lookup keys off the file's
    basename, so nesting never changes which recipe a model loads."""
    d = models_dir()
    out: List[Dict[str, Any]] = []
    for root, _dirs, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(".pth"):
                full = os.path.join(root, fn)
                name = os.path.splitext(os.path.relpath(full, d))[0].replace(os.sep, "/")
                out.append({"name": name, "path": full})
    out.sort(key=lambda m: m["name"].lower())
    return out


def _profile_for_model(model_path: str):
    """The matching ``<stem>.json`` profile for a discovered model, or ``None``.
    A discovered model keeps its tuned recipe (e.g. A.pth <- A.json: pitch 12 +
    V2.index); models without a profile load with neutral defaults."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    prof = os.path.join(PROFILES_DIR, f"{stem}.json")
    if os.path.isfile(prof):
        try:
            return load_model_profile(prof)
        except ModelProfileError:
            return None
    return None


def model_default_params(model_path: str) -> Dict[str, Any]:
    """Slider-init values for a discovered model: its ``<stem>.json`` recipe if
    present, else neutral defaults (no index, pitch 0)."""
    p = _profile_for_model(model_path)
    if p is None:
        # No profile: the index defaults to the .pth's own .index (same-stem,
        # else first; recursive under models/), so the GUI index slider is live.
        return {
            "pitch_shift": 0, "protect": 0.33, "index_rate": 0.0,
            "formant_timbre": 1.0, "formant_on": False,
            "has_index": bool(find_default_index(model_path, models_dir())),
            "target_f0_median": 0.0,
        }
    # formant_on mirrors build_configs_for_model's derivation so the GUI checkbox
    # reflects a saved gender shift (engine enables formant when timbre/qfrency != 1.0).
    formant_on = (float(p.formant_timbre) != 1.0) or (float(p.formant_qfrency) != 1.0)
    return {
        "pitch_shift": int(p.pitch_shift),
        "protect": float(p.protect),
        "index_rate": float(p.index_rate),
        "formant_timbre": float(p.formant_timbre),
        "formant_on": formant_on,
        # explicit profile index, else the .pth's own default index
        "has_index": bool(p.index_path or find_default_index(model_path, models_dir())),
        # >0 means auto-center has a target to aim at -> the GUI can offer the toggle.
        "target_f0_median": float(p.target_f0_median) if p.target_f0_median else 0.0,
    }


def build_configs_for_model(
    model_path: str,
    input_substr: Optional[str],
    output_substr: Optional[str],
    f0: str = DEFAULT_F0,
    monitor_substr: Optional[str] = None,
    game_mode: str = "off",
) -> Tuple[StreamingEngineConfig, AudioRuntimeConfig]:
    """Build the engine + audio configs from a discovered ``.pth`` model. The
    model's ``<stem>.json`` profile (if any) supplies the recipe (pitch / index /
    protect / formant); otherwise neutral defaults. ``model_path`` is ALWAYS the
    discovered ``.pth`` (the profile only contributes the recipe). The index is
    ALWAYS loaded when present so the GUI's index slider stays live
    (``set_index_rate`` would otherwise raise with no index loaded).

    ``game_mode`` (off / dgpu_light / cpu_zero) layers low/zero-dGPU load-time
    overrides on top of the recipe via :func:`apply_game_mode` (see GAME_MODES)."""
    p = _profile_for_model(model_path)
    if p is not None:
        index_path = p.index_path or ""
        index_rate = float(p.index_rate)
        protect = float(p.protect)
        pitch_shift = int(p.pitch_shift)
        formant_timbre = float(p.formant_timbre)
        formant_qfrency = float(p.formant_qfrency)
        target_f0_median = float(p.target_f0_median) if p.target_f0_median else 0.0
    else:
        index_path, index_rate, protect, pitch_shift = "", 0.0, 0.33, 0
        formant_timbre, formant_qfrency = 1.0, 1.0
        target_f0_median = 0.0
    formant_on = (formant_timbre != 1.0) or (formant_qfrency != 1.0)

    # No explicit profile index -> default to the .pth's own .index (same-stem
    # first, else the first .index; recursive under models/). The index file is
    # loaded so the slider stays live; index_rate still governs whether it is
    # applied, so this never silently changes the voice.
    if not index_path:
        index_path = find_default_index(model_path, models_dir())

    scfg = StreamingEngineConfig(
        model_path=model_path,
        index_path=index_path,
        f0_method=f0,
        embedder="contentvec",
        pitch_shift=pitch_shift,
        index_rate=index_rate,
        protect=protect,
        sid=0,
        stream_sr=STREAM_SR,
        block_ms=250.0,
        context_ms=2500.0,
        crossfade_ms=50.0,
        formant_shift=formant_on,
        formant_qfrency=formant_qfrency,
        formant_timbre=formant_timbre,
        auto_center_target_hz=target_f0_median,   # seed; auto_center stays OFF (GUI opt-in)
        device="cuda",
    )
    scfg = apply_game_mode(scfg, game_mode)       # layer low/zero-dGPU overrides (off = no-op)
    acfg = build_audio_config(input_substr, output_substr, monitor_substr)
    return scfg, acfg


def build_audio_config(
    input_substr: Optional[str] = None,
    output_substr: Optional[str] = None,
    monitor_substr: Optional[str] = None,
) -> AudioRuntimeConfig:
    """The single ``AudioRuntimeConfig`` builder, shared by the load path
    (:func:`build_configs_for_model`) and the GUI's fast-path Start
    (``Backend._build_audio``) so block size / queue depth / defaults live in one
    place. Output defaults to ``CABLE Input``; ``monitor_substr`` None -> system
    default output. Validated before return."""
    acfg = AudioRuntimeConfig(
        sample_rate=STREAM_SR,
        block_size=480,
        channels=1,
        input_device_substring=(input_substr or None),
        output_device_substring=(output_substr or "CABLE Input"),
        queue_blocks=64,
        monitor_device_substring=(monitor_substr or None),
    )
    acfg.validate()
    return acfg
