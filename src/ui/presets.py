"""捏脸 presets: built-in starter recipes + per-model save (Qt-free, pure).

A preset is a named set of INPUT-side carrier knobs (formant / index / protect /
autotune). Applying one is contract-safe — it conditions *what speech* the model
converts, never the model's output. Built-in presets are global timbre starters
and deliberately do NOT touch pitch (pitch is model-dependent; the main slider +
the per-model save own it).

``save_model_profile`` writes the current carrier knobs back to the model's
``<stem>.json`` profile, which the GUI auto-loads on select / first load
(``config_assembly.model_default_params`` / ``build_configs_for_model``). So each
model remembers its own knobs — this also removes the pitch=0 footgun (set pitch
once, save, and the model loads with it next time). Only valid ``ModelProfile``
keys are written so ``load_model_profile`` (which rejects unknown keys) accepts it.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict

# Built-in starter presets. Each touches only the timbre knobs it names; a missing
# field is left untouched on apply. NO pitch (model-dependent; set per-model).
BUILTIN_PRESETS = [
    {"name": "原始 / Neutral", "formant_on": False, "index_rate": 0.0,
     "protect": 0.33, "autotune_on": False},
    {"name": "更亮 · 女声", "formant_on": True, "formant_timbre": 1.25, "index_rate": 0.25},
    {"name": "更沉 · 男声", "formant_on": True, "formant_timbre": 0.80, "index_rate": 0.25},
    {"name": "锁音色 · 强检索", "index_rate": 0.60, "protect": 0.33},
    {"name": "电音 · Autotune", "autotune_on": True, "autotune_strength": 1.0},
]


def _clamp_finite(v: Any, lo: float, hi: float, default: float) -> float:
    """Clamp ``v`` to ``[lo, hi]``; fall back to ``default`` for non-finite input
    (NaN/inf). Mirrors the engine's own range checks so a saved profile is always
    loadable (the strict JSON loader does not range-check these floats)."""
    f = float(v)
    if not math.isfinite(f):
        return default
    return lo if f < lo else hi if f > hi else f


def save_model_profile(model_path: str, params: Dict[str, Any], profiles_dir: str) -> str:
    """Write the current carrier knobs in ``params`` to the model's
    ``<stem>.json`` profile under ``profiles_dir`` (creating it, or updating an
    existing one while preserving its ``name`` / ``index_path`` / legacy fields).
    Returns the profile path. Only valid ``ModelProfile`` keys are written.

    ``params`` (from the GUI): ``pitch_shift``, ``index_rate``, ``protect``,
    ``formant_timbre`` + ``formant_on`` (the on/off pair is folded into
    ``formant_timbre`` where 1.0 = off, since the engine enables formant when
    timbre != 1.0)."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    prof_path = os.path.join(profiles_dir, f"{stem}.json")

    out: Dict[str, Any] = {}
    if os.path.isfile(prof_path):
        try:
            loaded = json.loads(open(prof_path, encoding="utf-8").read())
            if isinstance(loaded, dict):
                out = loaded
        except Exception:
            out = {}

    out.setdefault("name", stem)
    out["model_path"] = "models/" + os.path.basename(model_path)

    # clamp every numeric knob to the engine's accepted range and drop NaN/inf, so a
    # saved profile is always loadable (input-side conditioning only — contract-safe).
    if "pitch_shift" in params:
        out["pitch_shift"] = int(round(_clamp_finite(params["pitch_shift"], -48, 48, 0)))
    if "index_rate" in params:
        out["index_rate"] = _clamp_finite(params["index_rate"], 0.0, 1.0, 0.0)
    if "protect" in params:
        out["protect"] = _clamp_finite(params["protect"], 0.0, 0.5, 0.33)
    if "formant_on" in params or "formant_timbre" in params:
        on = bool(params.get("formant_on", True))
        out["formant_timbre"] = _clamp_finite(params.get("formant_timbre", 1.0),
                                              0.5, 2.0, 1.0) if on else 1.0

    os.makedirs(profiles_dir, exist_ok=True)
    with open(prof_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return prof_path
