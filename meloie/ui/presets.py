"""捏脸 per-model save: persist the current carrier knobs as a model's default (Qt-free, pure).

``save_model_profile`` writes the current INPUT-side carrier knobs (formant /
index / protect / pitch) back to the model's ``<stem>.json`` profile, which the
GUI auto-loads on select / first load (``config_assembly.model_default_params`` /
``build_configs_for_model``). So each model remembers its own knobs — this also
removes the pitch=0 footgun (set pitch once, save, and the model loads with it
next time). Saving is contract-safe — it conditions *what speech* the model
converts, never the model's output. Only valid ``ModelProfile`` keys are written
so ``load_model_profile`` (which rejects unknown keys) accepts it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import fields as _dc_fields
from typing import Any, Dict

from ..app_paths import app_base_dir
from ..engine.model_profile import ModelProfile

# Only these keys may appear in a saved profile — the strict loader rejects
# anything else, so an existing file's stray/legacy keys are dropped on save.
_VALID_PROFILE_KEYS = {f.name for f in _dc_fields(ModelProfile)}


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
    existing one while preserving its other valid fields — ``name`` /
    ``index_path`` / ``target_f0_median`` / ``notes``...). Returns the profile
    path. Only valid ``ModelProfile`` keys are written.

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
                # keep only keys the strict loader accepts (drops stray/legacy)
                out = {k: v for k, v in loaded.items() if k in _VALID_PROFILE_KEYS}
        except Exception:
            out = {}

    out.setdefault("name", stem)
    # the model's REAL location relative to the app root (recursive discovery can
    # find models in subfolders, so a hardcoded "models/<basename>" would lie);
    # forward slashes for portability. Outside the root (other drive/tree) -> keep
    # the path as given. Profile filenames stay basename-keyed regardless.
    try:
        rel = os.path.relpath(os.path.abspath(model_path), app_base_dir())
    except ValueError:                          # different drive on Windows
        rel = model_path
    if rel.startswith(".."):                    # outside the app root
        rel = model_path
    out["model_path"] = rel.replace(os.sep, "/")

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
