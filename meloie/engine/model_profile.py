"""Model profile: the "voice identity" half of an RVC runtime config.

A trained RVC model (``.pth``) plus its retrieval index (``.index``),
together with the inference parameters it was trained against
(``f0_method``, ``index_rate``, ``protect``, etc.), define the voice.
They are properties of that trained unit — not user-facing sound-design
knobs. (The v2 direct engine embeds via contentvec + staged predictors;
the optional ``hubert_path`` / ``rmvpe_path`` fields below are legacy and
unused by it.)

This module loads a JSON model profile that groups those paths and
parameters into a single named bundle. The two CLIs (``meloie.main`` for
realtime, ``tools.offline_infer`` for offline) consume profiles via
``--model-profile`` so a normal "play this model" command does not
mention any of the voice-identity parameters at all.

Schema (all fields optional except ``model_path``)::

    {
      "name": "A",
      "model_path":  "models/A.pth",
      "index_path":  "models/V2.index",
      "f0_method":   "rmvpe",
      "index_rate":  0.5,
      "protect":     0.33,
      "filter_radius": 3,
      "rms_mix_rate":  1.0,
      "pitch_shift":   0,
      "resample_sr":   0,
      "notes":         "Example profile. Do not tune these knobs to "
                       "'tune' the voice."
    }

Relative paths inside the profile are NOT resolved here — they remain
verbatim strings. The CLIs interpret them relative to the current
working directory (the directory you invoked ``python -m ...`` from),
which is normally the project root.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelProfile:
    """The voice-identity half of an RVC runtime configuration."""

    name: str = ""
    model_path: str = ""
    index_path: Optional[str] = None
    f0_method: str = "rmvpe"
    index_rate: float = 0.5
    protect: float = 0.33
    pitch_shift: int = 0
    # INPUT-side formant / gender shift (性别因子): timbre>1 = brighter/feminine,
    # <1 = deeper/masculine, 1.0 = off. Engine enables formant when != 1.0.
    formant_qfrency: float = 1.0
    formant_timbre: float = 1.0
    # A2 auto pitch-centering target: the model's typical median F0 (Hz). When the
    # GUI enables auto-center, the engine transposes the carrier so the user's median
    # lands here -- replacing the hand-tuned pitch_shift. None = no seed (auto-center
    # has nothing to aim at and stays inert). Hand-seeded (e.g. A ~= 200) since the
    # emb_pitch weight-probe could not derive it (see docs/f0_remap_plan.md).
    target_f0_median: Optional[float] = None
    notes: str = ""
    # --- legacy fields, accepted for back-compat but IGNORED by the v2 direct
    # engine (kept so older profile JSONs still load). hubert_path/rmvpe_path:
    # v1-era encoder paths (the v2 engine uses contentvec + staged predictors).
    # filter_radius: old F0 median window (the realtime F0 estimators set their
    # own). rms_mix_rate: MUST stay 1.0 -- <1.0 would impose the source mic's
    # loudness on the OUTPUT (change_rms), breaking the faithful contract, so the
    # direct engine never honors it. resample_sr: output SR is structural (model
    # SR -> stream SR). ---
    hubert_path: Optional[str] = None
    rmvpe_path: Optional[str] = None
    filter_radius: int = 3
    rms_mix_rate: float = 1.0
    resample_sr: int = 0


_PROFILE_FIELD_NAMES = {f.name for f in fields(ModelProfile)}


class ModelProfileError(Exception):
    """Raised on unreadable / invalid model profile JSON."""


def load_model_profile(path: str) -> ModelProfile:
    """Load a model profile JSON file into a :class:`ModelProfile`.

    Path-typed fields stay as written; the caller resolves relative
    paths to disk. Unknown JSON keys are rejected so typos are caught
    early instead of being silently ignored.
    """
    p = Path(path)
    if not p.exists():
        raise ModelProfileError(f"model profile not found: {path!r}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelProfileError(f"invalid JSON in {path!r}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelProfileError(
            f"model profile must be a JSON object; got {type(raw).__name__}"
        )

    unknown = set(raw.keys()) - _PROFILE_FIELD_NAMES
    if unknown:
        raise ModelProfileError(
            f"unknown keys in profile {path!r}: {sorted(unknown)}. "
            f"Known keys: {sorted(_PROFILE_FIELD_NAMES)}"
        )

    if not raw.get("model_path"):
        raise ModelProfileError(
            f"profile {path!r} is missing required field 'model_path'"
        )

    kwargs: Dict[str, Any] = {}
    for f in fields(ModelProfile):
        if f.name in raw:
            kwargs[f.name] = raw[f.name]
    try:
        return ModelProfile(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ModelProfileError(
            f"profile {path!r} has invalid field value: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Default .index resolution for a selected .pth
# ---------------------------------------------------------------------------
#
# A trained model's retrieval index is part of its bundle, so when no profile
# explicitly pins one we default to the .index that belongs to the selected
# .pth, by this priority (first existing match wins):
#
#   1. a same-stem .index in the .pth's OWN directory
#      (stem compared case-SENSITIVELY; Unicode / Chinese names supported)
#   2. a same-stem .index found RECURSIVELY under the models root
#   3. the first .index (sorted) in the .pth's OWN directory
#   4. the first .index (sorted) found RECURSIVELY under the models root
#
# This only picks the index FILE; whether it is actually applied is governed by
# index_rate (a model parameter), so a default-resolved index never silently
# changes the voice until retrieval is turned on. An explicit profile
# ``index_path`` is honoured by callers BEFORE this fallback runs.

_INDEX_EXT = ".index"


def _is_index(name: str) -> bool:
    return name.lower().endswith(_INDEX_EXT)


def _index_stem(name: str) -> str:
    """Filename minus a case-insensitively-matched ``.index`` extension."""
    return name[: -len(_INDEX_EXT)] if _is_index(name) else name


def _index_files_recursive(root: str) -> List[str]:
    """All ``*.index`` paths under ``root``, deterministic (dirs+files sorted)."""
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if _is_index(fn):
                out.append(os.path.join(dirpath, fn))
    return out


def models_root_for(model_path: str) -> str:
    """The directory that bounds recursive index search for ``model_path``: the
    nearest ancestor literally named ``models`` (case-insensitive), else the
    .pth's own directory. Lets the CLI recurse the whole ``models/`` tree the
    same way the GUI does, regardless of how deep the .pth is nested."""
    own = os.path.dirname(os.path.abspath(str(model_path)))
    cur = own
    while True:
        if os.path.basename(cur).lower() == "models":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return own
        cur = parent


def find_default_index(model_path: str, models_root: Optional[str] = None) -> str:
    """Resolve the default ``.index`` for a selected ``.pth`` (see the priority
    note above). Returns an absolute path that exists, or ``""`` if none is
    found. ``models_root`` bounds the recursive tiers; when omitted it is
    derived via :func:`models_root_for`."""
    model_path = str(model_path)
    own_dir = os.path.dirname(os.path.abspath(model_path))
    stem = os.path.splitext(os.path.basename(model_path))[0]   # exact case / Unicode
    root = os.path.abspath(models_root) if models_root else models_root_for(model_path)

    try:
        own = sorted(fn for fn in os.listdir(own_dir) if _is_index(fn))
    except OSError:
        own = []

    # 1. same-stem (case-sensitive) in the .pth's own directory
    for fn in own:
        if _index_stem(fn) == stem:
            return os.path.join(own_dir, fn)

    rec = _index_files_recursive(root)

    # 2. same-stem (case-sensitive) recursively under the models root
    for p in rec:
        if _index_stem(os.path.basename(p)) == stem:
            return p

    # 3. first .index in the .pth's own directory
    if own:
        return os.path.join(own_dir, own[0])

    # 4. first .index recursively under the models root
    if rec:
        return rec[0]

    return ""
