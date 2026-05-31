"""Model profile: the "voice identity" half of an RVC runtime config.

A trained RVC model (``.pth``) plus its retrieval index (``.index``),
together with the inference parameters it was trained against
(``f0_method``, ``index_rate``, ``protect``, etc.), define the voice.
They are properties of that trained unit — not user-facing sound-design
knobs. (The v2 direct engine embeds via contentvec + staged predictors;
the optional ``hubert_path`` / ``rmvpe_path`` fields below are legacy and
unused by it.)

This module loads a JSON model profile that groups those paths and
parameters into a single named bundle. The two CLIs (``src.main`` for
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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional


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
