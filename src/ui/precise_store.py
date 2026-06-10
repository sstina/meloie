"""Persisted precise-mapping store (Qt-free, pure).

Saves a built precise CDF F0 map (its log2-Hz quantile anchors + the f0 method it
was built under + the two source basenames) to a JSON under ``config/precise_maps``,
so the user can reload a mapping later WITHOUT re-picking the two .wav files and
re-running the (multi-second) build. The quantiles ARE the map — loading them is
instant and needs no estimator, so a saved map can be applied even before Start.

Pure: json + numpy only, no Qt — unit-testable in isolation. The backend lists /
saves / loads through here; the engine attaches the quantiles via set_precise_mapping.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import numpy as np

from ..app_paths import app_base_dir
# next to the .exe when frozen (writable), else the source root — saved maps are
# read AND written at runtime, so never inside the read-only bundle.
RVC_ROOT = app_base_dir()
PRECISE_DIR = os.path.join(RVC_ROOT, "config", "precise_maps")

_BAD = re.compile(r'[<>:"/\\|?*]+')        # illegal Windows filename chars (mirror mergeModels)


def _safe_name(name: str) -> str:
    return _BAD.sub("_", (name or "").strip()) or "mapping"


def save_precise_map(name, method, voice_name, target_name, src_q, tgt_q,
                     maps_dir: str = PRECISE_DIR) -> str:
    """Write a saved map to ``maps_dir/<sanitized name>.json`` (overwriting a same-name
    file) and return its path. ``src_q``/``tgt_q`` are stored as plain float lists."""
    safe = _safe_name(name)
    path = os.path.join(maps_dir, safe + ".json")
    payload = {
        "name": (name or "").strip() or safe,
        "method": str(method or "rmvpe"),
        "voice_name": str(voice_name or ""),
        "target_name": str(target_name or ""),
        "src_q": np.asarray(src_q, dtype=np.float64).reshape(-1).tolist(),
        "tgt_q": np.asarray(tgt_q, dtype=np.float64).reshape(-1).tolist(),
    }
    os.makedirs(maps_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def list_precise_maps(maps_dir: str = PRECISE_DIR) -> List[Dict[str, Any]]:
    """Metadata for every saved map (no arrays) as ``{name, file, method, voice_name,
    target_name}`` dicts for a QML ComboBox. Sorted by name; ``[]`` if the dir is
    missing; junk / array-less files are skipped (tolerant)."""
    out: List[Dict[str, Any]] = []
    try:
        names = os.listdir(maps_dir)
    except OSError:
        return out
    for fn in names:
        if not fn.lower().endswith(".json"):
            continue
        full = os.path.join(maps_dir, fn)
        try:
            d = json.loads(open(full, encoding="utf-8").read())
        except Exception:
            continue
        if not isinstance(d, dict) or "src_q" not in d or "tgt_q" not in d:
            continue
        out.append({
            "name": str(d.get("name") or os.path.splitext(fn)[0]),
            "file": full,
            "method": str(d.get("method", "rmvpe")),
            "voice_name": str(d.get("voice_name", "")),
            "target_name": str(d.get("target_name", "")),
        })
    out.sort(key=lambda m: m["name"].lower())
    return out


def load_precise_map(path: str) -> Dict[str, Any]:
    """Read a saved map; returns ``{name, method, voice_name, target_name, src_q, tgt_q}``
    with the quantiles as 1-D float64 arrays. Raises ``ValueError`` on a malformed file
    (missing / unequal / too-short arrays)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("invalid precise map file (not an object)")
    src = np.asarray(d.get("src_q", []), dtype=np.float64).reshape(-1)
    tgt = np.asarray(d.get("tgt_q", []), dtype=np.float64).reshape(-1)
    if src.size < 2 or src.size != tgt.size:
        raise ValueError("precise map has invalid src_q/tgt_q arrays")
    return {
        "name": str(d.get("name") or os.path.splitext(os.path.basename(path))[0]),
        "method": str(d.get("method", "rmvpe")),
        "voice_name": str(d.get("voice_name", "")),
        "target_name": str(d.get("target_name", "")),
        "src_q": src,
        "tgt_q": tgt,
    }
