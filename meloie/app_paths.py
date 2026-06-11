"""Frozen-aware base-dir + cache-env resolution (Qt-free, stdlib only).

Shared by the engine, UI, and config layers so a PyInstaller ONEDIR build finds its
EXTERNAL data (``models/`` incl. ``models/{predictors,embedders}``, ``config/``,
``icon.svg``) next to the ``.exe`` instead of inside the bundle.

* Run from SOURCE: the base dir is the RVC project root (two dirs up from this file).
* Run FROZEN (``sys.frozen``): the base dir is the folder containing the ``.exe``
  (``dirname(sys.executable)``), where the user stages the external data tree. This
  matches ``config_assembly.models_dir()``'s long-standing frozen branch.

Code (``meloie`` itself, the QML files) is bundled INSIDE the build and resolves by
``__file__`` — only the external, often-writable data roots route through
:func:`app_base_dir`.
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.abspath(__file__)
_SOURCE_ROOT = os.path.dirname(os.path.dirname(_THIS))   # meloie/app_paths.py -> meloie -> RVC


def is_frozen() -> bool:
    """True when running inside a PyInstaller (or similar) bundle."""
    return bool(getattr(sys, "frozen", False))


def source_root() -> str:
    """The RVC source tree root (for code paths shipped with the sources)."""
    return _SOURCE_ROOT


def app_base_dir() -> str:
    """External-data root: the folder next to the ``.exe`` when frozen, else the
    RVC source root. All external data (models/ incl. predictors+embedders,
    config/, icon.svg) lives here; relative profile paths resolve against it."""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return _SOURCE_ROOT


def setup_frozen_cache_env() -> None:
    """When frozen, redirect HF/torch/numba/temp caches into ``<exe>/.cache`` and
    ``<exe>/.tmp`` so the double-clicked exe never writes to C: — ``setup_env_applio.ps1``
    only runs for the source launch. ``setdefault`` so an explicitly-set env wins.
    No-op when run from source (the ps1 already set these). Best-effort."""
    if not is_frozen():
        return
    base = app_base_dir()
    cache = os.path.join(base, ".cache")
    tmp = os.path.join(base, ".tmp")
    for d in (cache, tmp):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
    defaults = {
        "HF_HOME": os.path.join(cache, "hf"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(cache, "hf"),
        "TRANSFORMERS_CACHE": os.path.join(cache, "hf"),
        "TORCH_HOME": os.path.join(cache, "torch"),
        "XDG_CACHE_HOME": cache,
        "NUMBA_CACHE_DIR": os.path.join(cache, "numba"),
        "TMP": tmp,
        "TEMP": tmp,
        # models/embedders are loaded from LOCAL dirs; forbid any stray network probe.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
