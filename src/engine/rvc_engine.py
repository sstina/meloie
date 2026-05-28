"""Stage 2 RVC engine placeholder.

This file deliberately does NOT import torch, infer_rvc_python, or
rvc-python. Loading those is part of Stage 2 and is gated behind
explicit Stage 2 work — this skeleton refuses to do it.

The class exists so callers and tests can refer to the future API
surface; every operation that would require the real engine raises
``NotImplementedError`` with the exact message demanded by the project
plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


_NOT_IMPLEMENTED_MSG = (
    "RVC inference is Stage 2 and is not implemented in this skeleton."
)


@dataclass(frozen=True)
class RvcEngineConfig:
    """Paths and parameters the future engine will use.

    Storing them does not load anything; the engine constructor below
    accepts a config and remembers it for the day Stage 2 lands.
    """

    model_pth_path: Optional[str] = None
    index_path: Optional[str] = None
    sample_rate: int = 48000
    f0_method: str = "rmvpe"
    index_rate: float = 0.5
    protect: float = 0.33
    chunk_ms: int = 160


class RvcEngine:
    """Placeholder for the Stage 2 RVC engine.

    The constructor records config only. No model is loaded, no torch
    import happens, no file is read. ``infer_array`` raises with the
    Stage 2 sentinel message so any code path that reaches it during
    Stage 1 fails loudly instead of silently doing nothing.
    """

    def __init__(self, config: Optional[RvcEngineConfig] = None) -> None:
        self._config = config or RvcEngineConfig()
        self._loaded = False

    @property
    def config(self) -> RvcEngineConfig:
        return self._config

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def infer_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Will run RVC inference on ``audio`` at ``sample_rate`` once Stage 2 lands."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
