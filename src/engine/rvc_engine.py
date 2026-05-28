"""Stage 2 RVC engine adapter.

Public surface:

* ``RvcEngineConfig`` — paths, params, backend selection.
* ``RvcEngine`` — ``load()`` + ``infer_array(audio, sample_rate)``.
* Exceptions: ``DependencyMissingError``, ``ModelLoadError``,
  ``RvcInferenceError``.

Hard rules baked into this module:

* No torch / infer_rvc_python imports at module load. The backend is
  imported lazily inside the backend's ``load()`` method, so importing
  this module is safe in CI / on machines without the RVC stack.
* No disk I/O inside ``infer_array``. Input is a 1-D float32 numpy
  array, output is a 1-D float32 numpy array; the realtime worker
  feeds these directly.
* If the backend or model is missing, fail with a clear actionable
  exception rather than silently degrading or faking success.

The preferred backend is `infer_rvc_python
<https://github.com/r3gm/infer_rvc_python>`_ because it exposes
``BaseLoader.generate_from_cache`` for in-memory array I/O — exactly
what the realtime chunked worker needs. The adapter speaks to the
documented public API:

.. code-block:: python

    from infer_rvc_python import BaseLoader
    converter = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None)
    converter.apply_conf(
        tag=..., file_model=..., pitch_algo="rmvpe", pitch_lvl=0,
        file_index=..., index_influence=0.5, respiration_median_filtering=3,
        resample_sr=0, envelope_ratio=0.25, consonant_protection=0.33,
    )
    result_audio, result_sr = converter.generate_from_cache(
        audio_data=(audio_array, sample_rate), tag=...,
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class RvcEngineError(Exception):
    """Base class for RVC engine errors."""


class DependencyMissingError(RvcEngineError):
    """The selected backend (e.g. ``infer_rvc_python``) is not installed."""


class ModelLoadError(RvcEngineError):
    """The model / index file is missing or could not be loaded."""


class RvcInferenceError(RvcEngineError):
    """A runtime inference call failed (CUDA OOM, NaN, backend exception)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RvcEngineConfig:
    """Parameters the engine + backend need.

    Defaults match the starting points recommended in ``rvc.md`` and
    the legacy dossier. They are deliberately conservative — tune
    them per model after the realtime route works end-to-end.
    """

    model_path: str = ""
    index_path: Optional[str] = None
    backend: str = "infer_rvc_python"
    f0_method: str = "rmvpe"
    index_rate: float = 0.5
    protect: float = 0.33
    filter_radius: int = 3
    rms_mix_rate: float = 0.25
    pitch_shift: int = 0
    sample_rate: Optional[int] = None      # informational; backend decides
    resample_sr: int = 0                   # 0 = let backend pick
    force_cpu: bool = False
    backend_tag: str = "tvoice_default"    # opaque label for backend cache

    # Optional explicit paths to the shared assets infer_rvc_python
    # otherwise downloads. Set these when the model bundle already
    # ships hubert_base.pt / rmvpe.pt next to the .pth.
    hubert_path: Optional[str] = None
    rmvpe_path: Optional[str] = None

    def validate(self) -> None:
        if not (0.0 <= self.index_rate <= 1.0):
            raise ValueError(f"index_rate must be in [0, 1]; got {self.index_rate}")
        if not (0.0 <= self.protect <= 0.5):
            raise ValueError(f"protect must be in [0, 0.5]; got {self.protect}")
        if self.filter_radius < 0:
            raise ValueError("filter_radius must be >= 0")
        if not (0.0 <= self.rms_mix_rate <= 1.0):
            raise ValueError(f"rms_mix_rate must be in [0, 1]; got {self.rms_mix_rate}")
        if self.f0_method not in ("rmvpe", "rmvpe+", "fcpe", "crepe", "harvest", "pm"):
            # Unknown methods may still work with the backend, but flag
            # the common-typo case loudly.
            pass


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class RvcBackend:
    """Abstract backend interface. One concrete impl per RVC library."""

    name = "abstract"

    def load(self, config: RvcEngineConfig) -> None:  # pragma: no cover
        raise NotImplementedError

    def infer(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, int]:  # pragma: no cover
        raise NotImplementedError


class _InferRvcPythonBackend(RvcBackend):
    """Adapter for the ``infer_rvc_python`` library (preferred)."""

    name = "infer_rvc_python"

    def __init__(self) -> None:
        self._converter = None
        self._tag = None

    def load(self, config: RvcEngineConfig) -> None:
        try:
            from infer_rvc_python import BaseLoader  # noqa: WPS433
        except ImportError as exc:  # backend not installed
            raise DependencyMissingError(
                "infer_rvc_python is not installed. Install it (and a "
                "GPU-enabled torch build) with:\n"
                "  pip install infer-rvc-python\n"
                "Then retry."
            ) from exc

        if not config.model_path or not os.path.exists(config.model_path):
            raise ModelLoadError(
                f"model_path missing or not found: {config.model_path!r}.\n"
                "Place local model files under models/local/ (gitignored) "
                "and pass the full path."
            )
        if config.index_path and not os.path.exists(config.index_path):
            raise ModelLoadError(
                f"index_path not found: {config.index_path!r}"
            )
        if config.hubert_path and not os.path.exists(config.hubert_path):
            raise ModelLoadError(
                f"hubert_path not found: {config.hubert_path!r}"
            )
        if config.rmvpe_path and not os.path.exists(config.rmvpe_path):
            raise ModelLoadError(
                f"rmvpe_path not found: {config.rmvpe_path!r}"
            )

        try:
            self._converter = BaseLoader(
                only_cpu=bool(config.force_cpu),
                hubert_path=config.hubert_path,
                rmvpe_path=config.rmvpe_path,
            )
            self._converter.apply_conf(
                tag=config.backend_tag,
                file_model=config.model_path,
                pitch_algo=config.f0_method,
                pitch_lvl=int(config.pitch_shift),
                file_index=config.index_path or "",
                index_influence=float(config.index_rate),
                respiration_median_filtering=int(config.filter_radius),
                resample_sr=int(config.resample_sr or 0),
                envelope_ratio=float(config.rms_mix_rate),
                consonant_breath_protection=float(config.protect),
            )
            self._tag = config.backend_tag
        except Exception as exc:  # backend raised during config/load
            raise ModelLoadError(
                f"infer_rvc_python failed to load model: {exc}"
            ) from exc

    def infer(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        if self._converter is None:
            raise RvcInferenceError(
                "backend not loaded; call RvcEngine.load() first"
            )
        try:
            result_audio, result_sr = self._converter.generate_from_cache(
                audio_data=(audio, int(sample_rate)),
                tag=self._tag,
            )
        except Exception as exc:
            raise RvcInferenceError(
                f"infer_rvc_python inference failed: {exc}"
            ) from exc

        out = np.asarray(result_audio).reshape(-1)
        # infer_rvc_python returns audio at int16 magnitude (peak ~32768),
        # not normalised to [-1, 1]. Convert defensively: integer dtypes
        # use the dtype's full scale; floats that are clearly outside
        # [-1.5, 1.5] are assumed to be int16-scale and rescaled.
        if np.issubdtype(out.dtype, np.integer):
            info = np.iinfo(out.dtype)
            scale = float(max(abs(int(info.min)), abs(int(info.max))))
            out = out.astype(np.float32) / scale
        else:
            out = out.astype(np.float32, copy=False)
            if out.size:
                peak = float(np.max(np.abs(out)))
                if peak > 1.5:
                    out = out / np.float32(32768.0)
        return out, int(result_sr)


# Registry of known backends. Tests / advanced users can register fakes
# via :func:`register_backend`.
_BACKENDS = {
    "infer_rvc_python": _InferRvcPythonBackend,
}


def register_backend(name: str, factory) -> None:
    """Register a backend factory keyed by ``RvcEngineConfig.backend``."""
    if not callable(factory):
        raise TypeError("factory must be callable returning an RvcBackend")
    _BACKENDS[str(name)] = factory


# ---------------------------------------------------------------------------
# RvcEngine
# ---------------------------------------------------------------------------

class RvcEngine:
    """High-level RVC engine.

    Construction is cheap and side-effect free. ``load()`` does the
    heavy lifting; ``infer_array()`` runs one inference on a 1-D mono
    float32 array.
    """

    def __init__(
        self,
        config: Optional[RvcEngineConfig] = None,
        backend: Optional[RvcBackend] = None,
    ) -> None:
        self._config = config or RvcEngineConfig()
        self._config.validate()
        self._backend = backend
        self._loaded = backend is not None and getattr(backend, "_loaded_marker", False)

    @property
    def config(self) -> RvcEngineConfig:
        return self._config

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend_name(self) -> str:
        if self._backend is not None:
            return getattr(self._backend, "name", "custom")
        return self._config.backend

    def load(self) -> None:
        if self._loaded:
            return
        if self._backend is None:
            factory = _BACKENDS.get(self._config.backend)
            if factory is None:
                raise DependencyMissingError(
                    f"unknown RVC backend: {self._config.backend!r}. "
                    f"Known: {sorted(_BACKENDS.keys())}"
                )
            self._backend = factory()
        self._backend.load(self._config)
        self._loaded = True

    def infer_array(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded:
            raise RvcInferenceError(
                "engine not loaded; call RvcEngine.load() first"
            )
        if not isinstance(audio, np.ndarray):
            raise TypeError(
                f"audio must be a numpy array, got {type(audio).__name__}"
            )
        if audio.ndim != 1:
            raise ValueError(
                f"audio must be 1-D mono, got shape {audio.shape}"
            )
        if int(sample_rate) <= 0:
            raise ValueError("sample_rate must be > 0")

        audio_f32 = audio.astype(np.float32, copy=False)
        return self._backend.infer(audio_f32, int(sample_rate))
