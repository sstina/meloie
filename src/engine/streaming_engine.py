"""Path-A faithful realtime engine over the vendored Applio inference stack.

``StreamingRvcEngine`` owns the streaming STATE a stateless per-chunk engine
never had: a persistent 16 kHz ``convert_buffer`` plus persistent F0 caches and
an output SOLA buffer. Each :meth:`process_block` call shifts one fresh block into those
buffers (``circular_write``), so:

* **F0 is continuous** — recomputed only on the newest trailing window and shifted
  into the cache, not re-estimated from scratch per block (the main 听感 lever);
* the generator decodes **only the new region** (``voice_conversion`` uses
  ``net_g.infer(rate=)`` internally);
* every block is conditioned on **real past audio** (``context_ms``), not a mirror.

Faithful-carrier contract — this engine carries the model's samples and does ONLY:
  * resample in/out (stream SR <-> model SR), a structural adaptation;
  * SOLA seam alignment + a short sin² crossfade at the block seam. The crossfade
    is a *seam-only* blend of two renders of the same audio — explicitly sanctioned
    by the user as the one allowed relaxation; it never alters pitch/timbre.
OUTPUT-side colouring is stripped: the output×input-RMS scaling (``change_rms`` /
volume_envelope), the Pedalboard FX rack, and output-side noise reduction are all
disabled / never invoked. INPUT-side conditioning (pitch transpose, optional denoise,
formant/gender, autotune, proposed-pitch, silence gate) IS allowed — it changes *what
speech* is converted, never the model's voice — and is live-adjustable via ``set_*``.

Runs only in ``.venv-applio`` (torch 2.7 + transformers, the vendored Applio stack).
The vendored package + torch are imported lazily inside :meth:`load` so importing
this module is cheap and safe elsewhere.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Absolute roots derived from this file: .../RVC/src/engine/streaming_engine.py
_THIS = os.path.abspath(__file__)
RVC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))   # .../RVC
VENDOR_DIR = os.path.join(RVC_ROOT, "src", "vendor", "applio")        # holds the `rvc` pkg

STREAM_SR_DEFAULT = 48000
SR16 = 16000


class StreamingEngineError(Exception):
    """StreamingRvcEngine load/inference failure."""


@dataclass
class StreamingEngineConfig:
    """Everything the streaming engine needs. Model params mirror the model profile.

    block_ms / context_ms / crossfade_ms are the realtime knobs:
      * block_ms     -- output granularity (Applio default 250). Lower = lower
                        latency, more seams. Chosen so block_frame is /3 (48k->16k).
      * context_ms   -- REAL past audio fed to the encoders each block. Free
                        latency-wise (it is past); only costs compute. Bigger =
                        steadier timbre + F0 (Applio default 2500).
      * crossfade_ms -- sin² seam overlap (Applio default 50). Sanctioned blend.
    """

    model_path: str
    index_path: str = ""
    f0_method: str = "rmvpe"
    embedder: str = "contentvec"
    pitch_shift: int = 0
    index_rate: float = 0.0
    protect: float = 0.33
    sid: int = 0

    stream_sr: int = STREAM_SR_DEFAULT
    block_ms: float = 250.0
    context_ms: float = 2500.0           # = w-okada "额外推理时长" 2.5 s
    crossfade_ms: float = 50.0

    device: str = "cuda"

    # Silence gate (响应阈值 / w-okada silentThreshold): below this input level
    # (dBFS) emit clean silence and SKIP the GPU inference (the convert buffer +
    # F0 cache freeze together, so they stay in sync). A hangover keeps soft
    # trailing syllables. Input-side (decides WHETHER to convert; never reshapes
    # output). Default OFF (an over-high threshold would gate out soft speech).
    silence_threshold_dbfs: Optional[float] = None
    silence_hangover_ms: float = 250.0

    # Input-side noise reduction (Applio's noisereduce TorchGate). Cleans the
    # CARRIER before conversion so ambient noise is not converted into voice.
    # This is input conditioning (like pitch transpose), NOT output reshaping --
    # the model still defines the voice. Default OFF so a clean mic / soft speech
    # is never silently degraded.
    denoise: bool = False
    denoise_strength: float = 0.5        # prop_decrease 0..1 (1 = most aggressive)
    denoise_nonstationary: bool = True   # adapt to time-varying noise

    # Input-side FORMANT / gender shift (性别因子) -- Applio's stftpitchshift with
    # factors=1: PITCH is unchanged, only the spectral envelope (formants/vocal
    # tract) moves. Input conditioning like pitch transpose, NOT output reshaping
    # -- the model still defines the voice. timbre>1 = formants up (brighter/
    # feminine), <1 = down (deeper/masculine), 1.0 = off. Default OFF.
    formant_shift: bool = False
    formant_qfrency: float = 1.0         # cepstral lifter detail (Applio default 1.0)
    formant_timbre: float = 1.0          # the gender knob (1.0 = no shift)

    # Input-side F0 conditioning (change WHAT F0 the model is driven with, like
    # --pitch; not output reshaping). Both default OFF.
    f0_autotune: bool = False            # snap detected F0 to the nearest semitone
    f0_autotune_strength: float = 1.0
    proposed_pitch: bool = False         # auto-derive a transpose from the median F0
    proposed_pitch_threshold: float = 155.0

    # Input-side AUTO pitch-centering: track the user's running median F0 (slow EMA,
    # voiced-only, frozen on silence) and ADD a decimal transpose that centers it on
    # the model's target median Hz -- so any model lands the carrier in its comfort
    # range without a hand-tuned per-model pitch_shift. Changes WHAT F0 drives the
    # model (like --pitch), not the model's output. Default OFF. Unlike the vendored
    # proposed_pitch (per-block median -> flattens sentence prosody), the slow EMA
    # (tau >> sentence scale ~1-3s) preserves within-utterance intonation.
    auto_center: bool = False
    auto_center_target_hz: float = 0.0   # the model's target median F0 (per-model seed)
    auto_center_tau_s: float = 20.0      # EMA time constant (>> sentence prosody)
    auto_center_limit: float = 12.0      # clamp the auto offset to +-this many semitones

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required")
        if not (0.0 <= self.index_rate <= 1.0):
            raise ValueError("index_rate must be in [0,1]")
        if not (0.0 <= self.protect <= 0.5):
            raise ValueError("protect must be in [0,0.5]")
        if self.block_ms <= 0 or self.context_ms < 0 or self.crossfade_ms < 0:
            raise ValueError("block_ms>0, context_ms>=0, crossfade_ms>=0 required")
        if self.f0_method not in ("rmvpe", "fcpe"):
            raise ValueError(
                f"unsupported f0_method {self.f0_method!r}; the realtime engine "
                "backs only 'rmvpe' and 'fcpe'"
            )


class StreamingRvcEngine:
    """Stateful, faithful realtime RVC engine (Applio persistent-buffer core)."""

    def __init__(self, config: StreamingEngineConfig) -> None:
        config.validate()
        self.cfg = config
        self.stream_sr = int(config.stream_sr)
        self.pitch_shift = int(config.pitch_shift)
        self.index_rate = float(config.index_rate)
        self.protect = float(config.protect)
        self.f0_autotune = bool(config.f0_autotune)
        self.f0_autotune_strength = float(config.f0_autotune_strength)
        self.proposed_pitch = bool(config.proposed_pitch)
        self.proposed_pitch_threshold = float(config.proposed_pitch_threshold)
        self.num_speakers = 1          # trained speaker count (set in load from emb_g)

        # Auto pitch-centering (input-side; default OFF). The tracker (worker thread,
        # in process_block) is the ONLY writer of _auto_offset; _convert reads it on
        # the same thread, gated by the _auto_center_on flag -> no cross-thread race.
        self._auto_center_on = bool(config.auto_center)
        self._auto_offset = 0.0        # decimal semitones added to f0_up_key
        self._user_f0_ema = None       # running EMA of the user's median F0 (Hz)
        self._track_buf = None         # rolling 16k input window for the F0 tracker
        self._track_fill = 0
        self._track_counter = 0
        self._track_interval_blocks = 1

        # Serialises live set_* calls (control thread) against each other; the
        # audio worker reads the gated state lock-free (flag flipped LAST after
        # any object is built — see the set_* methods).
        self._lock = threading.RLock()

        self._loaded = False
        self._torch = None
        self._F = None
        self._circular_write = None
        self._pipeline = None
        self._f0_models = {}          # method -> built RMVPE/FCPE predictor (lazy cache)
        self._denoiser = None
        self._denoise_on = bool(config.denoise)   # runtime gate (live-toggleable)
        self._formant = None          # stftpitchshift instance (input-side gender)
        self._formant_on = bool(config.formant_shift)  # runtime gate (live-toggleable)
        self._formant_pad = 0
        self._formant_hist = None     # overlap-save history for block-seam continuity
        self._silence_thresh_lin = None   # linear RMS threshold (None = gate off)
        self._silence_hangover_blocks = 0
        self._silence_hangover_left = 0
        self._device = None
        self._dtype = None
        self.tgt_sr: Optional[int] = None
        self._resolved_device: Optional[str] = None
        self._cuda_name: Optional[str] = None

        # Frame sizes (filled by _realloc).
        self.block_frame = 0          # input block @ stream SR (samples/process_block)
        self.block_frame_16k = 0
        self.crossfade_frame = 0      # @ stream SR
        self.sola_search_frame = 0    # @ stream SR
        self._convert_size_16k = 0
        self._convert_feature_size_16k = 0
        self._skip_head = 0
        self._return_length = 0
        self._warmup = 0

        # Persistent state tensors (filled by _realloc / reset).
        self._convert_buffer = None
        self._pitch_buffer = None
        self._pitchf_buffer = None
        self._sola_buffer = None
        self._fade_in = None
        self._fade_out = None
        self._ones_kernel = None
        self._resample_in = None
        self._resample_out = None

        # Observable last-block state. The engine stays metrics-agnostic
        # (faithful): it only records what it just did; the worker reads these
        # after each process_block to update RuntimeMetrics.
        self.last_sola_offset = 0        # SOLA cut offset chosen for the last seam
        self.last_silence_skipped = False  # True if the last block was gated to silence

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._loaded:
            return
        # Applio's loaders resolve model files against CWD ("rvc/models/...") and
        # capture os.getcwd() at import time. Ensure CWD == RVC_ROOT BEFORE importing
        # the vendored stack so those relative paths point at RVC/rvc/models/.
        if os.path.abspath(os.getcwd()) != RVC_ROOT:
            os.chdir(RVC_ROOT)
        if VENDOR_DIR not in sys.path:
            sys.path.insert(0, VENDOR_DIR)

        try:
            import torch
            import torch.nn.functional as F
            import torchaudio.transforms as tat
            from rvc.realtime.pipeline import create_pipeline
            from rvc.realtime.utils.torch import circular_write
        except Exception as exc:  # pragma: no cover - env-specific
            raise StreamingEngineError(
                f"failed to import torch / vendored Applio stack (run in .venv-applio): {exc}"
            ) from exc

        self._torch = torch
        self._F = F
        self._circular_write = circular_write

        # Best-effort GPU label for the status banner. `_resolved_device` is set
        # authoritatively below from the pipeline's ACTUAL device -- the vendored
        # Config picks cuda:0 whenever CUDA is available, independent of
        # cfg.device -- so we report the real device, not the requested one.
        if torch.cuda.is_available():
            try:
                self._cuda_name = str(torch.cuda.get_device_name(0))
            except Exception:
                self._cuda_name = None

        # Preflight: only allow an embedder that is actually staged under
        # rvc/models/embedders, so an unknown/unstaged name fails with a clear
        # error instead of a deep KeyError or a surprise mid-run HF download
        # (the vendored load_embedding would otherwise wget it).
        emb_dir = os.path.join(RVC_ROOT, "rvc", "models", "embedders", self.cfg.embedder)
        if not os.path.isdir(emb_dir):
            raise StreamingEngineError(
                f"embedder {self.cfg.embedder!r} is not staged under "
                f"rvc/models/embedders ({emb_dir}); stage it first or use 'contentvec'."
            )

        try:
            self._pipeline = create_pipeline(
                self.cfg.model_path,
                self.cfg.index_path or "",
                self.cfg.f0_method,
                self.cfg.embedder,
                None,
                self.cfg.sid,
            )
        except Exception as exc:
            raise StreamingEngineError(f"create_pipeline failed: {exc}") from exc

        # v2-only build: this program runs ONLY v2-series (768-dim) RVC models.
        # The vendored loader sets `version` from the checkpoint, defaulting to
        # 'v1' for a version-less .pth (pipeline.py: cpt.get("version", "v1")).
        # Reject anything that is not v2 HERE -- before any buffers are
        # allocated -- so a v1 / 256-dim model fails loudly with a clear message
        # instead of silently running the wrong-dimension path.
        ver = str(getattr(self._pipeline, "version", "") or "").lower()
        if ver != "v2":
            raise StreamingEngineError(
                f"v2-only build: model {self.cfg.model_path!r} is "
                f"{ver or 'unknown'} (256-dim); load a v2 / 768-dim RVC model."
            )

        # sid range check against the model's trained speaker count (emb_g width).
        try:
            n_spk = int(self._pipeline.vc.cpt["weight"]["emb_g.weight"].shape[0])
        except Exception:
            n_spk = 1
        self.num_speakers = n_spk
        if not (0 <= self.cfg.sid < n_spk):
            raise StreamingEngineError(
                f"sid {self.cfg.sid} out of range for this model (valid 0..{n_spk - 1})."
            )

        self._device = self._pipeline.device
        self._resolved_device = "cuda" if str(self._device).startswith("cuda") else "cpu"
        self._dtype = self._pipeline.dtype          # torch.float32 (Applio realtime)
        self.tgt_sr = int(self._pipeline.tgt_sr)
        self._tat = tat

        # Seed the f0-predictor cache with the one the pipeline built at load, so
        # a later set_f0_method() can swap rmvpe<->fcpe without rebuilding it.
        self._f0_models = {self.cfg.f0_method: self._pipeline.f0_model}

        # Optional input-side machinery. Each object is built only when its
        # feature starts enabled; the set_* methods build it lazily on first
        # enable (always on the control thread, never the audio callback). The
        # per-block gates key off the _on flags so toggling needs no reload.
        if self.cfg.denoise:
            self._build_denoiser()
        if self.cfg.formant_shift:
            self._build_formant()
        if self.cfg.silence_threshold_dbfs is not None:
            self._apply_silence_cfg(self.cfg.silence_threshold_dbfs, self.cfg.silence_hangover_ms)

        self._realloc()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def resolved_device(self) -> Optional[str]:
        return self._resolved_device

    @property
    def cuda_device_name(self) -> Optional[str]:
        return self._cuda_name

    @property
    def resolved_precision(self) -> str:
        # Applio's realtime converter is float32 by design (see pipeline.py dtype).
        return "fp32 (applio realtime)"

    # -------------------------------------------------------------- internals
    def _realloc(self) -> None:
        torch = self._torch
        tat = self._tat
        asr = self.stream_sr
        window = SR16 // 100  # 160

        self.block_frame = int(round(self.cfg.block_ms / 1000.0 * asr))
        # keep block_frame divisible by 3 so 48k->16k is an exact integer (no drift)
        self.block_frame -= self.block_frame % 3
        self.crossfade_frame = int(round(self.cfg.crossfade_ms / 1000.0 * asr))
        extra_frame = int(round(self.cfg.context_ms / 1000.0 * asr))
        self.sola_search_frame = asr // 100  # 10 ms

        self.block_frame_16k = int(self.block_frame / asr * SR16)
        crossfade_16k = int(self.crossfade_frame / asr * SR16)
        sola_search_16k = int(self.sola_search_frame / asr * SR16)
        extra_16k = int(extra_frame / asr * SR16)

        convert_size_16k = self.block_frame_16k + sola_search_16k + extra_16k + crossfade_16k
        mod = convert_size_16k % window
        if mod != 0:
            convert_size_16k += window - mod
        self._convert_size_16k = convert_size_16k
        self._convert_feature_size_16k = convert_size_16k // window
        self._skip_head = extra_16k // window
        self._return_length = self._convert_feature_size_16k - self._skip_head
        # blocks needed to fill convert_buffer before real output (emit zeros meanwhile)
        self._warmup = int(np.ceil(convert_size_16k / max(self.block_frame_16k, 1))) + 1

        dev, dt = self._device, self._dtype
        self._convert_buffer = torch.zeros(convert_size_16k, dtype=dt, device=dev)
        # +1 frame headroom for the pitch estimator's extra output (Applio).
        self._pitch_buffer = torch.zeros(self._convert_feature_size_16k + 1, dtype=torch.int64, device=dev)
        self._pitchf_buffer = torch.zeros(self._convert_feature_size_16k + 1, dtype=dt, device=dev)

        # output-side seam state @ stream SR
        self._sola_buffer = torch.zeros(self.crossfade_frame, device=dev, dtype=torch.float32)
        fade_in = (torch.sin(0.5 * np.pi * torch.linspace(
            0.0, 1.0, steps=self.crossfade_frame, device=dev, dtype=torch.float32)) ** 2)
        self._fade_in = fade_in
        self._fade_out = 1.0 - fade_in
        self._ones_kernel = torch.ones(1, 1, self.crossfade_frame, device=dev, dtype=torch.float32)

        self._resample_in = tat.Resample(orig_freq=asr, new_freq=SR16, dtype=torch.float32).to(dev)
        self._resample_out = tat.Resample(orig_freq=self.tgt_sr, new_freq=asr, dtype=torch.float32).to(dev)

        # auto-center F0 tracker: ~1s ring of 16k input + a cadence (the EMA is slow,
        # so we only re-estimate every ~1s -> negligible amortized F0 cost).
        self._track_buf = np.zeros(max(SR16, self.block_frame_16k * 2), dtype=np.float32)
        self._track_fill = 0
        self._track_counter = 0
        self._track_interval_blocks = max(1, int(round(1.0 / max(self.cfg.block_ms / 1000.0, 1e-3))))

    def reset(self) -> None:
        """Clear streaming state (e.g. after a silence skip or fallback)."""
        if not self._loaded:
            return
        self._convert_buffer.zero_()
        self._pitch_buffer.zero_()
        self._pitchf_buffer.zero_()
        self._sola_buffer.zero_()
        if self._formant_hist is not None:
            self._formant_hist[:] = 0.0
        self._silence_hangover_left = 0
        self.last_sola_offset = 0
        self.last_silence_skipped = False
        self._auto_offset = 0.0
        self._user_f0_ema = None
        if self._track_buf is not None:
            self._track_buf[:] = 0.0
        self._track_fill = 0
        self._track_counter = 0
        self._warmup = int(np.ceil(self._convert_size_16k / max(self.block_frame_16k, 1))) + 1

    # --------------------------------------------------------- live control
    # All setters are INPUT-side conditioning only (faithful-carrier contract:
    # they change WHAT speech is converted, never the model's output). They run
    # on the control thread; the audio worker reads the mutated state fresh each
    # block. Object-building toggles (formant/denoise) build on THIS thread and
    # flip the runtime flag LAST, so the audio thread never sees a half-built
    # object. None of these resize the persistent torch buffers (block/context/
    # crossfade are load-time only — change them via a fresh load()).

    def _build_denoiser(self) -> None:
        """Build the input-side noisereduce TorchGate from cfg (strength /
        nonstationary are constructor-baked, so a change rebuilds it)."""
        try:
            from noisereduce.torchgate import TorchGate
        except Exception as exc:
            raise StreamingEngineError(
                "denoise requested but noisereduce is missing: "
                f"pip install noisereduce ({exc})"
            ) from exc
        self._denoiser = TorchGate(
            sr=self.stream_sr,
            nonstationary=bool(self.cfg.denoise_nonstationary),
            prop_decrease=float(self.cfg.denoise_strength),
        ).to(self._device)

    def _build_formant(self) -> None:
        """Build the input-side stftpitchshift formant shifter + overlap-save
        history (factors=1 -> pitch untouched, only the spectral envelope)."""
        try:
            from stftpitchshift import StftPitchShift
        except Exception as exc:
            raise StreamingEngineError(
                "formant shift requested but stftpitchshift is missing: "
                f"pip install stftpitchshift ({exc})"
            ) from exc
        self._formant = StftPitchShift(1024, 32, self.stream_sr)
        self._formant_pad = 1024
        self._formant_hist = np.zeros(self._formant_pad, dtype=np.float32)

    def _apply_silence_cfg(self, dbfs, hangover_ms) -> None:
        """(Re)compute the silence-gate linear RMS threshold + hangover blocks;
        ``dbfs=None`` disables the gate."""
        if dbfs is None:
            self._silence_thresh_lin = None
            self._silence_hangover_blocks = 0
            self._silence_hangover_left = 0
            return
        self._silence_thresh_lin = float(10.0 ** (float(dbfs) / 20.0))
        self._silence_hangover_blocks = int(
            np.ceil(float(hangover_ms) / max(float(self.cfg.block_ms), 1.0))
        )

    def _index_loaded(self) -> bool:
        return bool(
            self._loaded
            and self._pipeline is not None
            and getattr(self._pipeline, "index", None) is not None
        )

    def set_pitch_shift(self, semitones) -> None:
        """Transpose the input F0 (变调), in semitones. Live."""
        with self._lock:
            self.pitch_shift = int(semitones)

    def set_protect(self, p) -> None:
        """Voiceless-consonant protection (0..0.5). Live."""
        p = float(p)
        if not (0.0 <= p <= 0.5):
            raise ValueError(f"protect must be in [0, 0.5], got {p}")
        with self._lock:
            self.protect = p

    def set_index_rate(self, r) -> None:
        """FAISS retrieval strength (0..1). Requires an index to be loaded when
        > 0 (see set_index_path / load with index_path). Live."""
        r = float(r)
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"index_rate must be in [0, 1], got {r}")
        if r > 0.0 and not self._index_loaded():
            raise StreamingEngineError(
                "index_rate > 0 but no FAISS index is loaded; load the engine "
                "with an index_path or call set_index_path() first."
            )
        with self._lock:
            self.index_rate = r

    def set_index_path(self, path) -> None:
        """Load (or swap, or drop with '') the FAISS retrieval index live.
        Control-thread only (faiss.read_index can take tens of ms)."""
        if not self._loaded or self._pipeline is None:
            raise StreamingEngineError("engine not loaded; call load() first")
        from rvc.realtime.pipeline import load_faiss_index
        index, big_npy = load_faiss_index(
            str(path or "").strip().strip('"').replace("trained", "added")
        )
        with self._lock:
            self._pipeline.big_npy = big_npy
            self._pipeline.index = index
            if index is None and self.index_rate > 0.0:
                self.index_rate = 0.0   # nothing to retrieve from

    def set_autotune(self, on, strength=None) -> None:
        """Input-side F0 autotune (snap to nearest semitone). Live."""
        if strength is not None:
            strength = float(strength)
            if not (0.0 <= strength <= 1.0):
                raise ValueError(f"autotune strength must be in [0, 1], got {strength}")
        with self._lock:
            self.f0_autotune = bool(on)
            if strength is not None:
                self.f0_autotune_strength = strength

    def set_auto_pitch(self, on, threshold=None) -> None:
        """Input-side auto transpose from median F0 toward ``threshold`` Hz. Live."""
        if threshold is not None:
            threshold = float(threshold)
            if threshold <= 0.0:
                raise ValueError(f"auto-pitch threshold (Hz) must be > 0, got {threshold}")
        with self._lock:
            self.proposed_pitch = bool(on)
            if threshold is not None:
                self.proposed_pitch_threshold = threshold

    def set_auto_center(self, on, target_hz=None, tau_s=None) -> None:
        """Input-side AUTO pitch-centering: track the user's median F0 (slow EMA,
        voiced-only) and ADD a decimal transpose that centers it on ``target_hz``.
        Live. When OFF the auto offset is ignored (gated in ``_convert``)."""
        if target_hz is not None:
            target_hz = float(target_hz)
            if target_hz <= 0.0:
                raise ValueError(f"auto-center target_hz must be > 0, got {target_hz}")
        if tau_s is not None:
            tau_s = float(tau_s)
            if tau_s <= 0.0:
                raise ValueError(f"auto-center tau_s must be > 0, got {tau_s}")
        with self._lock:
            if target_hz is not None:
                self.cfg.auto_center_target_hz = target_hz
            if tau_s is not None:
                self.cfg.auto_center_tau_s = tau_s
            if not on:
                self._user_f0_ema = None
            self._auto_center_on = bool(on)

    def _update_auto_center(self, a16) -> None:
        """Slow-EMA tracker (worker thread): every ``_track_interval_blocks`` blocks,
        re-estimate the user's median F0 on a rolling ~1s window (same predictor the
        model uses), update a tau-smoothed EMA (FROZEN when the window is unvoiced),
        and set the decimal ``_auto_offset`` that centers it on the target median Hz.
        Best-effort: any hiccup just skips the update -- it never breaks the audio."""
        try:
            x = a16.detach().to("cpu").numpy().astype(np.float32).reshape(-1)
        except Exception:
            return
        buf = self._track_buf
        if buf is None:
            return
        n = min(x.shape[0], buf.shape[0])
        if n <= 0:
            return
        buf[:-n] = buf[n:]
        buf[-n:] = x[-n:]
        self._track_fill = min(self._track_fill + n, buf.shape[0])
        self._track_counter += 1
        if self._track_counter < self._track_interval_blocks:
            return
        self._track_counter = 0
        if self._track_fill < self.block_frame_16k * 2:
            return
        seg = buf[-self._track_fill:].copy()
        with self._lock:                       # snapshot the coupled (model, method) pair
            f0_model = getattr(self._pipeline, "f0_model", None)
            f0_method = getattr(self._pipeline, "f0_method", "rmvpe")
            target = float(self.cfg.auto_center_target_hz)
            tau = max(float(self.cfg.auto_center_tau_s), 1e-3)
            limit = float(self.cfg.auto_center_limit)
        if f0_model is None or target <= 0.0:
            return
        try:                                   # estimate OFF the lock (slow); swallow noise
            with contextlib.redirect_stdout(io.StringIO()):
                if f0_method == "fcpe":
                    f0 = f0_model.get_f0(seg, seg.shape[0] // 160, filter_radius=0.006)
                else:
                    f0 = f0_model.get_f0(seg, filter_radius=0.03)
        except Exception:
            return
        f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
        voiced = f0[f0 > 0]
        if voiced.shape[0] < 8:                # mostly unvoiced -> FREEZE the EMA
            return
        med = float(np.median(voiced))
        if not np.isfinite(med) or med <= 0:
            return
        dt = self._track_interval_blocks * (self.cfg.block_ms / 1000.0)
        alpha = 1.0 - float(np.exp(-dt / tau))
        ema = self._user_f0_ema                # snapshot (single read; race-safe)
        ema = med if ema is None else ema + alpha * (med - ema)
        self._user_f0_ema = ema
        self._auto_offset = max(-limit, min(limit, 12.0 * float(np.log2(target / ema))))

    def set_formant(self, on, timbre=None, qfrency=None) -> None:
        """Input-side formant / gender shift (性别因子). timbre>1 brighter, <1
        deeper, 1.0 neutral. Builds the shifter lazily on first enable. Live."""
        if timbre is not None:
            timbre = float(timbre)
            if timbre <= 0.0:
                raise ValueError(f"formant timbre must be > 0, got {timbre}")
        if qfrency is not None:
            qfrency = float(qfrency)
            if qfrency <= 0.0:
                raise ValueError(f"formant qfrency must be > 0, got {qfrency}")
        with self._lock:
            if timbre is not None:
                self.cfg.formant_timbre = timbre
            if qfrency is not None:
                self.cfg.formant_qfrency = qfrency
            if on and self._formant is None:
                self._build_formant()        # build BEFORE flipping the flag on
            self._formant_on = bool(on)

    def set_denoise(self, on, strength=None, nonstationary=None) -> None:
        """Input-side noise reduction before conversion. A strength /
        nonstationary change rebuilds the gate. Live."""
        if strength is not None:
            strength = float(strength)
            if not (0.0 <= strength <= 1.0):
                raise ValueError(f"denoise strength must be in [0, 1], got {strength}")
        with self._lock:
            rebuild = False
            if strength is not None and strength != float(self.cfg.denoise_strength):
                self.cfg.denoise_strength = strength
                rebuild = True
            if nonstationary is not None and bool(nonstationary) != bool(self.cfg.denoise_nonstationary):
                self.cfg.denoise_nonstationary = bool(nonstationary)
                rebuild = True
            if on and (self._denoiser is None or rebuild):
                self._build_denoiser()       # build/rebuild BEFORE flipping on
            self._denoise_on = bool(on)

    def set_silence_gate(self, dbfs, hangover_ms=None) -> None:
        """Silence gate (响应阈值): below ``dbfs`` input level, emit silence and
        skip inference. ``dbfs=None`` disables. Live."""
        if hangover_ms is not None:
            hangover_ms = float(hangover_ms)
            if hangover_ms < 0.0:
                raise ValueError(f"silence hangover_ms must be >= 0, got {hangover_ms}")
        with self._lock:
            self.cfg.silence_threshold_dbfs = None if dbfs is None else float(dbfs)
            if hangover_ms is not None:
                self.cfg.silence_hangover_ms = hangover_ms
            self._apply_silence_cfg(self.cfg.silence_threshold_dbfs, self.cfg.silence_hangover_ms)

    def set_sid(self, sid) -> None:
        """Select the trained speaker (声线) of a multi-speaker model. Live —
        the vendored pipeline reads ``torch_sid`` fresh per block. Faithful: it
        picks among the model's OWN trained voices; it does not reshape output."""
        if not self._loaded or self._pipeline is None:
            raise StreamingEngineError("engine not loaded; call load() first")
        sid = int(sid)
        if not (0 <= sid < self.num_speakers):
            raise StreamingEngineError(
                f"sid {sid} out of range (valid 0..{self.num_speakers - 1})"
            )
        torch = self._torch
        with self._lock:
            self._pipeline.sid = sid
            self._pipeline.torch_sid = torch.tensor(
                [sid], device=self._device, dtype=torch.int64
            )
            self.cfg.sid = sid

    def set_f0_method(self, method) -> None:
        """Swap the F0 estimator (rmvpe <-> fcpe) live, without a full reload.
        Input-side carrier conditioning -- it changes how the carrier's F0 is
        ESTIMATED, never the model's output -> faithful-carrier contract intact.

        The vendored get_f0 reads a COUPLED (f0_method string, f0_model object)
        pair per block; the swap is published under self._lock and the audio
        worker reads it under the same lock (see _convert), so a block never sees
        a crossed pair. The new predictor is BUILT off the lock (file load); only
        the three references are assigned under it. The F0 cache is intentionally
        NOT cleared (both methods write the same coarse-pitch+Hz format at the
        same hop -> the newest window is overwritten next block, no seam/pop)."""
        if not self._loaded or self._pipeline is None:
            raise StreamingEngineError("engine not loaded; call load() first")
        method = str(method)
        if method not in ("rmvpe", "fcpe"):
            raise StreamingEngineError(
                f"unsupported f0_method {method!r}; the realtime engine backs "
                "only 'rmvpe' and 'fcpe'"
            )
        if method == self.cfg.f0_method:
            return
        # Preflight the predictor weight: RMVPE/FCPE only torch.load this path
        # (no surprise download), so a clean error beats a deep loader error.
        weight = "rmvpe.pt" if method == "rmvpe" else "fcpe.pt"
        wpath = os.path.join(RVC_ROOT, "rvc", "models", "predictors", weight)
        if not os.path.isfile(wpath):
            raise StreamingEngineError(
                f"f0_method {method!r} needs {wpath}, which is missing; stage it first."
            )
        # Build (or reuse) the predictor OFF the lock (file load + GPU upload).
        model = self._f0_models.get(method)
        if model is None:
            model = self._pipeline.setup_f0(method)
            self._f0_models[method] = model
        with self._lock:
            self._pipeline.f0_model = model
            self._pipeline.f0_method = method
            self.cfg.f0_method = method

    def _fit16(self, t, n: int):
        if t.shape[0] == n:
            return t
        if t.shape[0] > n:
            return t[:n]
        torch = self._torch
        pad = torch.zeros(n - t.shape[0], device=t.device, dtype=t.dtype)
        return torch.cat([t, pad])

    def _convert(self):
        def _call():
            return self._pipeline.voice_conversion(
                self._convert_buffer, self._pitch_buffer, self._pitchf_buffer,
                f0_up_key=self.pitch_shift + (self._auto_offset if self._auto_center_on else 0.0),
                index_rate=self.index_rate,
                p_len=self._convert_feature_size_16k, silence_front=0,
                skip_head=self._skip_head, return_length=self._return_length,
                protect=self.protect, volume_envelope=1,        # 1 => no change_rms
                f0_autotune=self.f0_autotune,
                f0_autotune_strength=self.f0_autotune_strength,
                proposed_pitch=self.proposed_pitch,
                proposed_pitch_threshold=self.proposed_pitch_threshold,
                reduced_noise=None, board=None,                 # no FX / noise-reduce
                block_size_16k=self.block_frame_16k,
            )
        # Hold the engine lock across the inference so a concurrent live setter
        # that mutates the pipeline (set_f0_method's coupled (f0_method,f0_model)
        # swap; set_sid / set_index_path) is serialized against the F0/index read
        # inside voice_conversion -- the worker never sees a half-applied change.
        # Uncontended ~100ns/block; a setter waits <= one inference (~45ms). The
        # audio callback runs on a separate thread and is unaffected.
        # Also swallow stdout during the call: the vendored pipeline
        # (proposed_pitch's 'calculated pitch offset:') and torchfcpe (a benign
        # per-block mel range warning when a loud carrier overshoots ±1 after the
        # 48k->16k resample) print noise every block. Real failures RAISE (-> worker
        # fallback + metrics); they never print -- so this only hides cosmetic spam,
        # never an actionable error, and the audio is untouched.
        with self._lock, contextlib.redirect_stdout(io.StringIO()):
            return _call()

    def _sola_crossfade(self, audio):
        torch = self._torch
        F = self._F
        cf, ss, bf = self.crossfade_frame, self.sola_search_frame, self.block_frame
        conv_input = audio[None, None, : cf + ss]
        cor_nom = F.conv1d(conv_input, self._sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input ** 2, self._ones_kernel) + 1e-8)
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item())
        self.last_sola_offset = sola_offset
        audio = audio[sola_offset:].clone()
        # short sin² crossfade with the previous block's emitted tail (sanctioned seam)
        audio[:cf] = audio[:cf] * self._fade_in + self._sola_buffer * self._fade_out
        need = bf + cf
        if audio.shape[0] < need:
            audio = torch.cat([audio, torch.zeros(need - audio.shape[0],
                                                  device=audio.device, dtype=audio.dtype)])
        self._sola_buffer[:] = audio[bf: bf + cf]
        return audio[:bf]

    # --------------------------------------------------------------- process
    def process_block(self, block: np.ndarray, sr: int) -> np.ndarray:
        """Convert ONE block (``block_frame`` mono float32 @ ``stream_sr``).

        Returns ``block_frame`` mono float32 @ ``stream_sr``. During warm-up
        (buffer not yet full) returns zeros. Stateful: must be called with
        consecutive blocks in order.
        """
        if not self._loaded:
            raise StreamingEngineError("engine not loaded; call load() first")
        torch = self._torch
        block = np.ascontiguousarray(block, dtype=np.float32).reshape(-1)
        if int(sr) != self.stream_sr:
            raise StreamingEngineError(
                f"block sr {sr} != engine stream_sr {self.stream_sr}"
            )
        # The worker feeds exactly block_frame; pad/truncate defensively.
        if block.shape[0] != self.block_frame:
            if block.shape[0] < self.block_frame:
                block = np.concatenate([block, np.zeros(self.block_frame - block.shape[0], np.float32)])
            else:
                block = block[: self.block_frame]

        self.last_silence_skipped = False

        # Silence gate (响应阈值): decide on the RAW input level (before formant /
        # denoise). Below threshold -> not voiced; a hangover keeps soft trailing
        # syllables converting for a few blocks after the last loud block.
        voiced = True
        if self._silence_thresh_lin is not None:
            rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
            if rms >= self._silence_thresh_lin:
                self._silence_hangover_left = self._silence_hangover_blocks
            voiced = self._silence_hangover_left > 0
            if self._silence_hangover_left > 0:
                self._silence_hangover_left -= 1

        # input-side FORMANT / gender shift (性别因子) BEFORE conversion: moves the
        # spectral envelope only (pitch untouched). Overlap-save -- prepend the
        # previous block's tail, shift, drop the prepended region -- keeps the
        # block seam continuous so realtime matches a whole-file render.
        if self._formant_on and self._formant is not None:
            pad = self._formant_pad
            catted = np.concatenate([self._formant_hist, block])
            shifted = self._formant.shiftpitch(
                catted, factors=1,
                quefrency=self.cfg.formant_qfrency * 1e-3,
                distortion=self.cfg.formant_timbre,
            ).astype(np.float32)
            self._formant_hist = catted[-pad:].astype(np.float32)
            block = np.ascontiguousarray(shifted[pad:pad + self.block_frame])

        with torch.no_grad():
            x = torch.as_tensor(block, dtype=torch.float32, device=self._device)
            if self._denoise_on and self._denoiser is not None:
                # input-side denoise: clean the carrier BEFORE conversion so
                # ambient noise is not faithfully converted into warbly voice.
                x = self._denoiser(x.unsqueeze(0)).squeeze(0).to(torch.float32)
            a16 = self._fit16(self._resample_in(x).to(self._dtype), self.block_frame_16k)
            if self._auto_center_on:
                self._update_auto_center(a16)

            if self._warmup > 0:
                self._warmup -= 1
                self._circular_write(a16, self._convert_buffer)
                self._convert()  # keep GPU warm + advance the F0 cache during fill
                return np.zeros(self.block_frame, dtype=np.float32)

            if not voiced:
                # Below the silence threshold: emit clean silence and SKIP the GPU
                # inference. Freeze the convert buffer + F0 cache together (skip the
                # write AND _convert) so they stay in sync; zero the seam so speech
                # resumes with a clean onset, not a stale-tail click.
                self.last_silence_skipped = True
                self._sola_buffer.zero_()
                return np.zeros(self.block_frame, dtype=np.float32)

            self._circular_write(a16, self._convert_buffer)
            model_out = self._convert().float()
            audio_out = self._resample_out(model_out)   # -> stream SR ; NO ×vol (faithful)
            out = self._sola_crossfade(audio_out)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)
