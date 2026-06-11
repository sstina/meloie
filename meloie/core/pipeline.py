# Derived from Applio (https://github.com/IAHispano/Applio), MIT License,
# (c) 2026 AI Hispano. See meloie/core/NOTICE.md. Internalized 2026-06-10:
#  * device is an EXPLICIT parameter (the upstream Config singleton — and with
#    it the cwd-relative rvc/configs/*.json dependency — is gone);
#  * embedder/predictor weights resolve from explicit absolute dirs;
#  * the faithful-carrier contract is structural: voice_conversion has NO
#    volume-envelope / FX-board / output-denoise / change_rms parameters at all
#    (upstream's dead branches for them were deleted, not just pinned off);
#  * Autotune was lifted out of upstream's 559-line offline infer/pipeline.py,
#    which dragged torchcrepe + a scipy butter filter into the import closure.
"""The realtime RVC conversion pipeline (stateful, block-streaming).

Engine contract (meloie.engine.streaming_engine reads/writes these):
  attributes: f0_model, f0_method, f0_remap, index, big_npy, sid, torch_sid,
              device, dtype, tgt_sr, version, vc (.cpt)
  methods:    setup_f0(method), voice_conversion(...)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize
from torch import Tensor

from .embedder import HubertModelWithFinalProj, load_embedding
from .predictors.f0 import FCPE, RMVPE
from .synth.synthesizer import Synthesizer

import faiss


def circular_write(new_data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Shift ``target`` left by len(new_data) and write ``new_data`` at the tail."""
    offset = new_data.shape[0]
    target[:-offset] = target[offset:].detach().clone()
    target[-offset:] = new_data
    return target


def strip_parametrizations(module: torch.nn.Module):
    """Bake all parametrizations (weight norm) into plain tensors after load."""
    for _, submodule in module.named_modules():
        if hasattr(submodule, "parametrizations"):
            for pname in list(submodule.parametrizations):
                torch.nn.utils.parametrize.remove_parametrizations(
                    submodule, pname, leave_parametrized=True
                )


def load_faiss_index(file_index: str):
    """Read a FAISS index + its reconstructed vectors; (None, None) when absent
    or unreadable (callers decide whether that is an error)."""
    if file_index != "" and os.path.exists(file_index):
        try:
            index = faiss.read_index(file_index)
            big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception as error:
            print(f"An error occurred reading the FAISS index: {error}")
            index = big_npy = None
    else:
        index = big_npy = None

    return index, big_npy


def normalize_index_path(index_path) -> str:
    """Upstream's index-path cleanup: trim quotes/whitespace and map the
    'trained' filename convention to its usable 'added' counterpart."""
    return (
        str(index_path or "")
        .strip()
        .strip('"')
        .strip("\n")
        .strip('"')
        .strip()
        .replace("trained", "added")
    )


class Autotune:
    """Snap an F0 contour toward the nearest semitone (input-side conditioning)."""

    # G1..C6 equal-temperament reference frequencies (Hz).
    NOTE_DICT = [
        49.00, 51.91, 55.00, 58.27, 61.74,
        65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50,
        98.00, 103.83, 110.00, 116.54, 123.47,
        130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00,
        196.00, 207.65, 220.00, 233.08, 246.94,
        261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99,
        392.00, 415.30, 440.00, 466.16, 493.88,
        523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99,
        783.99, 830.61, 880.00, 932.33, 987.77, 1046.50,
    ]

    def autotune_f0(self, f0, f0_autotune_strength):
        autotuned_f0 = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            closest_note = min(self.NOTE_DICT, key=lambda x: abs(x - freq))
            autotuned_f0[i] = freq + (closest_note - freq) * f0_autotune_strength
        return autotuned_f0


class RealtimeVoiceConverter:
    """Loads an RVC .pth checkpoint and owns the Synthesizer (net_g)."""

    def __init__(self, weight_path: str, device: str):
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"model file not found: {weight_path}")
        self.device = device
        self.dtype = torch.float32  # the realtime stack is fp32 by design
        self.cpt = torch.load(weight_path, map_location="cpu", weights_only=True)

        self.tgt_sr = self.cpt["config"][-1]
        # Patch the speaker count from the actual embedding table (some
        # checkpoints carry a stale config value).
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        self.use_f0 = self.cpt.get("f0", 1)
        # Version-less checkpoints are v1 by convention; the engine's v2-only
        # guard relies on this default to reject them.
        self.version = self.cpt.get("version", "v1")
        self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")

        self.net_g = Synthesizer(
            *self.cpt["config"],
            use_f0=self.use_f0,
            text_enc_hidden_dim=768 if self.version == "v2" else 256,
            vocoder=self.vocoder,
        )
        # strict=False: checkpoints carry training-only keys (enc_q.*) that the
        # inference graph deliberately does not build.
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        strip_parametrizations(self.net_g)
        self.net_g = self.net_g.to(device).to(self.dtype)
        self.net_g.eval()

    def inference(
        self,
        feats: Tensor,
        p_len: Tensor,
        sid: Tensor,
        pitch: Tensor,
        pitchf: Tensor,
        rate: Tensor = None,
    ):
        output = self.net_g.infer(feats, p_len, pitch, pitchf, sid, rate)[0][0, 0]
        # Hard clip to DAC-valid range: identity for in-range samples (a sample
        # validity guard, not tonal shaping).
        return torch.clip(output, -1.0, 1.0, out=output)


class RealtimePipeline:
    """Block-streaming conversion: persistent caches live in the ENGINE; this
    class converts one persistent convert-buffer view per call."""

    def __init__(
        self,
        vc: RealtimeVoiceConverter,
        hubert_model: HubertModelWithFinalProj,
        index,
        big_npy,
        f0_method: str,
        sid: int,
        predictor_dir: str,
    ):
        self.vc = vc
        self.hubert_model = hubert_model
        self.index = index
        self.big_npy = big_npy
        self.use_f0 = vc.use_f0
        self.version = vc.version
        self.f0_method = f0_method
        self.sample_rate = 16000
        self.tgt_sr = vc.tgt_sr
        self.window = 160
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.device = vc.device
        self.dtype = vc.dtype
        self.sid = sid
        self.torch_sid = torch.tensor([sid], device=self.device, dtype=torch.int64)
        self.autotune = Autotune()
        self.predictor_dir = predictor_dir
        # Precise CDF F0 map (input-side carrier conditioning). None = inactive.
        # Installed live by the engine's set_precise_mapping under its lock.
        self.f0_remap = None
        self.f0_model = self.setup_f0(f0_method)
        # Reuse scalar tensors to avoid per-block allocations.
        self._rate_tensor = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._p_len_tensor = torch.zeros(1, device=self.device, dtype=torch.int64)

    def setup_f0(self, f0_method: str):
        """Build an F0 estimator from the staged predictor weights."""
        if f0_method == "rmvpe":
            return RMVPE(
                os.path.join(self.predictor_dir, "rmvpe.pt"),
                device=self.device,
                sample_rate=self.sample_rate,
                hop_size=self.window,
            )
        if f0_method == "fcpe":
            return FCPE(
                os.path.join(self.predictor_dir, "fcpe.pt"),
                device=self.device,
                sample_rate=self.sample_rate,
                hop_size=self.window,
            )
        raise ValueError(f"unsupported f0_method {f0_method!r} (rmvpe/fcpe)")

    def get_f0(
        self,
        x: Tensor,
        f0_up_key: float = 0,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        """Estimate F0 on the newest window, apply input-side F0 conditioning
        (precise map / autotune / proposed pitch / transpose), and return the
        (coarse 1..255 bucket, Hz) pair."""
        if torch.is_tensor(x):
            x = x.cpu().numpy()

        if self.f0_method == "rmvpe":
            f0 = self.f0_model.get_f0(x, filter_radius=0.03)
        elif self.f0_method == "fcpe":
            f0 = self.f0_model.get_f0(x, x.shape[0] // self.window, filter_radius=0.006)

        # Precise F0 mapping (CDF / quantile distribution match): remaps the
        # carrier's ESTIMATED F0 onto a target voice's distribution before the
        # model's pitch guidance; output samples are untouched (faithful-carrier).
        # When set, the engine also passes f0_up_key=0 / autotune=False /
        # proposed=False so this is the sole F0 transform.
        if self.f0_remap is not None:
            f0 = self.f0_remap(f0)

        if f0_autotune is True:
            f0 = self.autotune.autotune_f0(f0, f0_autotune_strength)
        elif proposed_pitch is True:
            # Derive a transpose that recenters the median F0 on the threshold.
            limit = 12
            valid_f0 = np.where(f0 > 0)[0]
            if len(valid_f0) < 2:
                up_key = 0
            else:
                median_f0 = float(
                    np.median(np.interp(np.arange(len(f0)), valid_f0, f0[valid_f0]))
                )
                if median_f0 <= 0 or np.isnan(median_f0):
                    up_key = 0
                else:
                    up_key = max(
                        -limit,
                        min(
                            limit,
                            int(
                                np.round(
                                    12 * np.log2(proposed_pitch_threshold / median_f0)
                                )
                            ),
                        ),
                    )
            f0 *= pow(2, (f0_up_key + up_key) / 12)
        else:
            f0 *= pow(2, f0_up_key / 12)

        f0 = torch.from_numpy(f0).to(self.device).float()

        # Quantize to 255 mel-spaced buckets for the coarse pitch embedding.
        f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
        f0_mel = torch.clip(
            (f0_mel - self.f0_min) * 254 / (self.f0_max - self.f0_min) + 1,
            1,
            255,
            out=f0_mel,
        )
        f0_coarse = torch.round(f0_mel, out=f0_mel).long()

        return f0_coarse, f0

    def voice_conversion(
        self,
        audio: Tensor,
        pitch: Tensor = None,
        pitchf: Tensor = None,
        f0_up_key: float = 0,
        index_rate: float = 0.0,
        p_len: int = 0,
        skip_head: int = None,
        return_length: int = None,
        protect: float = 0.5,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        block_size_16k: int = None,
    ):
        """Convert the persistent 16 kHz convert buffer; decode only the new
        region (rate head-trim) and return model-rate audio for it.

        ``pitch``/``pitchf`` are the engine's persistent F0 caches — shifted by
        one block here and refreshed from the newest estimation window."""
        with torch.no_grad():
            assert audio.dim() == 1, audio.dim()
            feats = audio.view(1, -1).to(self.device)

            if self.use_f0:
                # Estimate F0 from the most recent audio window only.
                shift = (block_size_16k or skip_head * self.window) // self.window
                f0_frame = (
                    block_size_16k + 800
                    if block_size_16k
                    else skip_head * self.window + 800
                )
                if self.f0_method == "rmvpe":
                    # rmvpe pads to its 5120-sample chunking internally.
                    f0_frame = 5120 * ((f0_frame - 1) // 5120 + 1) - 160
                f0_frame = min(f0_frame, audio.shape[0])

                f0_coarse_new, f0_new = self.get_f0(
                    audio[-f0_frame:],
                    f0_up_key,
                    f0_autotune,
                    f0_autotune_strength,
                    proposed_pitch,
                    proposed_pitch_threshold,
                )

                # Shift the pitch caches left by one block, then append the new
                # frames trimmed [3:-1] (boundary frames are unreliable).
                if shift > 0:
                    pitch[:-shift] = pitch[shift:].clone()
                    pitchf[:-shift] = pitchf[shift:].clone()
                interior_coarse = (
                    f0_coarse_new[3:-1] if f0_coarse_new.shape[0] > 4 else f0_coarse_new
                )
                interior_f = f0_new[3:-1] if f0_new.shape[0] > 4 else f0_new
                pitch[-interior_coarse.shape[0] :] = interior_coarse
                pitchf[-interior_f.shape[0] :] = interior_f
            else:
                pitch, pitchf = None, None

            # HuBERT features; v2 consumes the 768-dim last_hidden_state as-is.
            feats = self.hubert_model(feats)["last_hidden_state"]
            feats = (
                self.hubert_model.final_proj(feats[0]).unsqueeze(0)
                if self.version == "v1"
                else feats
            )

            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            # Copy kept for pitch-guidance protection blending.
            feats0 = feats.detach().clone() if self.use_f0 else None

            try:
                if self.index and index_rate > 0:
                    feats = self._retrieve_speaker_embeddings(
                        skip_head, feats, self.index, self.big_npy, index_rate
                    )
            except AssertionError:
                print("The index file structure is incompatible with the model.")
                self.index = self.big_npy = None

            # Feature upsampling: HuBERT hop 20 ms -> model frame 10 ms.
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )[:, :p_len, :]

            if self.use_f0:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )[:, :p_len, :]
                pitch_p = pitch[-p_len:].unsqueeze(0)
                pitchf_p = pitchf[-p_len:].unsqueeze(0)

                # Voiceless-consonant protection: blend retrieved features back
                # toward the clean ones on unvoiced frames. protect=0.5 = off.
                if protect < 0.5:
                    pitchff = pitchf_p.detach().clone()
                    pitchff[pitchf_p > 0] = 1
                    pitchff[pitchf_p < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                        1 - pitchff.unsqueeze(-1)
                    )
                    feats = feats.to(feats0.dtype)
            else:
                pitch_p, pitchf_p = None, None

            pitchf_p = pitchf_p.to(self.dtype) if self.use_f0 else None
            # rate = return_length / p_len: trim the oldest context so the model
            # decodes only the current block.
            self._rate_tensor.fill_(return_length / p_len)
            self._p_len_tensor.fill_(p_len)
            out_audio = self.vc.inference(
                feats,
                self._p_len_tensor,
                self.torch_sid,
                pitch_p,
                pitchf_p,
                self._rate_tensor,
            ).float()

        return out_audio

    def _retrieve_speaker_embeddings(
        self, skip_head, feats, index, big_npy, index_rate
    ):
        """FAISS k=8 inverse-square-distance retrieval blended at index_rate."""
        skip_offset = skip_head // 2
        npy = feats[0][skip_offset:].cpu().numpy()
        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        feats[0][skip_offset:] = (
            torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
            + (1 - index_rate) * feats[0][skip_offset:]
        )
        return feats


def create_pipeline(
    model_path: str,
    index_path: str,
    f0_method: str,
    embedder_dir: str,
    sid: int,
    device: str,
    predictor_dir: str,
) -> RealtimePipeline:
    """Build the full realtime pipeline from explicit paths + device."""
    vc = RealtimeVoiceConverter(model_path, device)
    index, big_npy = load_faiss_index(normalize_index_path(index_path))

    hubert_model = load_embedding(embedder_dir)
    hubert_model = hubert_model.to(device).to(vc.dtype)
    hubert_model.eval()

    return RealtimePipeline(
        vc,
        hubert_model,
        index,
        big_npy,
        f0_method,
        sid,
        predictor_dir,
    )
