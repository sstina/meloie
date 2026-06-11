# Derived from Applio (https://github.com/IAHispano/Applio), MIT License,
# (c) 2026 AI Hispano. See meloie/core/NOTICE.md. Inference-only: the training
# forward(), the posterior encoder (enc_q), remove_weight_norm plumbing and the
# randomized/checkpointing training knobs were removed. Checkpoints load with
# strict=False, so their enc_q.* keys are simply ignored.
"""The RVC v2 Synthesizer (net_g): text encoder + flow + speaker embedding + decoder.

The constructor consumes ``*cpt["config"]`` POSITIONALLY — the parameter order
is checkpoint contract; do not reorder. enc_p / dec / flow / emb_g are
state_dict key prefixes — do not rename. ``infer(rate=)`` trims the oldest
context columns so only the new region is decoded (the realtime trick).
"""

from typing import Optional

import torch

from .encoders import TextEncoder
from .hifigan import HiFiGANGenerator
from .hifigan_mrf import HiFiGANMRFGenerator
from .hifigan_nsf import HiFiGANNSFGenerator
from .refinegan import RefineGANGenerator
from .residuals import ResidualCouplingBlock


class Synthesizer(torch.nn.Module):
    """Builds the decoder matching the checkpoint's vocoder tag:
    'HiFi-GAN' (default; NSF variant when use_f0) / 'MRF HiFi-GAN' / 'RefineGAN'."""

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool,
        text_enc_hidden_dim: int = 768,
        vocoder: str = "HiFi-GAN",
        **kwargs,
    ):
        super().__init__()
        self.use_f0 = use_f0

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
            f0=use_f0,
        )
        if use_f0:
            if vocoder == "MRF HiFi-GAN":
                self.dec = HiFiGANMRFGenerator(
                    in_channel=inter_channels,
                    upsample_initial_channel=upsample_initial_channel,
                    upsample_rates=upsample_rates,
                    upsample_kernel_sizes=upsample_kernel_sizes,
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilations=resblock_dilation_sizes,
                    gin_channels=gin_channels,
                    sample_rate=sr,
                    harmonic_num=8,
                )
            elif vocoder == "RefineGAN":
                self.dec = RefineGANGenerator(
                    sample_rate=sr,
                    downsample_rates=upsample_rates[::-1],
                    upsample_rates=upsample_rates,
                    start_channels=16,
                    num_mels=inter_channels,
                )
            else:
                self.dec = HiFiGANNSFGenerator(
                    inter_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes,
                    upsample_rates,
                    upsample_initial_channel,
                    upsample_kernel_sizes,
                    gin_channels=gin_channels,
                    sr=sr,
                )
        else:
            if vocoder in ("MRF HiFi-GAN", "RefineGAN"):
                raise ValueError(f"{vocoder} requires pitch guidance (f0 models only)")
            self.dec = HiFiGANGenerator(
                inter_channels,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
            )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
    ):
        """phone: HuBERT features; pitch: coarse F0; nsff0: F0 in Hz; sid: speaker.
        rate = return_length / p_len: decode only the trailing fraction."""
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p, x_mask = z_p[:, :, head:], x_mask[:, :, head:]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head:]

        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = (
            self.dec(z * x_mask, nsff0, g=g)
            if self.use_f0
            else self.dec(z * x_mask, g=g)
        )

        return o, x_mask, (z, z_p, m_p, logs_p)
