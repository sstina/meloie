# Derived from Applio (https://github.com/IAHispano/Applio), MIT License,
# (c) 2026 AI Hispano. See meloie/core/NOTICE.md. PosteriorEncoder (enc_q,
# training-only) was removed together with Synthesizer.enc_q.
"""Transformer text encoder over HuBERT features (+ optional coarse pitch).

emb_phone / emb_pitch / encoder.{attn_layers,norm_layers_1,ffn_layers,
norm_layers_2} / proj are state_dict keys — keep the attribute names.
"""

import math
from typing import Optional

import torch

from .attentions import FFN, MultiHeadAttention
from .commons import sequence_mask
from .normalization import LayerNorm


class Encoder(torch.nn.Module):
    """Stack of relative-attention + FFN layers (window_size=10)."""

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.drop = torch.nn.Dropout(p_dropout)

        self.attn_layers = torch.nn.ModuleList(
            [
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_layers_1 = torch.nn.ModuleList(
            [LayerNorm(hidden_channels) for _ in range(n_layers)]
        )
        self.ffn_layers = torch.nn.ModuleList(
            [
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_layers_2 = torch.nn.ModuleList(
            [LayerNorm(hidden_channels) for _ in range(n_layers)]
        )

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)

        return x * x_mask


class TextEncoder(torch.nn.Module):
    """Projects HuBERT features (v2: 768-dim) + coarse pitch into (m, logs)."""

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        embedding_dim: int,
        f0: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.emb_pitch = torch.nn.Embedding(256, hidden_channels) if f0 else None

        self.encoder = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, phone: torch.Tensor, pitch: Optional[torch.Tensor], lengths: torch.Tensor
    ):
        x = self.emb_phone(phone)
        if pitch is not None and self.emb_pitch:
            x += self.emb_pitch(pitch)

        x *= math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = x.transpose(1, -1)  # [B, H, T]

        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask
