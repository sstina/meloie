# Derived from Applio (https://github.com/IAHispano/Applio), MIT License,
# (c) 2026 AI Hispano. See meloie/core/NOTICE.md. Training-only helpers
# (slice_segments / rand_slice_segments / grad_norm) were dropped.
"""Small shared ops for the synthesizer graph."""

from typing import Optional

import torch


def init_weights(m, mean=0.0, std=0.01):
    """Normal-init conv weights (pre-load only; overwritten by the checkpoint)."""
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    """[[a,b],[c,d],...] -> F.pad's reversed flat form [...,c,d,a,b]."""
    return [item for sublist in pad_shape[::-1] for item in sublist]


# NOTE (frozen builds): @torch.jit.script compiles via inspect.getsource at
# import time, so the frozen bundle must ship this module's .py on disk at its
# import path (meloie.spec datas) or the exe dies at import.
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """tanh((a+b)[:n]) * sigmoid((a+b)[n:]) — the WaveNet gate, fused for speed."""
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
