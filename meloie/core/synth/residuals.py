# Derived from Applio (https://github.com/IAHispano/Applio), MIT License,
# (c) 2026 AI Hispano. See meloie/core/NOTICE.md. remove_weight_norm plumbing
# removed (strip_parametrizations supersedes it after checkpoint load).
"""Residual blocks + the normalizing-flow coupling stack.

ModuleList layout is a state_dict contract: flow.flows alternates coupling
(even indices, with pre/enc/post) and Flip (odd, parameterless) — keep the
ordering and names. resblocks.{convs1,convs2} likewise.
"""

from typing import Optional, Tuple

import torch
from torch.nn.utils.parametrizations import weight_norm

from .commons import get_padding, init_weights
from .modules import WaveNet

LRELU_SLOPE = 0.1


def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(
        torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            1,
            dilation=dilation,
            padding=get_padding(kernel_size, dilation),
        )
    )


def apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]):
    return tensor * mask if mask else tensor


class ResBlock(torch.nn.Module):
    """Dilated conv residual block (the HiFi-GAN MRF cell)."""

    def __init__(
        self, channels: int, kernel_size: int = 3, dilations: Tuple[int] = (1, 3, 5)
    ):
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        layers = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        layers.apply(init_weights)
        return layers

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            x_residual = x
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            x = apply_mask(x, x_mask)
            x = torch.nn.functional.leaky_relu(conv1(x), LRELU_SLOPE)
            x = apply_mask(x, x_mask)
            x = conv2(x)
            x = x + x_residual
        return apply_mask(x, x_mask)


class Flip(torch.nn.Module):
    """Channel flip between coupling layers (parameterless flow step)."""

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet
        return x


class ResidualCouplingBlock(torch.nn.Module):
    """The flow: n_flows x (coupling layer + Flip). Inference runs it reversed."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.n_flows = n_flows

        self.flows = torch.nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow.forward(x, x_mask, g=g, reverse=reverse)
        return x


class ResidualCouplingLayer(torch.nn.Module):
    """Affine (mean-only) coupling layer over half the channels."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = torch.nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = torch.nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1
        )
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x
