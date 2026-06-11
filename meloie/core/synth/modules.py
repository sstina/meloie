# Derived from Applio (https://github.com/IAHispano/Applio), MIT License,
# (c) 2026 AI Hispano. See meloie/core/NOTICE.md.
"""WaveNet residual stack (used by the flow's coupling layers every block).

cond_layer / in_layers / res_skip_layers are state_dict keys — keep the names.
weight_norm uses torch.nn.utils.parametrizations (its registered load hook maps
legacy weight_g/weight_v checkpoint keys onto parametrizations.weight.original*).
"""

import torch

from .commons import fused_add_tanh_sigmoid_multiply


class WaveNet(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: int = 0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for proper padding."

        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        # IntTensor shape required by the jit-scripted fused gate's signature.
        self.n_channels_tensor = torch.IntTensor([hidden_channels])

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)

        if gin_channels:
            self.cond_layer = torch.nn.utils.parametrizations.weight_norm(
                torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1),
                name="weight",
            )

        dilations = [dilation_rate**i for i in range(n_layers)]
        paddings = [(kernel_size * d - d) // 2 for d in dilations]

        for i in range(n_layers):
            self.in_layers.append(
                torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(
                        hidden_channels,
                        2 * hidden_channels,
                        kernel_size,
                        dilation=dilations[i],
                        padding=paddings[i],
                    ),
                    name="weight",
                )
            )

            res_skip_channels = (
                hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            )
            self.res_skip_layers.append(
                torch.nn.utils.parametrizations.weight_norm(
                    torch.nn.Conv1d(hidden_channels, res_skip_channels, 1),
                    name="weight",
                )
            )

    def forward(self, x, x_mask, g=None):
        output = x.clone().zero_()

        g = self.cond_layer(g) if g is not None else None

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = (
                g[
                    :,
                    i * 2 * self.hidden_channels : (i + 1) * 2 * self.hidden_channels,
                    :,
                ]
                if g is not None
                else 0
            )

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return output * x_mask
