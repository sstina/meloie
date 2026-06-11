# Derived from Applio (https://github.com/IAHispano/Applio), MIT License,
# (c) 2026 AI Hispano. See meloie/core/NOTICE.md.
"""Channel-wise LayerNorm with gamma/beta parameter names.

The custom parameter names (gamma/beta, NOT torch.nn.LayerNorm's weight/bias)
are state_dict keys (enc_p.encoder.norm_layers_*.{gamma,beta}) — keep them.
"""

import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # (B, C, T) -> (B, T, C) for layer_norm, then back
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(
            x, (x.size(-1),), self.gamma, self.beta, self.eps
        )
        return x.transpose(1, -1)
