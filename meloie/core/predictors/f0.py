# Derived from Applio (https://github.com/IAHispano/Applio), MIT License,
# (c) 2026 AI Hispano. See meloie/core/NOTICE.md. Internalized: explicit weight
# paths (no CWD-relative resolution), CREPE + torchcrepe dropped (the realtime
# engine backs only rmvpe/fcpe).
"""RMVPE / FCPE wrapper classes with a uniform get_f0 surface.

Both estimate F0 in Hz on 16 kHz mono float32 at hop 160 (one frame / 10 ms).
``filter_radius`` is the salience/decoder threshold (NOT a smoothing radius):
rmvpe uses ~0.03, fcpe ~0.006.
"""

import os

import torch
from torchfcpe import spawn_infer_model_from_pt

from .rmvpe import RMVPE0Predictor


class RMVPE:
    def __init__(self, weight_path: str, device, sample_rate=16000, hop_size=160):
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"RMVPE weight not found: {weight_path}")
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = RMVPE0Predictor(weight_path, device=device)

    def get_f0(self, x, filter_radius=0.03):
        return self.model.infer_from_audio(x, thred=filter_radius)


class FCPE:
    def __init__(self, weight_path: str, device, sample_rate=16000, hop_size=160):
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"FCPE weight not found: {weight_path}")
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = spawn_infer_model_from_pt(weight_path, device, bundled_model=True)

    def get_f0(self, x, p_len=None, filter_radius=0.006):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        f0 = (
            self.model.infer(
                x.float().to(self.device).unsqueeze(0),
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=filter_radius,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return f0
