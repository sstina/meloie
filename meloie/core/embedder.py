# Derived from Applio (https://github.com/IAHispano/Applio), MIT License,
# (c) 2026 AI Hispano. See meloie/core/NOTICE.md. Internalized: explicit
# embedder directory (no CWD capture), download fallback removed (the app is
# offline by contract — a missing embedder fails loudly instead of wget'ing),
# and no process-wide warnings/logging mutation (only the transformers logger
# is quieted, scoped and deliberate).
"""ContentVec/HuBERT embedder loading."""

import logging
import os

from torch import nn
from transformers import HubertModel

# from_pretrained is chatty about unused final_proj weights on the v2 path;
# quiet ONLY the transformers logger (not the process).
logging.getLogger("transformers").setLevel(logging.ERROR)


class HubertModelWithFinalProj(HubertModel):
    """HubertModel + the final_proj head present in contentvec checkpoints.

    v2 models consume the 768-dim last_hidden_state directly and never call
    final_proj, but the attribute must exist for from_pretrained's key match.
    """

    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def load_embedding(embedder_dir: str) -> HubertModelWithFinalProj:
    """Load an embedder from an explicit local directory (pytorch_model.bin +
    config.json). Raises FileNotFoundError instead of downloading anything."""
    bin_file = os.path.join(embedder_dir, "pytorch_model.bin")
    json_file = os.path.join(embedder_dir, "config.json")
    missing = [p for p in (bin_file, json_file) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"embedder is not fully staged ({', '.join(missing)}); "
            "place pytorch_model.bin + config.json there first."
        )
    return HubertModelWithFinalProj.from_pretrained(embedder_dir)
