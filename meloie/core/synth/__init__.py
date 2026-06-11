"""The v2 RVC Synthesizer model graph (state_dict-compatible with RVC .pth).

Derived from Applio (MIT) — see meloie/core/NOTICE.md. Inference-only: the
training forward paths, posterior encoder (enc_q) and remove_weight_norm
plumbing were removed; `strip_parametrizations` (pipeline.py) replaces the
latter after checkpoint load. Attribute names are frozen by the checkpoint
format — do not rename module attributes here.
"""
