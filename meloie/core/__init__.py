"""Meloie inference core — the internalized RVC v2 realtime stack.

Derived from Applio (MIT, (c) 2026 AI Hispano) and internalized for Meloie:
see NOTICE.md in this package for provenance and the full license text.

Layout:
  pipeline.py    — RealtimeVoiceConverter / RealtimePipeline / create_pipeline /
                   load_faiss_index / strip_parametrizations / circular_write / Autotune
  embedder.py    — HubertModelWithFinalProj + load_embedding (explicit dir, no downloads)
  predictors/    — RMVPE + FCPE F0 estimators (explicit weight paths)
  synth/         — the v2 Synthesizer model graph (state_dict-compatible with RVC .pth)

Everything here is torch-heavy; import lazily (the engine imports inside load()).
This package never touches the output side beyond the model's own samples — the
faithful-carrier contract is baked into the API (no volume envelope / FX / output
denoise parameters exist at all).
"""
