# Tvoice / RVC — Python Realtime Voice Changer

Python rebuild of a realtime RVC (Retrieval-based Voice Conversion) voice
changer. Long-term target route:

```
physical microphone
  -> Python audio pipeline
  -> (Stage 2+) RVC conversion
  -> CABLE Input  (the app renders here)
  -> CABLE Output (Discord / OBS / Zoom select this as "microphone")
```

## Current stage

**Stage 0 / Stage 1 skeleton.** Scaffolding only.

- No realtime audio streams are opened anywhere in this code yet.
- No RVC inference is implemented.
- No torch / infer_rvc_python / rvc-python imports.
- No models are downloaded or loaded.
- No system, device, Discord, OBS, or Zoom settings are touched.

Realtime identity streaming (`mic -> python -> CABLE Input`) is the Stage 1
goal but is intentionally left as a guarded placeholder in this skeleton.
RVC inference is Stage 2 and is explicitly not implemented.

## VB-CABLE routing rules

These rules are inherited from the legacy project and must not be inverted:

- **The app renders (outputs) to `CABLE Input`.** That is the virtual
  cable's render endpoint.
- **Discord / OBS / Zoom select `CABLE Output` as their microphone.** That
  is the virtual cable's capture endpoint.
- **The app's input device must be a real, physical microphone.** Using
  `CABLE Output` as the app input creates a feedback loop and is rejected
  by the device helper.

## Project principles

- **Identity-first.** Every stage must prove a pure identity passthrough
  works before any processing is layered on. The Stage 1 identity worker
  is the safety floor that all later stages fall back to on error.
- **No system mutation.** The app never changes the Windows default audio
  device, never edits Discord / OBS / Zoom settings, never touches the
  registry, PATH, drivers, or any global config.
- **No generated artifact pollution.** Audio captures, sidecar JSON,
  reports, and model files are not committed. The `.gitignore` enforces
  this; do not bypass it.
- **Async queue architecture.** Real audio callbacks must never block.
  All heavy work runs on a worker thread connected by bounded queues.
  The skeleton already reflects this shape; only the worker function
  body changes between stages.

## Future commands (not yet wired)

Once realtime streaming is implemented:

```
python -m src.main --list-devices
python -m src.main --check-config config/runtime.example.json
python -m src.main --mode identity    # Stage 1 realtime identity
```

Currently `--list-devices` and `--check-config` work; `--mode identity`
prints a notice that realtime streaming is not implemented in this
skeleton.

## Testing

```
python -m pytest
```

Tests are pure: no audio hardware, no GPU, no internet, no model files,
no Discord. Tests must not open audio streams.

## Dependencies

See `requirements.txt`. Intentionally minimal: `numpy`, `sounddevice`,
`pytest`. Torch / RVC / faiss / CUDA / training / TTS / GUI dependencies
all belong to Stage 2 or later and are not installed here.
