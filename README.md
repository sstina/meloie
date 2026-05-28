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

**Stage 1 — identity realtime streaming + validation tools.**

- Realtime identity streaming is implemented: mic -> Python -> CABLE Input.
- Click-test latency measurement is implemented.
- Cable-route verification (Python -> CABLE Input -> CABLE Output non-silent)
  is implemented.
- RVC inference is **Stage 2** and is intentionally not implemented.
  All RVC code paths raise `NotImplementedError` with the exact message
  `"RVC inference is Stage 2 and is not implemented in this skeleton."`.
- No torch / infer_rvc_python / rvc-python imports.
- No models are downloaded or loaded.
- No system, device, Discord, OBS, or Zoom settings are touched.

## VB-CABLE routing rules

These rules are inherited from the legacy project and must not be inverted:

- **The app renders (outputs) to `CABLE Input`.** That is the virtual
  cable's render endpoint.
- **Discord / OBS / Zoom select `CABLE Output` as their microphone.** That
  is the virtual cable's capture endpoint.
- **The app's input device must be a real, physical microphone.** Using
  `CABLE Output` as the app input is refused by the runtime guard.
  The two diagnostic tools (`tools/click_test.py`,
  `tools/verify_cable_route.py`) bypass that guard on purpose — capturing
  from `CABLE Output` is the loopback they are designed to measure.

## Project principles

- **Identity-first.** Every stage must prove a pure identity passthrough
  works before any processing is layered on. The Stage 1 identity worker
  is the safety floor that all later stages fall back to on error.
- **No system mutation.** The app never changes the Windows default audio
  device, never edits Discord / OBS / Zoom settings, never touches the
  registry, PATH, drivers, or any global config.
- **No generated artifact pollution.** Audio captures, sidecar JSON,
  reports, and model files are not committed. The `.gitignore` covers
  `*.wav`, `*.jsonl`, `*.log`, `*.pt`, `*.pth`, `*.index`, `models/local/`,
  and `eval_corpus/reports/` recursively. The validation tools NEVER
  write artifacts unless you explicitly pass `--save --report-dir PATH`.
- **Async queue architecture.** Audio callbacks never block; all
  per-block work runs on a worker thread connected by bounded queues.

## Stage 1 validation commands

Install deps once (skip if already installed):

```
pip install -r requirements.txt
```

### 1. List audio devices (always safe — read-only enumeration)

```
python -m src.main --list-devices
```

### 2. 30-second identity smoke (mic -> Python -> CABLE Input)

```
python -m src.main --mode identity --config config/runtime.example.json \
    --duration-seconds 30
```

Override device substrings if needed (Windows often labels mics with
Chinese text or "WO Mic" etc.):

```
python -m src.main --mode identity --config config/runtime.example.json \
    --input-device-substring "WO Mic" \
    --output-device-substring "CABLE Input" \
    --duration-seconds 30
```

### 3. 5-minute identity run

```
python -m src.main --mode identity --config config/runtime.example.json \
    --duration-seconds 300
```

The CLI prints periodic metrics and a final summary. Acceptance gates
(see `CLAUDE.md` §8): `output_underruns <= 1` (startup only),
`*_queue_drops = 0`, `fallback_count = 0`.

### 4. Verify the cable route is alive

Renders a 440 Hz tone to `CABLE Input` and captures from `CABLE Output`;
exits non-zero if the capture is silent:

```
python -m tools.verify_cable_route \
    --output-device-substring "CABLE Input" \
    --input-device-substring "CABLE Output" \
    --duration-seconds 2
```

### 5. Measure identity-path click latency

Renders a short triangular click to `CABLE Input`, captures from
`CABLE Output`, cross-correlates, prints `latency_ms`:

```
python -m tools.click_test \
    --output-device-substring "CABLE Input" \
    --input-device-substring "CABLE Output" \
    --duration-seconds 2 \
    --pulse-amplitude 0.5
```

### 6. Save a report (opt-in only)

Add `--save --report-dir eval_corpus/reports/python_identity/` to either
tool to write a `.wav` capture + sidecar `.json` to the gitignored
report directory. Without these flags, the tools write nothing.

## Testing

```
python -m pytest
```

Tests are pure: no audio hardware, no GPU, no internet, no model files,
no Discord. Tests never open audio streams. The trip-wire test in
`tests/test_tools_import_safety.py` enforces that `tools.click_test`
and `tools.verify_cable_route` do not import `sounddevice` at module load.

## Dependencies

See `requirements.txt`. Intentionally minimal: `numpy`, `sounddevice`,
`pytest`. Torch / RVC / faiss / CUDA / training / TTS / GUI dependencies
all belong to Stage 2 or later and are not installed here.
