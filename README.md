# Tvoice / RVC — Python Realtime Voice Changer

Python rebuild of a realtime RVC (Retrieval-based Voice Conversion) voice
changer. Target route:

```
physical microphone
  -> Python audio pipeline
  -> (Stage 2) RVC chunked inference
  -> CABLE Input  (the app renders here)
  -> CABLE Output (Discord / OBS / Zoom select this as "microphone")
```

## Current stage

**Stage 1 — identity realtime streaming + validation tools — DONE.**

**Stage 2 — RVC inference path — IMPLEMENTED.**

- Offline RVC sanity CLI works (`tools/offline_infer.py`): WAV in -> RVC -> WAV out.
- Realtime RVC mode works (`python -m src.main --mode rvc ...`).
- The realtime path is **chunked**: input blocks accumulate to a
  ~180 ms chunk, the RVC worker thread runs inference, the result is
  split back into block-sized pieces for the output callback. The
  audio callbacks themselves never call RVC.
- Identity fallback is automatic: any backend exception (CUDA OOM,
  timeout, NaN, missing index, etc.) makes that chunk pass through
  unchanged. The audio link stays alive.
- No torch / infer_rvc_python imports happen at module load. The
  backend is imported lazily inside `RvcEngine.load()`, so importing
  the codebase is safe even when the backend is not installed.

Stages 3+ (refined crossfade, layered Discord dogfood, training, TTS, GUI)
remain explicitly out of scope.

## VB-CABLE routing rules

Same as Stage 1 — do not invert them:

- **The app renders (outputs) to `CABLE Input`.**
- **Discord / OBS / Zoom select `CABLE Output` as their microphone.**
- **The app's input device is a physical microphone.** The runtime
  guard refuses `CABLE Output` as input unless the explicit
  `--allow-virtual-cable-input` diagnostic override is set.
- The two diagnostic tools (`tools/click_test.py`,
  `tools/verify_cable_route.py`) bypass that guard on purpose to
  measure the loopback.

## Project principles

- **Identity-first.** RVC mode falls back to identity on any chunk
  inference error. The user hears themselves rather than silence or a
  crash. `fallback_count` and `rvc_fallback_count` make these events
  observable in metrics.
- **No system mutation.** No Windows default audio device, Discord /
  OBS / Zoom, registry, PATH, drivers, or global config are touched.
- **No generated artifact pollution.** Audio captures, sidecar JSON,
  reports, and model files are not committed. `.gitignore` covers
  `*.wav`, `*.jsonl`, `*.log`, `*.pt`, `*.pth`, `*.index`,
  `models/local/`, and `eval_corpus/reports/` recursively. The
  validation tools never write artifacts unless you explicitly pass
  `--save --report-dir PATH`.
- **Async queue architecture.** Audio callbacks never block; the
  worker thread owns RVC inference.

## Where to put RVC models

Place local `.pth` and `.index` files under:

```
models/local/
```

That directory is gitignored end-to-end. Never commit model or index
files into the repo — they are large, often license-restricted, and
do not belong in source control.

## Stage 1 validation commands (still relevant)

```
pip install -r requirements.txt
python -m src.main --list-devices
python -m src.main --mode identity --config config/runtime.example.json --duration-seconds 30
python -m tools.verify_cable_route --duration-seconds 2
python -m tools.click_test --duration-seconds 2 --pulse-amplitude 0.5
```

## Stage 2 — offline RVC sanity

Install the backend once (the project itself does not install it for you):

```
pip install infer-rvc-python
```

Then run an end-to-end offline conversion. The output WAV is written
only on a successful inference; failed loads / inference produce a
clear error without touching the output file.

```
python -m tools.offline_infer \
    --input-wav voices/sample.wav \
    --output-wav voices/sample_rvc.wav \
    --model-path models/local/MyVoice.pth \
    --index-path models/local/MyVoice.index \
    --f0-method rmvpe --index-rate 0.5 --protect 0.33 \
    --pitch-shift 0 --filter-radius 3 --rms-mix-rate 0.25
```

## Stage 2 — realtime RVC mode

The realtime command shares the Stage 1 audio loop, queues, devices,
and metrics — only the worker thread changes. The mic input is your
physical microphone; the output is `CABLE Input`; Discord / OBS / Zoom
still select `CABLE Output` to hear the converted voice.

```
python -m src.main --mode rvc \
    --config config/runtime.example.json \
    --model-path models/local/MyVoice.pth \
    --index-path models/local/MyVoice.index \
    --f0-method rmvpe --index-rate 0.5 --protect 0.33 \
    --filter-radius 3 --rms-mix-rate 0.25 --pitch-shift 0 \
    --chunk-ms 180 --crossfade-ms 20 \
    --duration-seconds 60
```

Override mic device substring if needed (e.g. on Chinese Windows):

```
python -m src.main --mode rvc ... \
    --input-device-substring "WO Mic" \
    --output-device-substring "CABLE Input"
```

### Recommended starting params (per legacy dossier + rvc.md)

| Flag             | Value     | Why                                              |
| ---------------- | --------- | ------------------------------------------------ |
| `--f0-method`    | `rmvpe`   | Best F0 quality on most voices                   |
| `--index-rate`   | `0.5`     | Balance source vs target timbre                  |
| `--protect`      | `0.33`    | Protect unvoiced / consonants from over-blending |
| `--filter-radius`| `3`       | Mild F0 median filter                            |
| `--rms-mix-rate` | `0.25`    | Lightly preserve source envelope                 |
| `--pitch-shift`  | `0`       | Tune later per target voice                      |
| `--chunk-ms`     | `180`     | 4080 GPU is fast; chunk dominates latency budget |
| `--crossfade-ms` | `20`      | Soften chunk-boundary clicks                     |

Tune *after* the route works end-to-end. Quality/latency trade-offs
are not part of this milestone — see Stage 3 in `rvc.md`.

## Metrics

Identity mode metrics (Stage 1) + the following RVC-specific fields:

- `rvc_chunks_processed`
- `rvc_inference_count`
- `rvc_inference_mean_ms`, `rvc_inference_max_ms`, `rvc_inference_last_ms`
- `rvc_fallback_count` (chunks routed through identity due to engine error)
- `nan_inf_scrub_count` (samples cleaned after inference)
- `chunk_ms`, `crossfade_ms` (session info)
- `model_basename`, `index_basename`, `f0_method`,
  `index_rate`, `protect`, `pitch_shift` (session info)
- `input_queue_drops`, `output_queue_drops`, `output_underruns`,
  `input_peak_dbfs`, `output_peak_dbfs` (shared with identity mode)

Metrics are printed roughly every second during the run, then again as
a final summary at exit.

## Safety in RVC mode

- The RVC engine runs **only** in the worker thread. Callbacks stay
  non-blocking.
- Any inference exception is caught: the chunk passes through as
  identity, `rvc_fallback_count` increments, and the link stays alive.
- NaN / Inf samples coming out of the backend are scrubbed before
  reaching the output queue.
- `KeyboardInterrupt` cleanly signals the worker via a stop event and
  a sentinel; the audio streams are closed by the `with` blocks.
- If `--crossfade-ms 0`: chunk-boundary clicks are possible. This is a
  known caveat carried forward to Stage 3.

## Testing

```
python -m pytest
```

Tests are pure: no audio hardware, no GPU, no internet, no model files,
no Discord, no `infer_rvc_python` install required. The RVC engine
tests use an injectable fake backend; the worker tests use a fake
engine class. Trip-wire tests in `tests/test_rvc_engine.py` and
`tests/test_tools_import_safety.py` enforce that `src.engine.rvc_engine`,
`tools.offline_infer`, `tools.click_test`, and `tools.verify_cable_route`
do NOT import `torch`, `infer_rvc_python`, or `sounddevice` at module
load.

## Dependencies

`requirements.txt` lists only the *runtime* dependencies that the
project itself manages: `numpy`, `sounddevice`, `pytest`. The RVC
backend (`infer-rvc-python`) and its torch/CUDA stack are installed
**separately by you** — the project never auto-installs them.
