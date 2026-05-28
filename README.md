# Tvoice / RVC — Python Realtime Voice Changer

Python rebuild of a realtime RVC (Retrieval-based Voice Conversion) voice
changer. Target route:

```
physical microphone
  -> Python audio pipeline
  -> RVC chunked inference (the model defines the voice)
  -> CABLE Input  (the app renders here)
  -> CABLE Output (Discord / OBS / Zoom select this as "microphone")
```

## Design stance: model-faithful runtime

**The trained model is the voice.** A model bundle is a `.pth` + its
`.index` + the supporting `hubert_base.pt` + `rmvpe.pt`, plus the
inference parameters those files were trained against. The runtime's
job is to load that bundle and play it through the realtime audio chain
**faithfully, stably, and safely**.

The runtime does **not** position itself as a sound-design tool:

- `pitch_shift`, `index_rate`, `protect`, `filter_radius`, `rms_mix_rate`,
  `f0_method` are **model profile parameters** — properties of the
  trained model, not user-facing tuning knobs. They live inside the
  model profile JSON.
- The CLI flags that override them exist for developer / debug purposes
  only and print a "developer override" warning when used.
- To get a different voice, train (or load) a different model.

## Model profile

Voice identity is bundled in a JSON file under `config/model_profiles/`.
The shipped example is [`config/model_profiles/kiki.example.json`](config/model_profiles/kiki.example.json):

```json
{
  "name": "kiki",
  "model_path":  "models/kiki/kikiV1.pth",
  "index_path":  "models/kiki/kikiV1.index",
  "hubert_path": "models/kiki/hubert_base.pt",
  "rmvpe_path":  "models/kiki/rmvpe.pt",
  "f0_method":   "rmvpe",
  "index_rate":  0.5,
  "protect":     0.33,
  "filter_radius": 3,
  "rms_mix_rate":  0.25,
  "pitch_shift":   0,
  "resample_sr":   0,
  "notes":         "Example profile for the kiki model. These values are the model's intended inference settings, not sound-design knobs."
}
```

Paths inside the profile are interpreted relative to the working
directory you run the command from (normally the project root).

Place the actual model assets under `models/` (gitignored end-to-end;
`*.pth`, `*.index`, `*.pt` files are never committed).

## Runtime engineering knobs (the *only* normal-user knobs)

These shape stability, latency, and correctness — not voice character:

| Flag | Default | What it controls |
| --- | --- | --- |
| `--device` | `auto` | Inference device (`auto` / `cuda` / `cpu`) |
| `--chunk-ms` | 1000 | RVC chunk size — accumulation latency |
| `--crossfade-ms` | 20 | Chunk-boundary smoothing |
| `--rvc-queue-ms` | 6000 | Per-direction queue capacity |
| `--rvc-prebuffer-ms` | `2 × chunk_ms` | Startup silence inserted before first real audio |
| `--warmup-rvc-count` | 2 | Dummy inferences run before opening audio stream |
| `--drop-stale-input` / `--no-drop-stale-input` | on | If inference falls behind, drop older chunks instead of growing latency |
| `--duration-seconds` | (none — run until Ctrl+C) | Stop after N seconds |
| `--input-device-substring` | from config | Mic device name fragment |
| `--output-device-substring` | from config | Output device name fragment (must be `CABLE Input`) |

## VB-CABLE routing rules

- **The app renders (outputs) to `CABLE Input`.**
- **Discord / OBS / Zoom select `CABLE Output` as their microphone.**
- **The app's input device is a physical microphone.** The runtime
  refuses `CABLE Output` as input unless the explicit
  `--allow-virtual-cable-input` diagnostic override is set.

## Stage 1 commands (still relevant)

```
.\.venv310\Scripts\python.exe -m pip install -r requirements.txt
.\.venv310\Scripts\python.exe -m src.main --list-devices
.\.venv310\Scripts\python.exe -m src.main --mode identity --config config/runtime.example.json --duration-seconds 30
.\.venv310\Scripts\python.exe -m tools.verify_cable_route --duration-seconds 2
.\.venv310\Scripts\python.exe -m tools.click_test --duration-seconds 2 --pulse-amplitude 0.5
```

## Stage 2 — offline RVC sanity (recommended invocation)

```
.\.venv310\Scripts\python.exe -m tools.offline_infer `
    --input-wav test.wav `
    --output-wav test_kiki_rvc.wav `
    --model-profile config/model_profiles/kiki.example.json `
    --device cuda
```

The profile supplies the voice. The CLI supplies only file I/O and
device. No tuning knobs appear in this command.

## Stage 2 — realtime RVC (recommended invocation)

```
.\.venv310\Scripts\python.exe -m src.main --mode rvc `
    --config config/runtime.example.json `
    --model-profile config/model_profiles/kiki.example.json `
    --input-device-substring "WO Mic" `
    --output-device-substring "CABLE Input" `
    --device cuda `
    --chunk-ms 1000 `
    --crossfade-ms 20 `
    --rvc-queue-ms 6000 `
    --rvc-prebuffer-ms 3000 `
    --warmup-rvc-count 2 `
    --duration-seconds 60
```

Discord / OBS / your recorder mic must be `CABLE Output (VB-Audio
Virtual Cable)`.

## Developer override flags (debug only)

The runtime accepts the following flags so developers can diff against
the model profile, but they are **not** part of normal usage:

```
--model-path  --index-path  --hubert-path  --rmvpe-path
--f0-method   --index-rate  --protect      --filter-radius
--rms-mix-rate  --pitch-shift
```

If any of these are set AND differ from the profile, the runtime prints:

```
WARNING: developer override: profile <field>=<old> replaced by CLI
--<field> = <new>. Model-faithful behaviour is to omit this flag.
```

Use them only when investigating a specific model behaviour. They never
appear in product-facing commands or recommendations.

## Identity-first safety net

Stage 2's RVC worker still falls back to identity passthrough on:

- backend exceptions (CUDA OOM, RMVPE NaN, model load failure)
- inference returning invalid output (zero-length, invalid sample rate)

The audio link stays alive in these cases. Observability via
`fallback_count`, `rvc_fallback_count`, `nan_inf_scrub_count`.

## Metrics

Per-second print line + final summary cover:

- frames in / out, queue depths, queue drops
- input/output peak/RMS dBFS, status flags
- `rvc_chunks_processed`, `rvc_inference_mean_ms`, `rvc_inference_median_ms`,
  `rvc_inference_p95_ms`, `rvc_inference_max_ms`
- `rvc_output_blocks_enqueued` / `rvc_output_blocks_dropped`
- `rvc_stale_chunk_drops`
- `rvc_resample_count` / `rvc_resample_mean_ms`
- `startup_output_underruns` vs `steady_state_output_underruns`
- `max_input_queue_depth` / `max_output_queue_depth`
- session info (model_basename, chunk_ms, crossfade_ms, etc.)

## Project principles

- **Identity-first.** RVC mode falls back to identity on any chunk
  inference error. The user hears themselves rather than silence or a
  crash.
- **Model-faithful.** Voice identity comes from the trained model
  bundle; the runtime plays it back as trained, with no "tone tuning"
  product surface.
- **No system mutation.** The app never changes the Windows default
  audio device, never edits Discord / OBS / Zoom settings, never
  touches the registry, PATH, drivers, or any global config.
- **No generated artifact pollution.** Audio captures, sidecar JSON,
  reports, and model files are not committed. `.gitignore` covers
  `*.wav`, `*.jsonl`, `*.log`, `*.pt`, `*.pth`, `*.index`,
  `models/local/`, `eval_corpus/reports/` recursively. Validation tools
  never write artifacts unless `--save --report-dir PATH` is supplied.
- **Async queue architecture.** Audio callbacks never block; the worker
  thread owns RVC inference.

## Testing

```
.\.venv310\Scripts\python.exe -m pytest -q
```

Pure tests only: no audio hardware, no GPU, no internet, no model
files, no Discord. The full RVC stack is mocked via fake backends and
fake engines so the tests run on any machine with numpy + pytest.

## Dependencies

`requirements.txt` lists only `numpy`, `sounddevice`, `pytest`. The
RVC stack (`infer-rvc-python`, `torch`, `torchaudio`, `faiss-cpu`,
`fairseq`, `pyworld`, etc.) is **not** installed by the project — you
install it manually into `.venv310` once. See the install notes in the
project plan.
