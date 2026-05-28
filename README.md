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
| `--chunk-ms` | 180 | RVC chunk size — accumulation latency. Recommended realtime value is 1000 (longer = more model context per call). |
| `--rvc-context-ms` | **200** | Stage 3 input-side LEFT context. Engine sees `[prev_input_tail, chunk]`; output is trimmed proportionally so emit duration ≈ chunk_ms. Set 0 to A/B. See "Stage 3 — input-side left-context" below. |
| `--crossfade-ms` | **0** | Stitched output-only crossfade. OFF by default — see "What the realtime path does NOT do" below. |
| `--rvc-queue-ms` | 6000 | Per-direction queue capacity |
| `--rvc-prebuffer-ms` | `2 × chunk_ms` | Startup silence inserted before first real audio |
| `--warmup-rvc-count` | 2 | Dummy inferences run before opening audio stream |
| `--drop-stale-input` / `--no-drop-stale-input` | on | If inference falls behind, drop older chunks instead of growing latency |
| `--duration-seconds` | (none — run until Ctrl+C) | Stop after N seconds |
| `--input-device-substring` | from config | Mic device name fragment |
| `--output-device-substring` | from config | Output device name fragment (must be `CABLE Input`) |

## What the realtime path does NOT do

The realtime worker is intentionally a thin carrier for the trained
model. Between `engine.infer_array(...)` returning a chunk and the
audio reaching `CABLE Input` it does only what is required:

- **No EQ, no limiter, no normalizer, no compressor.** The chunk goes
  out at the amplitude the model returned (one defensive int16-scale
  rescale lives in the engine adapter to match `infer_rvc_python`'s
  documented contract; that one is required).
- **No "smoothing" or "warming" of the model output.**
- **Stitched output-side crossfade is OFF by default.** An audit (run
  via `tools/pseudo_stream.py`) showed the previous 20 ms default
  shifted the output timeline by one crossfade length per chunk and
  blended chunk N's tail with chunk N+1's head — two
  temporally-disjoint regions, since the input chunks themselves do
  not overlap. The audit measured no faithfulness improvement vs
  no-crossfade. Re-enable with `--crossfade-ms N` for legacy
  comparison only.
- **Resampling between model-native SR and stream SR uses
  `scipy.signal.resample_poly`** (sinc-windowed polyphase) when scipy
  is importable, falling back to `np.interp` otherwise. The polyphase
  path measured ~+11 dB output SNR vs the linear fallback on
  kiki 40→48 kHz. Both run in well under a millisecond per chunk on
  this hardware.
- **`drop_stale_input` is ON** as a fail-safe: if inference falls
  behind faster than the mic feeds, the worker discards older queued
  chunks and processes the freshest one. In normal steady-state (mean
  inference ~160 ms at chunk_ms=1000 with default context on RTX 4080
  Laptop) this never fires; the `rvc_stale_chunk_drops` metric tells
  you if it does.
- **Identity fallback** runs only when the backend raises or returns
  garbage. The user hears their own voice for that one chunk instead
  of silence or a crash. The `rvc_fallback_count` metric tells you if
  it fires.

## Stage 3 — input-side left-context (the continuity strategy)

Per-chunk inference inherits no past audio across chunk boundaries.
HuBERT, RMVPE (F0), and the index retrieval re-initialise on every
call, so the first ~tens of ms of every chunk's output exhibits cold-
start instability: boundary clicks, F0 wobble, sustained-vowel
"flutter". The previous output-side crossfade did not fix this — it
blended two temporally-disjoint regions, which is geometrically wrong.

The model-faithful fix is to give the model real previous audio as
*input* warmup, then discard the output region that corresponds to
that warmup input. This is exactly what `--rvc-context-ms` does
(default 200 ms):

1. For each chunk N the engine receives `[chunk N−1's tail of
   context_size samples, chunk N]` as one input.
2. The model produces an output of length ~`(context_size +
   chunk_size) * (out_per_in_ratio)`.
3. The worker discards the first `round(context_size * out_len /
   in_len)` samples of model output, then emits the remainder.
4. The context buffer is refreshed to `chunk N[-context_size:]` for
   the next call.

Audit results (`tools/pseudo_stream.py --context-ms 0` vs `200`):

| metric | ctx=0 | ctx=200 (default) |
|---|---|---|
| RMS error vs offline 40k→polyphase 48k | 0.0395 | 0.0366 (−7 %) |
| SNR vs offline reference | +0.94 dB | **+1.60 dB** (+0.66 dB) |
| Cross-corr alignment shift vs offline | −983 samples | **−547 samples** (−44 %) |
| Pairwise SNR ctx=200 vs ctx=0 | — | **+5.31 dB** (materially different) |
| Per-chunk time deficit | 1.96 % | 1.67 % (smaller is better) |
| Steady-state inference time / chunk | 140 ms | 161 ms (+21 ms) |

**Timeline preservation**: emit duration per chunk stays ≈ chunk_ms.
Strictly, it grows slightly (47040 → 47200 samples per 1 s input at
48 kHz) because a longer model input loses proportionally less audio
to edge framing — but this *reduces* the running per-chunk time
deficit, it does not introduce drift. There is NO accumulating
timeline error of the sort the legacy output-side crossfade
introduced (that one shifted by exactly one crossfade length per
chunk and grew unboundedly with session length).

**Latency cost**: zero added to the audio chain. Audio still arrives
at the input callback at the same cadence and is emitted at the same
cadence. Only the worker's per-chunk inference time grows by ~21 ms
(measured), which is still far below `chunk_ms`, so the worker
continues to keep up with the mic.

**Diminishing returns past 200 ms**: `--rvc-context-ms 400` gives
only +0.07 dB SNR over 200 ms in the audit, so 200 is the default.
Set 0 to A/B against Stage 2G.

The remaining structural ceiling is the per-chunk-vs-whole-file
inference gap that left-context alone cannot fully close (chunk N's
output for time T differs slightly from chunk N+1's output for the
same T due to internal model state). A future revision could attempt
true bilateral overlap with same-input-time crossfade; this is
deferred until measured listening evidence justifies the added
complexity.

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

## Audit tool — reproducing the model-faithfulness comparison

`tools/pseudo_stream.py` runs the same model the realtime path uses,
but in pure offline mode (no audio devices, no worker thread, no
queues). It exists so a regression in the realtime chain can be
isolated end-to-end without bringing the audio stack online.

```
.\.venv310\Scripts\python.exe -m tools.pseudo_stream `
    --input-wav test.wav `
    --output-wav audit_pseudo_stream.wav `
    --model-profile config/model_profiles/kiki.example.json `
    --device cuda `
    --chunk-ms 1000 `
    --crossfade-ms 0 `
    --resampler polyphase `
    --stream-sr 48000 `
    --report-json audit_pseudo_stream.json
```

A/B the resulting WAV against `tools.offline_infer`'s output (the
offline whole-file reference). The output is gitignored under
`*.wav`. The JSON report captures per-chunk inference timing and
amplitude metrics for regression tracking.

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

## Stage 3 — realtime RVC (recommended invocation)

```
.\.venv310\Scripts\python.exe -m src.main --mode rvc `
    --config config/runtime.example.json `
    --model-profile config/model_profiles/kiki.example.json `
    --input-device-substring "WO Mic" `
    --output-device-substring "CABLE Input" `
    --device cuda `
    --chunk-ms 1000 `
    --rvc-context-ms 200 `
    --rvc-queue-ms 6000 `
    --rvc-prebuffer-ms 3000 `
    --warmup-rvc-count 2 `
    --duration-seconds 60
```

`--rvc-context-ms 200` (the default — shown explicitly here) gives
the model 200 ms of real previous input audio as warmup left-context
before each chunk, and the worker trims the corresponding region of
model output proportionally so emit duration stays ≈ chunk_ms (no
timeline drift). See "Stage 3 — input-side left-context" below for
the audit measurements. `--crossfade-ms` is omitted; the model-
faithful default is 0.

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
