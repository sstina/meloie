# Tvoice / RVC — Realtime Voice Changer

Turn the voice you speak into a trained target voice (RVC) in real time and
send it to Discord / OBS / a recorder. There is exactly one route:

```
系统默认麦克风 (Windows default recording device)
  -> Python pipeline (chunk -> RVC inference -> resample to stream SR)
  -> CABLE Input   (the app renders here)
  -> CABLE Output  (Discord / OBS / recorder select this as their "microphone")
```

## Design stance: a faithful carrier, not a sound-design tool

**The trained model is the voice.** A model bundle is a `.pth` + its `.index`
+ the supporting `hubert_base.pt` + `rmvpe.pt`, plus the inference parameters
they were trained against (in the model profile). The runtime's whole job is
to play that bundle through the realtime audio chain **faithfully, stably, and
safely**.

Between `engine.infer_array(...)` returning a chunk and the audio reaching
`CABLE Input`, the runtime does **only** what is structurally required:

- resample the model's native sample rate to the stream rate (sinc-polyphase),
- a sample-accurate slice that drops the input-side warm-up context and the
  look-ahead tail pad,
- a NaN/Inf scrub (safety, not shaping).

There is **no pitch shift, no time-stretch, no crossfade, no EQ, no limiter,
no gain shaping** anywhere in the runtime. To get a different voice, train or
load a different model. Voice-identity parameters (`f0_method`, `index_rate`,
`protect`, `filter_radius`, `rms_mix_rate`, `pitch_shift`) live in the model
profile — they are properties of the trained model, not user knobs, and the
CLI has no flags to override them. One of them, `rms_mix_rate`, is special:
it must stay at **1.0** to be faithful. Below 1.0 the backend's `change_rms`
imposes the *source mic's* loudness envelope onto the model output — runtime
gain shaping that breaks the contract — so 1.0 (the model's own loudness) is
the only contract-compliant value.

## Quick start (PowerShell)

```powershell
# 1. dot-source the env script (caches/temp -> RVC\, UTF-8 console, venv on)
. .\setup_env.ps1

# 2. see your devices — the system default mic is marked "I", outputs "O"
python -m src.main --list-devices

# 3. (optional) confirm the cable carries audio: tone -> CABLE Input -> CABLE Output
python -m tools.verify_cable_route --duration-seconds 2

# 4. run: system default mic -> kiki model -> CABLE Input
python -m src.main `
    --config config/runtime.example.json `
    --model-profile config/model_profiles/kiki.example.json `
    --device cuda
```

In Discord / OBS / your recorder, select **`CABLE Output (VB-Audio Virtual
Cable)`** as the microphone. Speak — you are heard as the model's voice.

`Ctrl+C` stops the run; a final metrics summary is printed.

## The microphone: system default, with an override

By default the runtime captures the **Windows default recording device**
(your "系统默认 mic") — change it in Windows sound settings and the runtime
follows. To pin a specific mic instead, set `input_device_substring` in the
config or pass `--input-device "Realtek"` (a name fragment). The runtime
refuses `CABLE Output` as input (that would feed the cable back into itself).

## VB-CABLE routing rules (do not invert)

- The app **renders to `CABLE Input`** (the virtual cable's render side).
- Downstream apps **select `CABLE Output`** as their microphone.
- The app's input is a **physical microphone** — `CABLE Output` is refused as
  input unless the diagnostic `--allow-virtual-cable-input` flag is set.

## Model profile

Voice identity is a JSON file under `config/model_profiles/`
([`kiki.example.json`](config/model_profiles/kiki.example.json)):

```json
{
  "name": "kiki",
  "model_path":  "models/kiki/kikiV1.pth",
  "index_path":  "models/kiki/kikiV1.index",
  "hubert_path": "models/kiki/hubert_base.pt",
  "rmvpe_path":  "models/kiki/rmvpe.pt",
  "f0_method": "rmvpe", "index_rate": 0.0, "protect": 0.33,
  "filter_radius": 3, "rms_mix_rate": 1.0, "pitch_shift": 12, "resample_sr": 0
}
```

Paths are relative to the directory you run `python -m` from (the project
root). Place model assets under `models/` (gitignored).

### `pitch_shift` is the transpose (变调) — it matters a lot

`pitch_shift` (semitones) transposes the **input F0** before conversion. It is
the single most important knob for getting a usable voice: a **female model
driven by a male voice needs roughly +12** (one octave). At `0`, a female
model produces a female timbre stuck at a too-low pitch — the uncanny
"电音"/robotic quality. kiki's intended value (from the tool it shipped with)
is **+12**, with `index_rate=0.0`. Pitch is voice-dependent, so override it
per run with **`--pitch SEMITONES`** (e.g. `--pitch 12`, `--pitch 7`) to find
what suits your voice. `--pitch` conditions the model's input pitch — it is
**not** an output pitch-shift, so it stays within the faithful-carrier design.

## Engineering knobs (the only user-facing controls — none change the voice)

| Flag | Default | What it controls |
| --- | --- | --- |
| `--device` | `auto` | Inference device (`auto` / `cuda` / `cpu`) |
| `--chunk-ms` | 500 | RVC chunk size (accumulation latency); larger = more model context. Floor ~400: worst-case inference is a fixed ~350 ms regardless of chunk |
| `--rvc-context-ms` | 500 | Input-side left-context warm-up fed to the model, then sliced away. Continuity only; clears the decoder's ~240 ms internal lead-in margin (see [docs/realtime_study_notes.md](docs/realtime_study_notes.md)). |
| `--tail-pad-ms` | 30 | Look-ahead tail pad that absorbs the model's deterministic ~20 ms tail-frame loss, then sliced away. No stretch, no pitch. |
| `--sola-search-ms` | 10 | SOLA seam-alignment search window. Phase-matches each chunk's seam to the previous chunk's tail by **choosing the cut offset** (no crossfade, no blend, no sample edit) — kills chunk-boundary comb-filter "电音". 0 disables. Must be ≤ `--tail-pad-ms`. |
| `--rvc-queue-ms` | 6000 | Per-direction queue capacity |
| `--rvc-prebuffer-ms` | 800 | Output silence before first real audio = the standing output latency. Absolute cushion (decoupled from chunk) sized to cover one inference spike + one output burst. Lower = less latency, more underruns |
| `--warmup-rvc-count` | 2 | Dummy inferences before opening the stream (hides the ~30 s cold start) |
| `--drop-stale-input` / `--no-drop-stale-input` | on | If inference falls behind, drop oldest chunks so latency stays bounded |
| `--silence-threshold-dbfs` | off | SilenceFront: skip inference on chunks below this input RMS (dBFS) and emit silence — saves GPU, no voice change. Opt-in (off by default so it can never gate out soft speech) |
| `--silence-hangover-ms` | 500 | Keep processing this long after the last voiced chunk so soft/trailing syllables aren't clipped (only used when the threshold is set) |
| `--input-device` / `--output-device` | system default / `CABLE Input` | Device name fragments |
| `--duration-seconds` | run until Ctrl+C | Stop after N seconds |

**Latency budget** ≈ `chunk_ms` (accumulation) + `prebuffer_ms` (standing output) +
~150 ms inference + ~40 ms device ≈ **~1.5 s** at the defaults. The standing output
prebuffer used to dominate (it defaulted to 3 × chunk = 3 s); decoupling it to an
absolute 800 ms and dropping the chunk to 500 ms is what brought the link down from
~2 s. Going lower means a smaller chunk, but the ~350 ms inference floor caps that at
~400 ms; sub-second would require pacing the output burst (a deferred code change),
not just smaller buffers. None of these touch the voice — see the design stance above.

## Continuity & stability (why the chunked pipeline stays drift-free)

Per-chunk RVC inference has three structural quirks; all are handled faithfully,
with no reshaping of the model's samples (distilled from a study of five mature
RVC projects — see [docs/realtime_study_notes.md](docs/realtime_study_notes.md)):

- **Cold start per chunk.** HuBERT / RMVPE / index re-initialise each call, so
  the first tens of ms of a chunk would wobble. `--rvc-context-ms` (500 ms)
  feeds the model the previous chunk's real audio as warm-up, then the output
  for that region is sliced away. 500 ms clears the RVC decoder's internal
  ~240 ms lead-in margin. (Real past audio, faithfully sliced.)
- **Tail-frame loss.** The model emits exactly one ~20 ms HuBERT frame less
  than the input demands, at the tail. `--tail-pad-ms` feeds the next chunk's
  real audio as a look-ahead tail so the lost frame lands in the pad, then the
  worker emits the exact `chunk_size` slice. Output stays drift-free with no
  time-stretch and no pitch change.
- **Seam phase mismatch ("电音").** Two independent renders meeting at a hard
  cut differ in phase → a comb-filter/electronic artifact. `--sola-search-ms`
  cross-correlates each chunk's seam against the previously emitted tail and
  **chooses the phase-matched cut offset** (SOLA *alignment* only — it picks
  *where* to join; it never blends, gain-ramps, stretches, or pitch-shifts a
  sample, so it stays a faithful carrier). This is the canonical fix every
  reference project uses; we adopt only its faithful half (alignment), never
  the voice-altering crossfade blend.

## Identity-first safety net

On any backend error (CUDA OOM, NaN, model fault) or degenerate output, the
worker emits **the chunk's own audio** (the user's own voice) for that chunk
instead of silence or a crash. The link stays alive; `rvc_fallback_count`
makes it observable.

## Health checks (bisect the link if it ever breaks)

- `python -m tools.offline_infer --input-wav test.wav --output-wav out.wav --model-profile config/model_profiles/kiki.example.json --device cuda`
  — proves the **model + inference** are healthy on a file (no audio devices).
- `python -m tools.verify_cable_route --duration-seconds 2`
  — proves the **VB-CABLE transport** (CABLE Input → CABLE Output) carries audio.

If offline inference works and the cable carries a tone, but downstream is
silent, the problem is **device routing** (wrong/silent input device, or the
downstream app not listening on `CABLE Output`).

## Metrics

A per-second line and a final summary report: frames in/out, queue depths and
drops, output underruns (split startup vs steady-state), input/output peak/RMS
dBFS, NaN scrubs, status flags, RVC chunks processed, inference last/mean/max
ms vs the per-chunk budget, fallback count, stale-chunk drops, resample timing.

## Testing

```powershell
python -m pytest -q
```

Pure tests only — no audio hardware, no GPU, no internet, no model files. The
RVC stack is faked via duck-typed engines so the suite runs anywhere with
numpy + pytest.

## Dependencies

`requirements.txt` lists only `numpy`, `sounddevice`, `pytest`. The RVC stack
(`infer-rvc-python`, `torch`+CUDA, `torchaudio`, `faiss-cpu`, `fairseq`,
`pyworld`, `librosa`, `scipy`) is installed once into `.venv310` (the runtime
venv). All caches/temp are redirected into `RVC\.cache` and `RVC\.tmp` by
`setup_env.ps1` — nothing is written to the C: drive.

## Principles

- **Faithful.** The model defines the voice; the runtime is a clean carrier.
  No runtime voice-shaping of any kind.
- **Stable first.** Quality-first defaults trade latency for continuity; the
  link should run without glitching, draining, or drifting.
- **Identity-first safety.** Inference failure falls back to the user's own
  voice, never silence or a crash.
- **No system mutation.** Never changes the Windows default device, app
  settings, registry, PATH, or drivers.
