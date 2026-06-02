# Tvoice / RVC — Realtime Voice Changer (v2)

Turn the voice you speak into a trained target voice (RVC) in real time and
send it to Discord / OBS / a recorder. There is exactly one route:

```
系统默认麦克风 (Windows default recording device)
  -> Python pipeline (StreamingRvcEngine: persistent buffer + F0 continuity + RVC inference)
  -> CABLE Input   (the app renders here)
  -> CABLE Output  (Discord / OBS / recorder select this as their "microphone")
```

This is a **v2-only build**: it runs v2-series (768-dim) RVC models such as the
default model **A** (`models/A.pth` + `models/V2.index`). A v1 / 256-dim model is
rejected at load with a clear message. The single realtime engine is the
**direct** Applio persistent-buffer engine, which runs in `.venv-applio`.

## Design stance: a faithful carrier, not a sound-design tool

**The trained model is the voice.** A model bundle is a `.pth` + its `.index`,
plus the inference parameters it was trained against (in the model profile). The
runtime's whole job is to play that bundle through the realtime audio chain
**faithfully, stably, and safely**.

The contract distinguishes **output** from **input**:

- **OUTPUT stays faithful.** Between the model's samples and `CABLE Input` there
  is **no EQ, no limiter, no normalize, no gain shaping, no time-stretch, and no
  output pitch-shift** — only a structural resample (model SR → stream SR), a
  sample-accurate slice, and a NaN/Inf scrub. The **one** sanctioned blend is a
  short sin² **seam crossfade** at block boundaries (plus SOLA alignment): a
  seam-only join of two renders of the *same* audio — it never changes pitch or
  timbre.
- **INPUT (the carrier) may be conditioned before conversion.** This does not
  vary the model's voice, only *what speech* it faithfully converts:
  - **`pitch_shift` (变调)** — transposes the input F0 (a female model driven by a
    male voice typically needs about **+12**). Model A defaults to `0`.
  - **real-audio context warm-up + look-ahead** for continuity, and
  - **optional input noise reduction** (`--direct-denoise`) so ambient noise is
    not converted into warbly voice. Default off, so a clean mic / soft speech is
    never silently degraded.

Voice-identity parameters (`f0_method`, `index_rate`, `protect`, `pitch_shift`)
live in the model profile — they are properties of the trained model.

## Quick start (PowerShell)

The simplest path — just double-click **`run_A_direct.bat`** (it bakes
`--direct-f0 fcpe` for model A and **asks at launch whether to enable input
denoise**, default off). Or, by hand:

```powershell
# 1. dot-source the env script (caches/temp -> RVC\, UTF-8 console, .venv-applio on)
. .\setup_env_applio.ps1

# 2. see your devices — the system default mic is marked "I", outputs "O"
python -m src.main --list-devices

# 3. (optional) confirm the cable carries audio: tone -> CABLE Input -> CABLE Output
python -m tools.verify_cable_route --duration-seconds 2

# 4. run: system default mic -> model A (v2) -> CABLE Input
python -m src.main `
    --config config/runtime.example.json `
    --model-profile config/model_profiles/A.json `
    --device cuda --direct-f0 fcpe --direct-denoise
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
([`A.json`](config/model_profiles/A.json)):

```json
{
  "name": "A",
  "model_path":  "models/A.pth",
  "index_path":  "models/V2.index",
  "f0_method": "rmvpe", "index_rate": 0.5, "protect": 0.33,
  "filter_radius": 3, "rms_mix_rate": 1.0, "pitch_shift": 0, "resample_sr": 0
}
```

Paths are relative to the directory you run `python -m` from (the project
root). Place model assets under `models/` (gitignored). The **index is loaded
only when `index_rate > 0`** — A uses `0.5` so `V2.index` is active; raise it for
a stronger timbre lock, lower it if it warbles. `pitch_shift=0` is A's default;
override per run with **`--pitch SEMITONES`** to find what suits your voice (it
conditions the model's input pitch — not an output pitch-shift).

**In the GUI**, the model dropdown lists the `.pth` files in `models/`. Tune the
carrier knobs live, then click **💾 记住当前** to save the current knobs as that
model's defaults (written to this profile) so it loads that way next time — set a
model's pitch once and it sticks.

## 融合音色 / Merge voices (捏脸 voice-morphing)

Blend two or more **same-architecture v2 models you own** into a new "in-between"
voice — an offline, contract-safe way to sculpt voices without training anything.
The result is a normal `.pth` the engine loads as usual.

**In the GUI** (`run_gui.bat`): the model dropdown lists the `.pth` files in your
`models/` folder by name. Open the Creative card's **🧬 融合模式**, check the
models to blend into the current (base) model, set each weight, and click **融合并
加载** — the merged voice is saved to `models/`, appears in the dropdown, and is
auto-selected. Or by hand with the CLI:

```powershell
. .\setup_env_applio.ps1
python -m tools.merge_models `
    --models models/A.pth models/C.pth --weights 0.6 0.4 `
    --output models/AC_mix.pth --name "A+C" --write-profile --profile-pitch-shift 12
```

- Only models with the **same** sampling rate / architecture merge (A + C are both
  40 kHz; mixing a 40 kHz with a 48 kHz model is refused with a clear message).
- `--weights` are normalized to sum 1 (default: equal). `--write-profile` drops a
  `config/model_profiles/<name>.json` so the GUI lists it; it sets `index_rate 0`
  (a merged voice has no shared index) — **remember to tune `pitch_shift`** (10–14
  for high/female voices) or it may sound neutral / 电音.
- The output must live under `RVC/` (no C: writes). Add `--verify-load` to confirm
  the merged model passes the v2 guard before you use it.

Faithful-carrier still holds: the merged **model** defines the voice; the runtime
plays it without reshaping.

## Engineering knobs (the only user-facing controls — none reshape the output)

| Flag | Default | What it controls |
| --- | --- | --- |
| `--device` | `auto` | Inference device (`auto` / `cuda` / `cpu`) |
| `--direct-f0` | profile's | F0 estimator — `rmvpe` or `fcpe` (the realtime engine backs only these two). `fcpe` is smoother + ~30% faster |
| `--direct-block-ms` | 250 | Output block size (ms). Lower = lower latency, more seams |
| `--direct-context-ms` | 2500 | Real past audio fed to the encoders each block (w-okada 额外推理时长; free latency-wise; bigger = steadier timbre + F0) |
| `--direct-crossfade-ms` | 50 | sin² seam crossfade overlap — the one sanctioned output blend; smooths block seams |
| `--direct-protect` | profile's | Protect voiceless consonants / breath (0..0.5); higher = less artifacting |
| `--direct-silence-dbfs` | off | Silence gate (响应阈值): below this input dBFS, emit silence + skip inference. Input-side; opt-in |
| `--direct-denoise` / `--no-direct-denoise` | off (launcher prompts y/N) | Input-side noise reduction before conversion (input conditioning, not output reshaping). The launcher asks each run; pass either flag to skip the prompt |
| `--direct-denoise-strength` | 0.5 | Denoise aggressiveness 0..1 (higher cleans more, can muffle soft speech) |
| `--pitch` | profile's | Transpose (变调) the input F0 in semitones |
| `--direct-formant` + `--direct-formant-timbre` | off / 1.0 | Input-side formant/gender shift (性别因子): timbre >1 = brighter/feminine, <1 = deeper/masculine; pitch untouched |
| `--direct-autotune` | off | Input-side F0 autotune (snap pitch to nearest semitone; creative) |
| `--direct-auto-pitch` + `--direct-auto-pitch-threshold` | off / 155 | Auto-derive the transpose from median F0 toward the target Hz (smart `--pitch`) |
| `--sid` | 0 | Speaker id for multi-speaker models (the model's own trained voice) |
| `--rvc-prebuffer-ms` | 800 | Output silence before first real audio = the standing output latency. Lower = less latency, more underruns |
| `--rvc-queue-ms` | 6000 | Per-direction queue capacity |
| `--drop-stale-input` / `--no-drop-stale-input` | on | If inference falls behind, drop oldest blocks so latency stays bounded |
| `--input-device` / `--output-device` | system default / `CABLE Input` | Device name fragments |
| `--duration-seconds` | run until Ctrl+C | Stop after N seconds |

**Latency budget** ≈ `direct-block-ms` (accumulation) + `prebuffer-ms` (standing
output) + per-block inference (~30–45 ms on an RTX 4080) + ~40 ms device. There is
also a one-time startup warm-up (~`direct-context-ms`) while the engine's context
buffer fills (it emits silence, not glitches). Lower `--direct-context-ms` to
shorten that startup; none of these touch the voice — see the design stance above.

## Continuity & stability (why the streamed pipeline stays drift-free)

The direct engine owns streaming **state** the old per-chunk path never had — a
persistent 16 kHz buffer plus F0 caches — so the three structural quirks of
chunked RVC are handled faithfully (distilled from a study of mature RVC projects
and the Applio realtime core — see
[docs/realtime_study_notes.md](docs/realtime_study_notes.md)):

- **F0 continuity.** F0 is recomputed only on the newest trailing window and
  shifted into a persistent cache, not re-estimated from scratch per block — the
  main perceptual lever (this is what `fcpe` smooths).
- **Real context, not a mirror.** Every block is conditioned on real past audio
  (`--direct-context-ms`); the generator decodes only the new region.
- **Seam phase match ("电音").** Two renders meeting at a hard cut differ in
  phase → a comb-filter artifact. SOLA cross-correlates each block's seam against
  the previously emitted tail and **chooses the phase-matched cut offset**, then a
  short sin² crossfade joins them — the one sanctioned seam blend.

## Identity-first safety net

On any engine error (CUDA OOM, NaN, model fault) the worker emits **the block's
own audio** (the user's own voice) for that block instead of silence or a crash.
The link stays alive; `rvc_fallback_count` makes it observable.

## Health checks (bisect the link if it ever breaks)

- `python -m tools.offline_infer --input-wav test.wav --output-wav out.wav --model-profile config/model_profiles/A.json --device cuda --f0-method fcpe`
  — proves the **model + inference** are healthy on a file (no audio devices); it
  runs the same direct engine, so an offline render predicts the live result.
- `python -m tools.verify_cable_route --duration-seconds 2`
  — proves the **VB-CABLE transport** (CABLE Input → CABLE Output) carries audio.

If offline inference works and the cable carries a tone, but downstream is
silent, the problem is **device routing** (wrong/silent input device, or the
downstream app not listening on `CABLE Output`).

## Metrics

A per-second line and a final summary report: frames in/out, queue depths and
drops, output underruns (split startup vs steady-state), input/output peak/RMS
dBFS, NaN scrubs, blocks processed, inference last/mean/max ms, fallback count,
stale-block drops, SOLA offset.

## Testing

```powershell
. .\setup_env_applio.ps1
python -m pytest -q
```

Pure tests only — no audio hardware, no GPU, no internet, no model files. The
RVC stack is faked via duck-typed engines so the suite runs anywhere with
numpy + pytest. The runtime itself (the direct engine) needs `.venv-applio`.

## Dependencies

`requirements.txt` lists only `numpy`, `sounddevice`, `pytest`. The v2 RVC stack
(the vendored **Applio** inference core under `src/vendor/applio/`, `torch`+CUDA,
`torchaudio`, `transformers`, `faiss-cpu`, `torchfcpe`, `noisereduce`, `librosa`,
`scipy`) is installed once into `.venv-applio` (the runtime venv). All caches/temp
are redirected into `RVC\.cache` and `RVC\.tmp` by `setup_env_applio.ps1` —
nothing is written to the C: drive.

## Principles

- **Faithful.** The model defines the voice; the runtime is a clean carrier of
  its output. Input conditioning (transpose, optional denoise) is allowed; output
  reshaping is not.
- **v2-only.** This build runs v2-series models; a v1 model is rejected at load.
- **Stable first.** Quality-first defaults trade latency for continuity; the
  link should run without glitching, draining, or drifting.
- **Identity-first safety.** Inference failure falls back to the user's own
  voice, never silence or a crash.
- **No system mutation.** Never changes the Windows default device, app
  settings, registry, PATH, or drivers.
```
