# Known issues / open quality items

Tracked quality items that are deferred, under investigation, or need a specific
test condition to reproduce. Latency/stability metrics are in the runtime summary;
this file is for **voice-conversion quality** issues (the core priority).

> Note: this build is now **v2-only**, running the direct (Applio persistent-buffer)
> engine on model **A** with `fcpe` F0 and optional input denoise. Earlier notes
> referencing the retired cache engine (kiki, `--chunk-ms`, `--precision fp32`,
> per-chunk RMVPE) are kept only as history — the current knobs are
> `--direct-block-ms` / `--direct-context-ms` / `--direct-f0` / `--direct-denoise`.

---

## 1. Quiet / soft speech may convert less accurately  — OPEN, watch-item

**Symptom.** When the user speaks **softly (小声)**, the model can fail to convert
the voice accurately — the output stays close to the source / sounds un-transposed,
instead of taking on the target timbre. Normal-volume speech converts fine.

**Most likely cause.** Low input level → weak signal → the F0 estimator cannot lock
onto the fundamental → frames are treated as unvoiced → no pitch conditioning/
transpose applied → "not converted." This is an **input-SNR / F0-voicing** effect,
largely independent of block/latency tuning (a buffering change cannot alter the
voice). A low-level phone-as-mic input compounds it.

**Mitigations within the faithful-carrier contract (all input-side).**
1. Raise the mic input level/boost in Windows sound settings, or speak closer —
   giving the model a stronger input changes what it *hears*, not the voice it
   *emits*.
2. The direct engine continuity (real context + F0 cache via `fcpe`) already
   steadies weak-signal F0 better than per-chunk RMVPE did.
3. If mic-level alone is insufficient, a faithful **input normalization** (scale
   the input toward a target level before inference, with a gate so near-silence
   does not raise the noise floor) is input conditioning analogous to `pitch_shift`
   — contract-compatible. Defer until a clean mic confirms the cause.

**Status:** OPEN (watch-item). Re-test on model A with a good microphone; if soft
speech is still rough, try input level/boost first, then consider a gated input
normalization. History: under the old cache engine, switching to FP32 RMVPE made
soft speech noticeably more intelligible — consistent with the weak-signal-F0
theory; the direct engine's `fcpe` + real context is the current equivalent.
