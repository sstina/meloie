# Known issues / open quality items

Tracked quality items that are deferred, under investigation, or need a specific
test condition to reproduce. Latency/stability metrics are in the runtime summary;
this file is for **voice-conversion quality** issues (the core priority).

---

## 1. Quiet / soft speech is not converted accurately  — OPEN, needs re-test

**Symptom.** When the user speaks **softly (小声)**, the model fails to convert the
voice accurately — the output stays close to the source / sounds un-transposed,
instead of taking on the target (kiki) timbre. Normal-volume speech converts fine.

**First observed.** 2026-05-30, on the `WO Mic Device` input (a phone-as-mic app,
which itself runs at a low input level), with `--chunk-ms 500 --rvc-prebuffer-ms 800`.

**Most likely cause (not yet confirmed).** Low input level → weak signal → RMVPE
cannot lock onto the fundamental frequency (F0) → frames are treated as unvoiced →
no pitch conditioning/transpose applied → "not converted." This is an **input-SNR /
F0-voicing** effect that is *largely independent of chunk size*. The `WO Mic` phone
mic compounds it by delivering a low baseline level.

**What it is probably NOT.** Almost certainly not the chunk size or the latency
tuning: a buffering change cannot alter the voice, and the model + params are
byte-identical to the known-good config. But this has **not been A/B-confirmed yet**.

**Re-test plan (do after switching to a better microphone).**
1. With the new mic, speak the *same soft phrase* at `--chunk-ms 1000` and at
   `--chunk-ms 500`. If both convert soft speech equally (in)accurately → the cause
   is input level / RMVPE, **independent of chunk** → the conservative latency config
   (chunk 500) is cleared to stay.
2. If only `--chunk-ms 500` is worse on soft speech → chunk reduction is implicated →
   raise the default chunk back up until soft-speech conversion matches chunk 1000.
3. Quick zero-cost mitigation to try regardless: raise the mic input level/boost in
   Windows sound settings, or speak closer. (Giving the model a stronger input is
   input-side conditioning — it does not reshape the model's output, so it stays
   within the faithful-carrier contract; see README "Design stance".)

**Possible faithful fix if mic-level alone is insufficient.** A runtime **input
normalization** (scale the input toward a target level before inference) is input
conditioning analogous to `pitch_shift` (+12) — it changes what the model *hears*,
not the voice it *emits*, so it is contract-compatible. Risk: it would also raise the
noise floor during near-silence, so it needs a gate/threshold. Defer until the
re-test confirms the cause and mic-level tuning proves insufficient.

**Update 2026-05-30 — FP32 helped (partial).** Switching inference to FP32
(`--precision fp32`, now the default) made soft speech **noticeably better**: still
a bit rough/coarse, but now **intelligible** ("能听清说什么了"). Consistent with the
weak-signal-F0 theory — FP32 RMVPE is more precise on low-level input. FP32 is now the
default (faster too: live inference max dropped ~390 ms → ~257 ms). The issue is
improved, **not closed**.

**Status:** OPEN (improved by FP32). Latency config (chunk 500 / prebuffer 800,
FP32) is locked in as **provisional**. Next: re-test with the **new microphone**
(arriving ~2026-05-31). If a clean mic + FP32 still leaves soft speech rough, revisit
(mic input level/boost, or a faithful input-normalization gate). The chunk-1000-vs-500
A/B (step 1) only matters if soft speech is still bad after the mic swap.
