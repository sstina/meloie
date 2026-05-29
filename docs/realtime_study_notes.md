# Realtime RVC — study notes from reference projects

Distilled from five open-source RVC projects studied as textbooks (cloned to
`../_reference/`, **not** dependencies): `w-okada/voice-changer`,
`IAHispano/Applio`, `RVC-Project/Retrieval-based-Voice-Conversion` (+ `-WebUI`),
`codename0og/codename-rvc-fork`. Goal: fix realtime chunk-boundary artifacts
("电音") **without** violating our faithful-carrier contract (the runtime never
alters the model's voice — no pitch/formant/crossfade/time-stretch/gain/RMS/F0
post-processing; see `[[rvc-faithful-carrier-contract]]`).

## 1. The clean line: faithful vs voice-altering

Every project achieves smooth realtime seams with a **mix** of two kinds of
techniques. The line between them is what matters for us:

**Faithful — only conditions the model or chooses which unmodified samples to emit:**
- **Pad → infer → slice** with generous left context (and a little look-ahead).
  The context is fed to the model then discarded; the emitted samples are the
  model's own. This is our core design and theirs.
- **SOLA *alignment*** — cross-correlate the chunk's leading overlap against the
  previous chunk's tail to pick a phase-matched cut point. Chooses *where* to
  join; never modifies a sample.
- **Cut at energy minima / near-zero-crossings** (offline RVC, codename fork).
- **Persistent F0/pitch state across chunks** (Applio, codename `rtrvc`) — feeds
  the model continuous pitch; doesn't reshape output.

**Voice-altering — FORBIDDEN here (this is how *they* smooth seams, but it edits the carrier):**
- **SOLA *crossfade blend*** — equal-power `cos²`/`sin²` overlap-add of two
  different model renderings across the seam. Modifies samples in the overlap.
  (w-okada `VoiceChangerV2.py:268-269`, RVC-WebUI `gui_v1.py:983-987`,
  Applio `core.py`.)
- **Phase-vocoder seam** (`use_pv`, RVC-WebUI `gui_v1.py:26-47`).
- **`rms_mix_rate` / `change_rms`** — multiplies output by the *source's* loudness
  envelope (RVC `pipeline.py:39-58`, codename `pipeline.py:1088-1107`).
- **formant shift, noise-reduce (TorchGate), Pedalboard FX, f0 autotune** — all
  explicit voice edits; all off by default in the references, all off-limits here.

**Conclusion:** our omission of crossfade/SOLA-blend/RMS-mixing is correct and
matches every project's *off-by-default* or *contract-violating* category. The
faithful levers below are what we adopt.

## 2. Context / EXTRA defaults — the headline comparison

| Project | block/chunk | left context (EXTRA, sliced away) | seam method |
|---|---|---|---|
| w-okada voice-changer | ~683 ms (`serverReadChunkSize=256`) | **~85 ms** (`extraConvertSize=4096`@48k, `RVCSettings.py:12`) | SOLA align + crossfade blend |
| Applio realtime | 250 ms (`realtime.py:1597`) | **2500 ms** (`extra_convert_size`, `realtime.py:1619`) | SOLA + crossfade |
| RVC-WebUI `gui_v1` | 250 ms (`block_time`, `:123`) | **2500 ms** (`extra_time`, `:126`) | SOLA + crossfade (+opt PV) |
| codename realtime cfg | 520 ms (`block_time`) | **2460 ms** (`extra_time`, `config.json`) | stock SOLA |
| RVC offline reference | whole file | **3000 ms** symmetric reflect-pad (`x_pad=3`, sliced at `t_pad_tgt`) | cut at energy-min, **butt-join, no fade** |
| **ours (before)** | 1000 ms | **200 ms** | contiguous hard-cut + bilateral context |
| **ours (now)** | 1000 ms | **500 ms** | contiguous hard-cut + bilateral context |

Common frame constants everywhere: HuBERT input **16 kHz**, hop/window **160**
(⇒ 100 feature frames/s). F0 `f0_min=50, f0_max=1100`.

## 3. Why we raised context 200 → 500 ms (the load-bearing finding)

The RVC realtime generator slices its context **in the latent/flow domain** and
deliberately keeps extra lead-in:

```python
# RVC-WebUI infer/lib/infer_pack/models.py:758-770 (and Applio synthesizers.py:230)
flow_head = torch.clamp(skip_head - 24, min=0)   # keep 24 EXTRA flow frames
...
z = z[:, :, dec_head : dec_head + length]         # context sliced off here
```

The decoder wants **≥24 feature frames ≈ 240 ms** of left lead-in to render the
chunk's leading edge artifact-free. Our old `context_ms=200` (~20 frames) sat
*just below* that internal margin — a plausible contributor to boundary "电音".
Raising to **500 ms** clears it with headroom, is fully faithful (sliced away,
emit is still exactly `chunk_size`), and costs only a little inference time —
we measured huge budget headroom: inference mean ≈174 ms at the old 200 ms
setting, and **≈190 ms mean / 438 ms max at 500 ms context** — both far under the
1000 ms/chunk budget on the RTX 4080, with 0 underruns / 0 fallback / 0 drops in
a 30 s live run. Canonical realtime RVC uses up to 2500 ms, so 500 ms is
conservative; `--rvc-context-ms` can go higher if artifacts persist.

This is the **#1 faithful anti-artifact lever** all four studies converged on:
a cold-started chunk starves the model's receptive field at the edge; real
context (not sample editing) is the fix.

## 4. What we adopted now (faithful, low-risk)

- **`--rvc-context-ms` default 200 → 500 ms** (clears the decoder's 24-frame
  lead-in margin). Voice-neutral; emit unchanged.

We already do, and keep: per-chunk pad→infer→slice; one-sided left context +
small look-ahead tail (the references have *no* look-ahead — our `tail_pad_ms`
additionally gives the chunk's trailing edge real future context, which is
strictly better for the trailing seam); identity fallback; NaN scrub; drop-stale.

## 5. SOLA *alignment* — IMPLEMENTED (faithful, default on)

The references' dominant seam-smoother is SOLA, which is **two parts**:
1. **alignment** (faithful) — `argmax(normalized cross-correlation)` over a
   ~10 ms search window picks a phase-matched cut point;
2. **crossfade blend** (forbidden) — overlap-add of two renderings.

Part 1 alone removes the *dominant* comb-filter "电音", because that artifact
comes from **phase mismatch** at an arbitrary hard cut, and alignment fixes the
phase. Part 1 emits only the model's own unmodified samples — it just chooses
the join point — so it is faithful-carrier compatible. We adopt part 1 and
**never** part 2.

**Our implementation (clean-room, our own style — not a port):**
`find_sola_offset(haystack, needle)` in `src/audio/chunker.py` is a pure,
unit-tested normalised cross-correlation that returns the best alignment index.
The worker (`src/engine/worker.py`, `_seam_aligned_start`) exploits the overlap
we *already* render: because each chunk is fed `[context][chunk][tail_pad]`,
consecutive renders cover the same input audio around the seam. So we keep the
previous chunk's emitted tail (`sola_link`, ~20 ms), search ±`sola_search_size`
around the context anchor for the phase-matched offset, and emit the exact
`chunk_size` slice from there. No extra rendering, no chunk-stride change, and
**no timeline drift** — each chunk re-anchors at `context_size`, so offsets
(bounded by ±10 ms) never accumulate. On an identity render the offset is 0
(perfectly aligned); after a fallback the link resets so non-model audio is
never used as a phase reference. Defaults: `--sola-search-ms=10` (0 disables),
link = 2× search; requires `context_ms ≥ search+link` and `tail_pad_ms ≥ search`
(auto-disabled otherwise). Observability: `rvc_sola_applied_count`,
`rvc_sola_offset_last` (and `sola=±N` on the per-second line).

Reference we learned the alignment math from (we took only the faithful half):
`w-okada/server/voice_changer/VoiceChangerV2.py:252-266` (alignment) — and
explicitly **NOT** lines 268-269 (the crossfade blend).

## 6. Things we deliberately do NOT adopt

Crossfade/overlap-add blend, phase-vocoder seam, `rms_mix_rate` loudness mixing,
formant shift, noise-reduce, autotune. All edit the carrier; all violate the
contract; all are off-by-default even in the reference projects.
