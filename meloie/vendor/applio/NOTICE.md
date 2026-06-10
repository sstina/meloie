# Vendored from Applio (MIT)

This directory (`rvc/`) is a **vendored subset** of [Applio](https://github.com/IAHispano/Applio)
by **AI Hispano**, used as our Path-A realtime RVC inference core (its `Synthesizer`,
generators, encoders, RMVPE/FCPE predictors, embedder loader, and realtime persistent-buffer
pipeline). Applio is MIT-licensed; the full license text is reproduced in `rvc/LICENSE` /
below, and is retained per the MIT terms.

## What was copied
The inference import-closure only. **Excluded:** `rvc/train/` (training) and `rvc/models/`
(weights — we point the loaders at `RVC/models/` instead).

## What we do NOT import (and why)
- `rvc/realtime/core.py`, `rvc/infer/infer.py` — import `pedalboard` / `noisereduce` (FX +
  noise-reduction). We deliberately do **not** install or import those: our worker owns
  scheduling and the faithful-carrier contract forbids output FX. Our `StreamingRvcEngine`
  imports only `rvc.realtime.pipeline` (+ its clean deps).
- `rvc/realtime/{worker,client,callbacks}.py` — the websocket server; replaced by our own
  async-queue worker.
- All voice-altering knobs in the pipeline (autotune, proposed_pitch, formant, volume_envelope,
  output×RMS scaling) are left unused — we drive only the faithful conversion path.

## Upstream
Applio © 2026 AI Hispano, MIT License. https://github.com/IAHispano/Applio
Vendored 2026-05-30 for the Tvoice realtime RVC project.

---

MIT License

Copyright (c) 2026 AI Hispano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
