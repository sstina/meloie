# Third-party provenance: Applio (MIT)

`meloie/core/` is **derived from [Applio](https://github.com/IAHispano/Applio)**
by **AI Hispano** (MIT license). It began as a verbatim vendored subset of
Applio's inference import-closure and was internalized into first-party Meloie
code on 2026-06-10: dead code (training paths, downloaders, TTS, the offline
pipeline, CREPE, the Config singleton) was removed, all data paths were made
explicit (no CWD-relative resolution, no network downloads), and the
faithful-carrier contract was baked into the API (the internalized
`voice_conversion` has no volume-envelope / FX / output-denoise parameters).

**Model compatibility is preserved**: every `torch.nn.Module` attribute name and
`ModuleList` layout that forms a `state_dict` key (`enc_p.*`, `dec.*`, `flow.*`,
`emb_g.*`, RMVPE `unet./cnn./fc.`) is unchanged, so standard RVC v2 `.pth`
checkpoints and `rmvpe.pt` load exactly as upstream.

Files derived from Applio carry a short attribution header pointing here.

## Upstream

Applio © 2026 AI Hispano, MIT License. https://github.com/IAHispano/Applio
Vendored 2026-05-30, internalized 2026-06-10.

## Third-party components (transitive lineage)

`meloie/core/` forked from Applio, which itself integrates several upstream
projects. All are permissively licensed and redistributable here:

- **RMVPE** (F0 estimator — `predictors/rmvpe.py`): Dream-High/RMVPE,
  **Apache License 2.0**, © the RMVPE authors.
  https://github.com/Dream-High/RMVPE — redistributed with attribution and a
  statement of modifications per Apache-2.0 §4 (see the file header).
- **HiFi-GAN** (vocoder decoders — `synth/hifigan*.py`): jik876/hifi-gan,
  MIT License, © 2020 Jungil Kong. https://github.com/jik876/hifi-gan
- **NSF source-filter** (the `SineGenerator` in the NSF decoder): from the
  RVC / Applio lineage of Xin Wang et al.'s neural source-filter work.
- **ContentVec** (speech-embedder config under
  `models/embedders/contentvec/`): auspicious3000/contentvec, MIT License.
  https://github.com/auspicious3000/contentvec

Weight files (`*.pth`, `*.index`, `rmvpe.pt`) are NOT part of this repository.

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
