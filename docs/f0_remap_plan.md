# F0 重映射方案 — 设计与执行计划（A-first）

> 状态：**A1 base-free 探测完成 = NO-GO**（emb_pitch 权重无法恢复训练音域）；按预先约定转向
> formant-first + 手种子 target。详见 §8。本文件是单一可信设计稿；实现落地后同步 `rvc.md`。
> 起因：用户拿到一份 CDF/quantile F0 映射 proposal，想优化现有"裸 pitch+12"。经两轮 Claude 交叉
> 评审 + 对真码核验，定为下述 A-first 方案。

---

## 0. 目标 & 契约
- **目标**：把"每个模型手填一个 pitch_shift"升级成"自动把用户语音搬进该模型的舒适音域"，并让
  共振峰真正承担"变性别"。直接收益：**B/C/F 等未调音模型不再因 F0 落在训练域外而电音**。
- **契约（硬约束）**：所有改动**纯输入端**（改"喂模型什么 F0/载波"，不改模型输出），与现有
  `pitch_shift` / `proposed_pitch` / `formant` 同类。**输出端零塑形**（§4 忠实搬运契约）。新增的离线
  分析工具只读权重、容器化到 D 盘。

## 1. 概念基础（为什么不是 full-quantile / 不是 γ-σ 匹配）
- RVC 的 NSF vocoder **复刻你喂进去的 F0**（输出基频 ≈ 输入×2^(transpose/12)）。所以模型**没有**一个
  独立的"目标说话人 F0 分布"等你去匹配——它真正相关的属性是**舒适音域**（音色干净、不电音的 F0 区间）。
- 推论：full-quantile（逐点分布匹配）和 γ=σ_t/σ_s（range 匹配）在 RVC 上**没有原则性目标**，只会
  改写用户自己的语调起伏。**真正有用的只有两件事**：(a) 精确把用户**中位**移到舒适音域中心；
  (b) 把**极端音高**软压回音域边界（专治电音）。共振峰是第三件，正交且已实现。

## 2. 数据来源：emb_pitch 偏移法（核心创新）
RVC v2 的 F0-条件 TextEncoder 有一层离散音高嵌入 `enc_p.emb_pitch = nn.Embedding(256, h)`
（`encoders.py:121`，`x += emb_pitch(coarse_f0_bucket)`），可从 checkpoint `cpt["weight"]
["enc_p.emb_pitch.weight"]` 直接读出 `[256, h]`（与 `emb_g.weight` 同样可读，无需实例化模型）。

- **原理**：嵌入查表是稀疏梯度——训练数据出现过的桶（=训练过的 F0 区间）其行向量被推动得多；
  没出现的桶只受 weight-decay 均匀缩。于是"活跃桶集合 ≈ 模型舒适音域"。
- **质心（A 用）**：数据驱动，**对底模基本免疫**（refinement #2）→ 给自动对齐的 `target_f0_median`。
- **边界（Phase B 用）**：依赖"训练区外 Δ→0"的零地板，**底模一错就失效** → 风险只落在**推迟的
  Phase B**，不影响先 ship 的 A。这反而加固 A-first。
- **base-free（A1 实际用法，refinement #3）**：不依赖外部底模——直接相减两个**微调后**矩阵
  `emb_pitch_A − emb_pitch_B`（共享底模一阶抵消），看 A/B 音高使用差异；再用**一个**外部锚点
  （卖家 ~200Hz 或一次 5s 参考）把相对结果钉到绝对 Hz。辅以单模型**逐桶行范数 ‖emb_M[k]‖** 剖面
  （weight-decay 把未用桶缩小 → 活跃带浮现）。
- **边界用 Δ-加权分位数（2/98 百分位），不取最外侧活跃桶**（refinement #4）——单个 creak/八度错误
  帧落进远桶会撑歪 min/max；质心（Δ-加权均值桶）不受影响。
- **桶↔Hz**：反 mel 量化前先核模型训练用的 `f0_min/f0_max`（RVC 默认 **50/1100**，refinement #5）；
  非默认会让绝对 Hz 整体偏。`bucket→Hz: f0_mel=(b-1)·(mel_max-mel_min)/254+mel_min; f0=700·(e^(mel/1127)-1)`。

## 3. 分层计划

### Phase A —— 自动对齐 + 共振峰（零 vendored 改动、零子类、默认关、可 A/B）
- **A1（只读，进行中）** `tools/analyze_model_f0.py`：读 A/B/C/F 的 `emb_pitch`，base-free 求每模型
  活跃带（行范数 + 跨模型差分）、Δ-加权 2/98 边界 + 质心 → 反 mel → `target_f0_median`/`[min,max]`。
  **go/no-go 见 §5**。成了才写进各 `config/model_profiles/<stem>.json`（扩 `ModelProfile` 加 optional
  `target_f0_median/min/max`；否则 `model_profile.py:105` 拒未知键）。
- **A2 小数 + 自动 pitch_shift**：放开 `streaming_engine.set_pitch_shift` 的 `int()` → float。第一方
  **慢 EMA register 跟踪器**（不需子类，经现有 `f0_up_key` 标量每块喂入；`proposed_pitch` 关）：
  - **τ ≈ 10–30s**（refinement #1），必须**显著长于句子语调尺度**（declination ≈ 1–3s）。太短=复现逐块
    中位的语调压平 bug；偏长是安全方向。
  - **清音/静音/低置信块冻结 EMA**、不更新 running median。
  - 上面两条是 A2 真正优于逐块 `proposed_pitch` 的前提，缺了就只是"平滑版的同一个 bug"。
  - 计算：`f0_up_key = clamp(12·log2(target_median / ema_user_median))`，小数，慢变 → 句内微语调原样
    透传（这顺带实现 proposal 第3层"register/detail 两带"，无需未来帧/CWT/子类）。
  - MVP 退化档：静态一次性计算（用户 median 用默认~110Hz 或一次 5s 校准），先救 B/C/F 电音。
- **A3 共振峰（A）**：接 `formant_timbre` 进 A.json，但**调参靠测输出共振峰上移到 ~1.15–1.25（或 A/B）**，
  **不盲信卖家"0.25"这个数**（refinement: 0.25 是 formant_timbre 参数值，其实际共振峰缩放方向/幅度未
  确认）。这步对"听感变女"ROI 最高。

### Phase B —— 尾部软压（解锁但暂不建）
A1 顺带算好 `[min_t,max_t]` → 零额外数据成本为 B 解锁。**仅当** A 居中后极端音高仍电音才建：第一方
**subclass `RealtimeVoiceConverter` 覆写 `get_f0`**（`pipeline.py:270` 是唯一 F0 钩子，vendored 不可改），
对连续 F0 做 tanh 软压到 `[min_t,max_t]` 再重量化。注意：边界依赖底模正确性（§2）→ 建 B 前需校准底模。

### 不做
- **Phase C**（γ-仿射 + 用户 μ_s/σ_s 校准 UX）：RVC 上低 ROI + 校准是新 UX。
- **Phase D**（CWT 两带 / full-quantile）：至多离线"精确模式"。

## 4. 底模依赖（集中说明）
- 绝对 Δ-vs-base 法需要 A 微调所基于的 v2 底模（A=40k → 标准件 `f0G40k`，需下载到 D 盘容器化）。
- **A1 刻意做成 base-free**（§2）以绕开此依赖；底模风险只影响 Phase B 的边界精度，不影响 A 的质心。
- 若 base-free 信号不清，再考虑 A1b：下载 `f0G40k` 算绝对 Δ（属离线分析数据，非运行时改动）。

## 5. go/no-go 判据（refinement #6 —— 不用循环论证）
- **不可用**："A median 与现有 +12 一致" —— 循环（+12 本就来自卖家"典型男声"假设 ≈ 假设
  target≈2×110≈220，自我印证）。
- **go**：(a) 某模型的 **Δ-vs-桶曲线是清晰单峰活跃带**；(b) **跨模型八度关系**成立（A 女质心 Hz
  ≈ 2× 某男声模型质心 Hz，在 Hz 域比、非桶域）。
- **no-go / 回退**：曲线平/噪声一片 → 微调太短或底模错配 → **回退耳调种子**，别硬解释。

## 6. 风险
- 底模匹配（见 §4，已用 base-free + 质心免疫缓解）。
- A2 跟踪器多一次（可降采样，慢跟踪天然低频）输入 F0 估计 — 成本可忽略。
- 共振峰参数方向/幅度未知（A3 用测量对 ~1.2 而非盲信数字）。

## 7. 验收
`pytest` 全绿（纯输入端无回归）；B/C/F 开自动对齐后不电音、语调不被压平（对比 proposed_pitch）；
A 调 formant 后 A/B 试听更自然。

## 8. A1 执行结果（2026-06-01）+ 方案修订
**结果：base-free emb_pitch 探测 = NO-GO。** `tools/analyze_model_f0.py`（只读 A/B/C/F 权重）：
- 单模型行范数 ‖emb[k]‖ 四个模型几乎相同、峰都在 ~700–805Hz、speakingFrac=0.00 → 它反映**底模
  emb_pitch 固有幅度结构，不是训练使用**（行范数随桶号增长是嵌入的内禀性质，非用量直方图）。
- 同底组配对差分（A−C @40k、B−F @48k，共享底模一阶抵消）：峰落在 b6–11（~63–76Hz 边缘），但
  speakingFrac 仅 0.10–0.22，能量主体仍在高桶——被底模幅度尺度淹没，speaking 段无干净带。
- 跨模型八度检验失败（质心比 1.04×，非 ~2×）。
**判定**（依 §5 与 refinement #6，不硬解释）：微调过短 / 底模主导 → emb_pitch 未显著移动训练桶 →
**权重无法恢复训练音域**。`emb_g`(说话人) 可能动得多，但 `emb_pitch`(音高桶) 没有。

**修订（fallback，按预先约定）：**
- **target_f0_median 改为 seller/耳种子**（A≈200Hz；B/C/F 各一次耳调或参考音频测量），不再从权重导出。
- **A1b（真底模 f0G40k/48k + 归一化 Δ=‖emb−base‖/‖base‖）**：可选升级、**低置信**（短微调信号弱 +
  卖家可能用非标底模），**不主动做**，待定。
- **优先级重排**：
  1. **A3（A 的共振峰）= 现最高 ROI**：零 F0/target 依赖、机制已建（`set_formant`/`formant_timbre`），
     只需对 ~1.15–1.25 共振峰上移调参。需要一个**离线共振峰测量工具**把"听感变女"客观化（避免盲调 0.25）。
  2. **A2（小数 pitch_shift + 用户侧慢 EMA 居中）= 稳健性升级**，用手种子 target_median；小数化本身无论
     如何都该做（移除 `int()` 截断）。
  3. **Phase B 尾部软压进一步推迟**：它依赖音域边界，而边界现在无可信来源（除非走 A1b / artifact 探针）。

`tools/analyze_model_f0.py` 保留为只读诊断（若日后拿到真底模可加归一化重测）。

### A3 实测结果（2026-06-01，`tools/measure_formant.py`）
读 stftpitchshift 源码 + 合成元音实测确认：**`formant_timbre`（= 库的 `distortion`）是共振峰频率的
直接乘数**——它 lift 出共振峰包络、白化、按 `distortion` **重采样包络频率轴**再贴回（`pitcher`/
`resampler.linear`）。实测 centroidRatio ≈ distortion，近乎线性：0.25→0.28，1.0→1.00，1.15→1.15，
**1.20→1.19**，1.5→1.52，2.0→2.03。
- **卖家"0.25"方向错了**：把共振峰压到 ~28%（巨低共振峰、男声反方向），用在我们引擎里只会更不像女声。
  （0.25 大概是卖家工具里某个 0–1 性别滑块的刻度，不是我们引擎直传的 `distortion`。）
- **M→F 上移 ~1.15–1.25 → 设 `formant_timbre` ≈ 1.20**（实测 ~1.19×）。
- `formant_timbre` 已是 `ModelProfile` 字段 → 确认听感后直接给 A.json 加 `"formant_timbre": 1.20`
  即可（`formant_on` 由 timbre≠1.0 自动派生），无需改 schema。纯 INPUT 端、契约安全。
- **下一步（需你的耳朵）**：GUI formant 滑块在模型 A 上设 ~1.20 实时 A/B 试听 → 满意则 bake 进 A.json。

### A2 进度（2026-06-01）：引擎核心已实现 + 测试，接线待办
`streaming_engine.py`：`set_auto_center(on,target_hz,tau_s)` + worker 线程 `_update_auto_center`（慢 EMA
τ≈10–30s、清音冻结、同款 f0 predictor、小数 `_auto_offset`），`_convert` 用 `f0_up_key=pitch_shift+
(_auto_offset if on else 0)`。**纯加法、默认 OFF、`_auto_offset` 唯一写者=worker 线程→无竞争、异常不断音频。**
`tests/test_auto_center.py` 7 项绿，pytest 157 passed。**待办**：session.set_auto_center 委托 + backend @Slot +
per-model `target_f0_median`(扩 ModelProfile，A 种子≈200Hz) + CLI/GUI 开关 → 用户才能实际开启/试听。

### 辅音诊断（2026-06-01，与 A3/formant 强相关）
点(diǎn)→扁(biǎn) 的 d→b：**~70–85% 是 RVC/ContentVec 固有地板**（塞音爆破+F2 locus 在 5–30ms，
16kHz/~20ms 帧/说话人解耦丢细节→偏低位唇音；中文最小对立+送气更敏感；旋钮补不回）。**我们能动的 15–30%**：
- **formant↑ 伤辅音**（StftPitchShift 抹爆破/F2）→ **A3 别一味加 formant，1.10–1.15 可能比 1.20 平衡**；
- **protect 0.33→0.45**（清音帧多留原辅音，代价小）；
- **宽带有线麦**（蓝牙 HFP 窄带是 d→b 灾难放大器）；
- index_rate=0（A 未开）已排除。

---
（变更日志：实现各 Phase 时在 `rvc.md` §10 追加条目。）
