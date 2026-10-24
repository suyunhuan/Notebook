# Handwriting Raw Batch C

> 来源：7 张手写笔记照片（附件顺序 = Page 15–21）。转录已做笔误/缩写还原；模糊处标 `[?]`。

## Page 15 — aeb1-783e

**主题**: Video Thinking：历史视频 latent/state/action 经 WAN2.2 与 Action Gen 交互，预测未来帧与动作，并以 MoT/场景 prompt/开源 LLM 做场景理解条件。

**转录（纠错后）**:

**Video Thinking** `{`
- 视频 Caption
- 过去 N 帧 预测 未来 M 帧 和 action
`}`

```
hist Video latent
hist state
hist action
        ↓
   [encoder / latent block]
        ├──→ WAN2.2 Video Gen  ┐
        │         ↕ ↕ ↕        ├──→ output prediction
        └──→ Action Gen        ┘
```

**prompt + scene understanding 场景理解**

1. ① MoT 传入 text Tower
2. ② 场景 prompt
3. ③ 开源 LLM forward. 文本 hidden

**不确定项**:

- `WAN2.2`：按视频生成模型 Wan 2.2 理解；原笔记写作 WAN2.2 / WAN 2.2。
- MoT：本笔记体系多为 Mixture of Transformers；此处接 text Tower，也可能兼指感知 MoT 特征。
- 「传入」：亦可能读作「深入」。
- 「开源 LLM」：局部裁剪亦见「利用 LLM / 开源 VLM」读法，取与全文一致的「开源 LLM」。

---

## Page 16 — af72-7ba9

**主题**: 4D Encoder → 多模 4D+LiDAR/鱼眼 → MoT 多头，再经 SD+TBT/车辆运动学到 Action，并以 Latent Reasoning（AR LWM）做短时域 rollout。

**转录（纠错后）**:

（页眉旁）PV 示意图（透视锥 → 矩形）；旁注：`159000` / `160000` / `500m` / `66min`[?]

**①** `4D Encoder → { Depth, Ray Map, Semantic }`

**②** `多模 4D + Lidar → 4D`  
`Fisheye → { Seg, Depth }`  
二者 → **MoT** →  
- Depth  
- Ray  
- PV seg  
- BEV `{ B, C, G }`  
- Sparse BEV  

**③**

```
SD + TBT ────────┐
                 │→ [backbone] → BEV / PV / Sparse BEV
Vehicle Kinematics ──┴──────────────→ Action（运动学旁路）
```

**Latent Reasoning**

- rollout 4 步，2s 未来，用 GOD gt[?]
- action decoder 接入 AR LWM

```
Large Transformer → World state Reasoning
                      ├ t+0.5
                      ├ t+1.0
                      ├ t+1.5
                      └ t+2.5  → Action Decoder
```

**不确定项**:

- `66min`：亦可能是 `66mm`（焦距）；与 `500m` 并列时两种读法都说得通。
- BEV `{ B, C, G }`：对应后页 BZL / GOD / COG 一类头；局部亦见 `BZU`/`BZL` 草写。
- 「GOD gt」：亦可能是 `GOD st`（state）；按 ground-truth 监督读作 gt。
- `t+2.5`：相对 0.5 步长序列，末步跳到 2.5 可能为笔误（本应 t+2.0），按原文保留。
- SD+TBT：Standard Definition map + Turn-by-Turn（非 SOT）。
- Vehicle：原文笔误 `Vechicle`。

---

## Page 17 — b036-724c

**主题**: 相机时序注意力感知管线 + 多模 4D（Cam–LiDAR）融合，以及 S0–S3（预训练→dense depth→BGC→E2E）训练路线。

**转录（纠错后）**:

- RL 训练 DAgger
- 带 dense depth

```
Cam → PE → Frame Attn → Temp Attn → token compress → Spatial Attn
         ↑ intrinsic / extrinsic
         （旁注：S1 → PE）
```

- 开源 DA3 无几何语义，补 PVSeg
- 加内外参校验[?]
- 导出 metric depth
- stream POC：0.3s → 1s
- ray map 监督，解 pose 优于解[?]；优化求解训练不稳定
- 鱼眼相机接入

**多模 4D**

```
Cam → Patch → Transformer → DPT Head → PVSeg
                ↑ (+)
Lidar → Feat Extract
```

| S0 | S1 | S2 | S3 |
|----|----|----|----|
| 预训练 | dense depth | BGC | E2E |

**不确定项**:

- PE / PZ：主流读作 Positional Encoding；亦见 Patch/Projection Zone（PZ）读法；「S1 → PE/PZ」箭头位置略糊。
- 「校验」：亦可能是「约束 / 监督」。
- 「解 pose 优于解[?]」：后半对比对象模糊（位姿/POC/其它几何量）；ray map 与 pose 联合训练不稳定较明确。
- S2 `BGC`：亦可能读作 `RGC` / `BCG`；结合后页 BZL/GOD/COG，暂取 BGC（B/G/C 头）。
- DA3：与前批一致，倾向 Depth Anything 3。
- POC：流式 PoC 时长；与 PNC 易混，此处按 stream POC 保留。

---

## Page 18 — b114-7a46

**主题**: Fisheye/Camera/Navi/Ego 多路 tokenizer → 共享骨干 → BEV/PV/Action 多头，并约定 Occ Flow、Sparse BEV 与 Scene token 长时序规格。

**转录（纠错后）**:

```
Fisheye → tokenizer @ 20M ──→ BEV Token (504) ──PE─┐
Camera → ResNet → PA3D @ 0.4B → PV Token          │
         (220*2 + 60*5) ──PE──────────────────────┤→ [Transformer]
SD + TBT → Navi tokenizer @ 0.5M → Navi token 32 ─┤
Vehicle Kine → Ego tokenizer @ 0.5M ──────────────┘
                                                    ├→ BEV { BZL, GOD, COG, Navi }
                                                    ├→ PV { Seg, Depth }
                                                    └→ Action Decoder
```

**数据** `{` 一阶段；RL；DAgger；小障碍物采集[?]；PNC `}`

1. ① Occ Flow：5s Occ + Flow + SDF（0.6s 间隔，8 帧）
2. ② Sparse BEV 长时序（6 帧 @ 2Hz）
3. ③ PV token 压缩  
   - camera token 作 query，融合 Navi、Ego  
   - 隐式业务 traj.；压表[?]
4. Scene token 长时序 64 个（6 帧 @ 2Hz）

- 动态 weight
- 小障碍物 结合 tag
- MetaAction 辅助
- navi encode + rc, tsd, road class & dist 2 cross

**不确定项**:

- `PA3D @ 0.4B`：参数量亦见 `0.9B` 读法；模块名亦可能写作 PA3。
- PV token 计数：`(220*2 + 60*5)` 与 `(270*2 + 60*5)` 两种读法并存，取前者。
- 「一阶段」：亦可能是 HITL/HITS 草写。
- 「采集」：原文似「采构」。
- PNC / poc：Planning & Control 与 PoC 易混，括号内按 PNC 还原并标疑。
- 「压表」：查表/压缩查找表之意，字迹简。
- `rc` / `tsd`：倾向 route/road cross、traffic sign detection；亦可能是 sd（SD map）。
- BZL / COG：项目内 BEV 头缩写（车道/认知类），全称未写。

---

## Page 19 — b1e6-784c

**主题**: OPD/MOPD 蒸馏要点（one-hot vs soft、单/全词表 KL、单/多轨迹）及指令跟随训练超参与 special token 配置。

**转录（纠错后）**:

算法技能树 · 跟踪 · 决策[?]

- OPD
- MOPD

- gt 是 one-hot
- teacher 是 soft 分布
- student 解码 rollout
- 多场景单强[?] expert
- SFT + RL → teacher large
- 单词表 KL
- 全词表 计算量大
- 比 KL `{` 无 reward hacking / dense reward `}`
- off policy
- 单轨迹
- VLM[?] 多轨迹
- 性能逼近 teacher

单配置测试最优方案  
指令跟随 / 物体 + TBT 等任务

- bf16；`num_workers=8`；data 不解压
- `bs = bs/gpu * gradient_accum * GPUs = 1`
- 10 epochs；3k 存 ckpt

**special token:**

- `[i]`[?]
- unused `[0, 1, 30, 31]`
- extra-`[i]`[?]
- ignore
- unk
- answer
- long-cat · short-cro[?]

**不确定项**:

- 「决策」：页眉亦见「沉淀」读法。
- 「单强 expert」：亦可能「更强 expert」。
- 「单词表 / 全词表」：与 OPD 全词表讨论一致；局部 OCR 曾误作单样本/全句。
- 「VLM 多轨迹」：亦见 PLLM / DLM；按本笔记多用 VLM 还原。
- special token 尖括号形式（如 `<|…|>`）笔迹不清，暂记 `[i]`。
- `long-cat · short-cro`：末词可能是 short-crop / short-cut / short-no。
- `num_workers`：原文笔误 `nums_workers`。

---

## Page 20 — b2c5-76a3

**主题**: 词元级 Clip 缓解 off-policy 偏差的 PPO 式目标，以及 Toggle 算法在 test-time 预算下交替优化推理长度。

**转录（纠错后）**:

$K$ 个响应；$\pi_{old}$；$\{y_1,\ldots,y_K\}$

- $y^j_{0:i}$：第 $j$ 个 response、第 $i$ 个词元的前缀序列
- $\bar{r}(x)=\frac{1}{K}\sum_{j=1}^{K} r(x,y_j)$（奖励均值）
- $N=\sum_{j=1}^{K}|y_j|$（总词元数）

**词元裁剪机制**：缓解 off-policy 偏差 ⇒ 梯度掩码/截断  
（新旧策略概率比；对数比落在 $[\alpha,\beta]$ 正常，之外清 0）↑ 训练稳定性

$$
\mathbb{E}_{x\sim D}\Bigg[\frac{1}{N}\sum_{j=1}^{K}\sum_{i=1}^{|y_j|}
\mathrm{Clip}\Big(\frac{\pi_\theta(y^j_i\mid x,y^j_{0:i})}{\pi_{old}(y^j_i\mid x,y^j_{0:i})},\alpha,\beta\Big)
\cdot\big(r(x,y_j)-\bar{r}(x)\big)
-\tau\Big(\log\frac{\pi_\theta}{\pi_{old}}\Big)^2\Bigg]
$$

**test-time 缩放**：简洁思维链，避免不必要词元扩张；但可能长度过拟合。

**Toggle 算法**：推理时缩放，预算约束优化，交替训练。

$\tilde{r}(x,y)=$  
- Phase 0（$\lfloor t/m\rfloor \bmod 2=0$）：  
  $r(x,y)\cdot\mathbb{I}\{\frac{1}{K}\sum_j r(x,y_j)<\lambda\ \mathrm{or}\ |y|\le\mathrm{budget}(x)\}$  
- Phase 1：$\tilde{r}=r(x,y)$（平均 acc $>\lambda$ 才在 budget 内执行；允许更多词元预算）

$\mathrm{budget}(x)=\mathrm{Percentile}\big(\{|y_j|\mid r(x,y_j)=1\},\ \alpha[?]\big)$

- 双目标：推理能力与交替优化
- ↓ 25%–30% token
- 随机交替优化方法

**不确定项**:

- 原文奖励下标偶见 $y_i$ / $y_j$ 混用，转录已按求和指标统一为 $y_j$。
- 「梯度掩码」与「梯度截断」两种读法；框注为区间外清 0，偏 mask。
- Percentile 第二参数：$\alpha$ 或其它分位值，笔迹淡。
- Phase 条件旁注「平均 acc >> / > λ」符号略糊。

---

## Page 21 — b39f-77f1

**主题**: 分类器/STL 约束引导下的规划奖励与结构化推理（DriveLM / Reason2Drive），以及自适应 Latent-CoT（ThinkDrive）与扩散规划要点。

**转录（纠错后）**:

- 分类器引导 Classifier Guidance
- 逻辑约束（e.g. STL）注入规则：Gen-Drive, DiAD[?]
- Planning Acc.；Action-Weighted Reward（如时间加权）；Planning Diversity（规划多样性）；format

**结构化推理**

- DriveLM-01：感知、预测、规划（logical）
- Reason2Drive：对象 / 场景
- 评分函数 Reasoning Alignment Score
- RA · RD · MS → 准确 · 多样 · 匹配

**Latent-CoT-Drive?**

- Adaptive & efficient CoT · ThinkDrive  
  - difficulty estimator →  
    - easy：浅层推理 → 直接 output action  
    - 高熵、不确定 → 自适应深度 CoT  
  - 课程学习
- Soft[?] Budgeter
- Fast DriveCoT：并行子任务解码
- 视觉 CoT · FSDrive：预测未来帧再规划  
  - 避免压缩成一维时的信息丢失（融合时空、平面[?]）

**扩散** `{`

1. ① 生成评估：RL 微调
2. ② 联合感知–规划：DiAD[?] Diffusion Planner
3. ③ 逻辑约束引导：STL（Signal Temporal Logic）。计算违背规则的程度、回传梯度

`}`

**不确定项**:

- `DiAD` / `DiAAD` / `DiffAD` / `DiAAP`：扩散规划专名笔迹多变，保留疑标。
- `Soft Budgeter`：亦可能 Step Budgeter / Self Budgeter。
- 「平面」：亦见「平场」读法。
- 分组标题「扩散」：初读曾误作「打散」，按 Diffusion Planner 语境取「扩散」。
- DriveLM-01：亦可能写作 DriveLM-1。
