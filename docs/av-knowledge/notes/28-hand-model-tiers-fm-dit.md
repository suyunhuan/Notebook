# 模型档位（Max/Ultra/Ultimate）、Loss 权重与 CrossAttn+DiT Waypoints

## 元信息

| 字段 | 内容 |
|------|------|
| 来源 | 手写笔记 Page 22–24（`handwriting/_raw_batch_d.md`） |
| 资产 ID | `b483-74fe` · `b56b-7052` · `b66b-7f2b` |
| 所属分类 | 智能驾驶核心技术 / 基础模型档位与动作头 |
| 关联原始页 | Batch D：档位表；lr/loss；Fm_feat → AZx3 → DiT waypoints |

## 一句话摘要

按 Max→Ultra→Ultimate 放大 FM/DiT/MLLM 容量，用分层 loss 权重联训，再经 Fm_feat 多层 hidden 的 Cross-Attn 条件化 DiT，输出规划 waypoints。

## 核心关键词

`Max/Ultra/Ultimate` · `FastViT` · `FM` · `DiT` · `MLLM` · `Causal Mask` · `loss 权重` · `AZx3 Cross Attn` · `waypoints` · `GRPO/PPO/DPO 监控`

## 知识脉络

### 1. 术语：FastViT 与 DiT

| 名词 | 解释 |
|------|------|
| **FastViT** | 混合 **CNN + ViT** 的高效视觉骨干；笔记规格约 **30M×7**（多相机/多塔），强调 **token 压缩**、可加载 vision 预训练或蒸馏加速；适合高分辨率（如 1920×1080）车载输入 |
| **DiT** | Diffusion Transformer：用 Transformer 块做扩散/流匹配去噪网络，生成动作或轨迹；全局条件常用 adaLN，结构化条件用 **Cross-Attn**（见 note 10） |
| **FM** | 此处为基础/融合模型（Foundation / Fusion Model），吃 PV/Lidar/Navi 等 token，再条件化 DiT 或接 MLLM |

### 2. 三档骨干与规模（笔记表）

| 档位 | 视觉/骨干要点 | FM 规模（约） | DiT 规模（约） |
|------|----------------|---------------|----------------|
| **Max** | 共享 BG，ResNet-18 级；~9M；**无 BEV project** | 768 · 18 层 · ~0.13B | 768·3 · ~0.05B |
| **Ultra** | ResNet 加深加宽（如 →ResNet-101）；FastViT 混合 | 1024 · 24 · ~0.35B | 1024·3 · ~0.1B |
| **Ultimate** | Text·cam×2·instruct → **FM→MLLM** → DiT；FM/MLLM ~2048·28·**1.8B** | 同上量级 | DiT 2048·3 · ~0.2B |

备选更大：1536·20·~0.7B。  
**Ultimate-MLLM**：约 40B token SFT 基模；理解·规划·**COC/CoT** → action；MOT 融合 VL·FM·WM；**Causal Mask** 控自回归依赖并利推理效率。

输入→FM→DiT 三路示例：

1. Vision（PV）`10Hz→2Hz` → FM → DiT（优化位置编码）
2. Lidar BEV（~0.3s）→ FM → DiT
3. Navi（SD/Navi）→ FM → DiT

### 3. Loss / lr 权重配置（联训）

笔记示意（相对权重，非绝对 lr）：

| 项 | lr 侧 | loss 侧 |
|----|-------|---------|
| llm | 0.1 | 0.1 |
| FM2LLM | 0.1 | 0（或关闭该 loss） |
| AZ | 1 | AR 1；**PP 0.05** |
| 其他 | — | total≈5；uvp；grad loss≈1.6 |

- **PP**：Path/Planning points loss，权重远小于 AR，避免轨迹项淹没语言/表征。
- **uvp**：与多视角→BEV（CaDDN/UVP 族）相关的辅助损失。
- 图示还有 DiT↔FM 双向与帧率 `10Hz→5Hz`（约 250ms 节拍）——规划更新频率低于感知。

### 4. Fm_feat → Cross-Attn → DiT waypoints

```
Fm_feat (B, N, 768)
  → MLP (768→1536→2048) → RMSNorm(2048)
hidden [0, 9, 18]
  → MLP (2048→1536→1536) → RMSNorm(1536)

AZx3 Cross Attn:
  K,V ← (3, B, seq, 1536)   # 三层 FM hidden
  Q   ← lm token Q (512 × 1536)
  出  → lm-head / m-head (512 × 1536)

Condition: lat+vision · point · time · ego
  → DiT ×2 → PP (5, 2) waypoints
```

工程读法：

- 从 FM 取多层 hidden（浅/中/深）作 **K/V**，语言模型侧 token 作 **Q**，做 **×3 Cross-Attn（AZx3）** 融合。
- 再附加 lat/vision、点、时间、ego 等条件，经 **两段 DiT** 出 **PP (5,2) waypoints**（例如 5 个点 × 2 维，或 5 步×2 通道——以代码为准）。
- 特殊 token 是否剥离再进 MLP，需与 chat template 一致。

### 5. RL 监控清单（GRPO · PPO · DPO）

监控：**KL**、温度/难度、**lr**、**reward hacking**、**collapse**；外加配置、常见问题、调试项。  
与 note 12/27 一致：对齐阶段必须盯策略漂移与奖励黑客，而非只看分数上涨。

## 知识点校验

1. **Correct**：小模型无 BEV 投影、大模型接 MLLM+CoT、轨迹用 DiT 条件生成，是清晰的档位分层；多层 hidden Cross-Attn 再 DiT 是合理的条件注入拆分（全局/语言走 attn，扩散走 DiT）。
2. **Caution**：表中 0.13B/0.35B/1.8B 与通道层数来自手写，可能含估算或划改（如 DiT 1024→2048）；上线以实际配置文件为准。
3. **Caution**：`FM2LLM loss=0` 与 `llm loss=0.1` 并存时，易出现「接口在训、对齐没训」或相反；需明确阶段（先对齐投影再联训）。
4. **Expand**：FastViT `30M×7` 是 7 相机独立塔还是时间维复制，原稿未闭合。
5. **Expand**：PP (5,2) 与更长 horizon / Codebook（note 11）如何切换，未说明。

## 与其他主题的关联

- [02 BEV+PNC](02-bev-pnc-stack.md) — DiT/AdaLN 在 PNC 中的位置
- [04 训练效率](04-training-efficiency.md) — 大档位显存与并行
- [10 Action Expert DiT](10-action-expert-dit.md) — adaLN vs Cross-Attn 条件
- [11 Trajectory Codebook](11-trajectory-codebook.md) — 离散动作替代路线
- [12 Planning DPO/RL](12-planning-dpo-rl.md) — GRPO/DPO 与监控
- [14 PV Backbone](14-pv-backbone-view-transform.md) — ResNet / 无 BEV project 的 Max 档
- [16 Qwen3-VL VLA](16-qwen3vl-vla-architecture.md) — MLLM+Action Expert
- [27 OPD/Toggle/STL](27-hand-opd-grpo-toggle-stl-cot.md) — 后训练与 CoT
- [29 Swift/Megatron](29-hand-swift-megatron-train.md) — 大档位怎么训
- [30 全栈](30-hand-fm-full-stack-architecture.md) — 传感器→FM 总图
- 可能的 **18+**：档位命名/产品线笔记若存在，以本篇规模表为技术锚点
