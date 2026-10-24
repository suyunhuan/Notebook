# 手写笔记：LLM→VL→4D 多阶段训练管线

## 元信息

| 字段 | 内容 |
|------|------|
| 来源页码 | Handwriting Page 02（`_raw_batch_a.md` · a4bb-7ae7） |
| 所属分类 | B. VLA / 决策 · B1 训练阶段管线 |
| 笔记类型 | 手写精炼（非豆包对话） |
| 相关原始图 | `../assets/handwriting/01a169b1-a4bb-7ae7-*.jpg` |

## 一句话摘要

主链路是 **LLM → VL-stage1/2/2.5**，自 stage2 起并行接入 **4D Encoder 预训练与 projector 对齐**，再经 2D/4D 混训 SFT 与 RL；规模信号约 **150M 2D + 110M 4D**。

## 核心关键词

`LLM` · `VL-stage` · `4D Encoder` · `Projector` · `CLIP-like` · `AR` · `Diffusion` · `SFT` · `RL` · `混训`

## 缩写与术语详解

| 缩写/词 | 全称 / 含义 | 在本页语境 |
|---------|-------------|------------|
| **LLM** | Large Language Model | 文本底座，后续阶段逐步接入视觉 |
| **VL** | Vision-Language | 图文对齐阶段 |
| **4D** | 三维空间 + 时间 | 环视/视频时空表征（非单纯 RGB 帧） |
| **Projector / Connector** | 视觉或 4D token → LLM 嵌入的线性/MLP 桥 | stage1 常「只训 projector」保护主干 |
| **AR** | Autoregressive | 自回归建模范式（next-token） |
| **Diffusion / FM** | 扩散 / Flow Matching | 与 AR 并列的连续生成范式 |
| **SFT** | Supervised Fine-Tuning | 指令/轨迹监督微调 |
| **RL** | Reinforcement Learning | 后期偏好/闭环强化 |
| **CLIP-like** | Contrastive Language–Image Pretraining 类对比预训练 | 4D Encoder 离线对比预训练口径 |

## 知识脉络

### 1. 三维杠杆

- **结构**：Transformer  
- **规模**：model / data / compute  
- **范式**：AR **或** Diffusion（可并存于不同头）  
- 手稿划掉「离线→在线」：强调阶段划分，而非简单离线/在线二元。

### 2. 主链路 + 4D 并行支路

```
LLM
 └─ VL-stage 1
 └─ VL-stage 2 ──┬─（并行）4D Encoder pre-train（~100M，CLIP-like 离线）
                 ├─ 4D-stage 1：~10M 4D，常仅训 4D projector
                 └─ 4D-stage 2：~100M 4D，训 4D encoder + projector
 └─ VL-stage 2.5（2D/4D）
 └─ 2D/4D SFT（全类混训、全量）
 └─ 2D/4D RL
```

### 3. 数据规模（手稿口径）

| 模态 | 量级 |
|------|------|
| 2D | ~150M |
| 4D | ~110M（含 pretrain 100M + stage 增量） |

### 4. 工程 knowhow

- **冻结阶梯**：先 projector → 再 encoder+projector → 再与 LLM/SFT 联训，降低表征崩塌。
- **2D/4D 全类混训**：防止只训 4D 后丢失通用 VQA/文档能力（与 [01](01-vlm-training-stages.md) replay 思想一致）。
- 「仅训」字迹存疑：按 LLaVA 惯例推断；落地需看实际解冻表。

## 知识点校验

1. **Correct**：先对齐连接器、再解冻编码器，是多模态训练稳定策略；4D 从 VL-stage2 并行接入合理。
2. **Caution**：150M/110M/100M/10M 均为手稿工程量级，**无公开复现协议**；勿直接对比外部论文数据规模。
3. **Caution**：「仅训 projector / encoder」依赖笔迹推断；实际可能 LoRA 或全量——以训练配置单为准。
4. **Expand**：AR vs Diffusion 在「理解头 vs 动作头」如何分工；stage2.5 与 SFT 的数据配比、动态帧率（6s@2Hz / 2s@10Hz，见 [01](01-vlm-training-stages.md)）如何嵌入本管线。

## 与其他主题的关联

- [01 VLM 训练阶段](01-vlm-training-stages.md) — 解冻顺序与动态帧率的完整表
- [16 Qwen3-VL VLA](16-qwen3vl-vla-architecture.md) — 4D 感知插件与分阶段防崩
- [04 训练效率](04-training-efficiency.md) — 大规模 compute 落地
- [17 持续学习](17-continual-learning-vla.md) — 混训/replay 抗遗忘
- [07 端到端数据 Trick](07-e2e-data-tricks.md) — 垂域数据配比
