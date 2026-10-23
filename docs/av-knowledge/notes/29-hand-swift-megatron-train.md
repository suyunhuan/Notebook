# Swift LoRA 与 Megatron 并行训练流水线

## 元信息

| 字段 | 内容 |
|------|------|
| 来源 | 手写笔记 Page 25–26（`handwriting/_raw_batch_d.md`） |
| 资产 ID | `b73a-75bf` · `b7ef-77c8` |
| 所属分类 | 大模型技术 / 训练工程 |
| 关联原始页 | Batch D：Swift（小于 7B）LoRA；大于 7B Megatron；ModuleWrapper 流水线 |

## 一句话摘要

小于 7B 用 Swift LoRA（约 1% 参数）+ DDP/FSDP 快速微调；大于 7B–100B 切 Megatron 的 TP/PP/DP（及 EP/CP）三维并行，配 flash attn、重计算与通信重叠，按 env→DataLoader→训评流水线落地。

## 核心关键词

`ms-swift` · `LoRA` · `DDP/FSDP` · `Megatron` · `TP` · `PP` · `DP` · `EP/CP` · `sequence parallel` · `recompute` · `COC(通信)` · `ModuleWrapper` · `AMP/bf16`

## 知识脉络

### 1. 并行范式名词（必记）

| 缩写 | 全称 | 切什么 | 典型用途 |
|------|------|--------|----------|
| **DP** | Data Parallel | 数据/batch | 每卡一份（或分片）模型，不同 micro-batch |
| **TP** | Tensor Parallel | 层内矩阵按列/行 | 单层太大，多卡算一次 MatMul |
| **PP** | Pipeline Parallel | 层间切 stage | 深层网络流水线，注意 **bubble** |
| **EP** | Expert Parallel | MoE expert | 专家分卡 |
| **CP** | Context/Sequence Parallel 类 | 长序列分片 | 超长上下文分摊激活 |
| **SP** | Sequence Parallel | 常与 TP 配套 | 再分序列维激活 |

另：**DDP** = 常规数据并行；**FSDP** ≈ ZeRO 式参数/梯度/优化器分片（见 note 04/15）。

选型经验（笔记）：Swift 路径强调 **DDP/FSDP**；Megatron 路径上齐 **TP/PP/DP/EP/CP**。

### 2. Swift：&lt;7B LoRA / 多模态上限

**LoRA（Low-Rank Adaptation）**：在冻结权重旁学低秩更新 \(\Delta W=BA\)，只训少量参数。

- 笔记配置：`rank=8`，`alpha=32` → 约 **~1% param** 可训。
- **target modules** 由窄到宽：`q+v` → `qkv o` → `qkv o gate up down` → `all-linear`；多模态另有 **aligner / merger**。
- 视觉上限备忘：`max pixel 16777216`；`max img tokens 16384`；video 帧数约 50–500（笔迹存疑）。
- 精度：fp32 / **bf16** / fp16；损失以 **CE** 为主（另见 channel loss）。
- 适合：指令跟随、多模态 SFT、快速实验；非万亿预训练。

### 3. Megatron：&gt;7B–100B

必备旋钮：

- `micro batch` vs `global batch`（global ≈ micro × accum × DP）
- **tensor / pipeline / sequence parallel**
- **packing**、**flash attention**、**recompute**（激活重计算）
- **COC（Communication-Overlap-Computation）**：通信与计算重叠，降气泡与等待——**勿与智驾「CoC 因果链」混淆**（同形异义）
- **lazy tokenize**：延迟 tokenize 抬吞吐
- 精度旁注：bf16 / fp16

### 4. 超参与 ModuleWrapper 流水线

**超参备忘**：`lr=2e-6`；`max length 8192`；bfloat16；`image token ~1150`；`video token ~128`；`max frames 16`；gradient clip / gradient checkpoint。

**ModuleWrapper 职责**：

| 方法 | 作用 |
|------|------|
| `networks` | 组装/持有网络 |
| `sample` | 数据采样 |
| `net output` | 前向 |
| `update` | 参数更新 |

**流程**：

```
env 初始化 → 加载模型 → Dataset · preprocess · DataLoader
  → 训练（forward · loss · backward · param update）
  → 保存和评估
```

联训时额外盯 **DiT loss**（与 note 28 的 PP/AR 权重一致）。

### 5. 工程 Know-how

- &lt;7B 先 LoRA 验证数据与模板，再决定是否全参或上 Megatron。
- TP 增大通信；PP 增大 bubble——用 micro-batch 与 COC 填流水线。
- 多模态 token 预算（image/video）必须与 `max length` 同账，否则截断静默伤规划 CoT。
- 训推 **chat template / special token** 与 OPD 页（note 27）对齐。

## 知识点校验

1. **Correct**：Swift LoRA 做小模型微调、Megatron 三维并行做大模型，是工业主流分工；LoRA rank/alpha、flash attn、recompute、通信重叠均为标准手段。
2. **Caution**：本页 **COC=通信优化**，与 note 08/28 的 **CoC/COC 推理** 无关，阅读与检索时必须分词义。
3. **Caution**：`video 50/500`、`video token 128/188` 等数字笔迹不清；且 `lr=2e-6` 偏保守，换模型族需重扫。
4. **Expand**：EP/CP 与具体 MoE/长上下文实现绑定，笔记只列名，落地要对照 Megatron-LM / Megatron-Core 版本文档。
5. **Expand**：DiT loss 与 LLM CE 的梯度比例、是否同一 Optimizer，应与 note 28 权重表写成同一配置源。

## 与其他主题的关联

- [04 训练效率](04-training-efficiency.md) — FlashAttn / ZeRO / TP·PP
- [06 AdamW](06-adamw-optimizer.md) — 低 lr、clip、稳定训
- [15 LLM 训推栈](15-llm-train-infer-stack.md) — Swift/Megatron/vLLM 总览
- [01 训练阶段](01-vlm-training-stages.md) — 何时 LoRA、何时全参
- [03 OPD](03-online-policy-distillation.md) — 蒸馏阶段同样走 Swift/Megatron
- [28 档位与 DiT](28-hand-model-tiers-fm-dit.md) — 1.8B+ 档位依赖本篇并行
- [19 多阶段训练](19-hand-vl-4d-training-pipeline.md) — 阶段切换时用本篇框架训
- [23 Agent Loop 基建](23-hand-data-sft-agent-loop.md) — 训练编排与 Runner
