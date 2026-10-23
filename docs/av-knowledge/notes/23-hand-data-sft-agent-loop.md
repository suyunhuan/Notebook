# 手写笔记：数据多样性、Agent Loop、ViT 族与 CoC Reward

## 元信息

| 字段 | 内容 |
|------|------|
| 来源页码 | Handwriting Page 08–11（`_raw_batch_b.md` · a954 / aa0d / aad4 / abac） |
| 所属分类 | C. 数据与对齐 · C1/C2 + D. Agent 工程骨架 |
| 笔记类型 | 手写精炼（非豆包对话） |
| 相关原始图 | `../assets/handwriting/01a169b1-a954-*` … `01a169b1-abac-*` |

## 一句话摘要

SFT 要靠 **P1 多样性 + PV/BEV 组合与多模态 Dropout** 避免 P2 固化；训练基建是 **MessageBus→AgentLoop→Runner**；视觉骨干按任务选 DINO/DA3/FastViT/Qwen-ViT；对齐用 **CoC 质量与一致性 Reward**。

## 核心关键词

`VQA` · `SFT` · `P1/P2` · `meta-action` · `Polaris2` · `Agent Loop` · `MessageBus` · `DINO` · `DA3` · `Qwen-ViT` · `CoC` · `TBT` · `Reward Model` · `Spatial PE`

## 缩写与术语详解

| 缩写/词 | 全称 / 含义 | 在本页语境 |
|---------|-------------|------------|
| **VQA** | Visual Question Answering | 补视觉问答，加强 grounding |
| **SFT** | Supervised Fine-Tuning | 含 VQA 与 MA-Traj |
| **P1 / P2** | 两阶段提示/任务格式 | `P1+Q→A`（多样）；`P2+Q→MA-Traj`（易固化、继承不到 P1） |
| **MA** | Meta-Action | 与 Traj 组成决策–轨迹监督 |
| **FM** | Flow Matching 或 Foundation Model（手稿歧义） | `FM→MA·Traj` 数据驱动条件链 |
| **ViT** | Vision Transformer | 图像编码器族 |
| **DINO / DINOv2** | Self-distillation ViT | 无监督、单视角表征，再整合多视角 |
| **DA3** | 推断 Depth Anything 3 | 相对位置 + token 重排，偏多视角 3D |
| **FastViT** | 高效 ViT | BN + Depthwise 卷积，基于 DINOv2 |
| **Qwen-ViT** | Qwen 系视觉塔 | Conv 多模、旋转 PE（2D/3D/4D）、window attn、DeepStack、`hwt→adapter` |
| **DW** | Depthwise Convolution | FastViT 效率组件 |
| **hwt** | Height×Width×Time | 时空 token 进 adapter |
| **Polaris2** | HKU NLP 评测/数据相关（手稿） | high-overlap / low-difference 诊断收敛 |
| **C-EVAL** | 中文能力评测集 | 手稿称泛化提点约 1.4%（或 14%，存疑） |
| **CoC** | Chain of Causation（因果链，非普通 CoT） | 感知–推理–决策可验证叙述 |
| **TBT** | Turn-by-Turn | 与 CoC 导航一致性校验 |
| **Spatial PE** | Spatial Positional Encoding | `[x,y,dist,sin(yaw),cos(yaw),rel_ang]` |
| **MCP** | Model Context Protocol | Agent 工具接口之一 |
| **Agent Loop / Runner** | 会话编排 vs LLM+工具循环 | 见下节架构 |

## 知识脉络

### 1. 数据与 SFT（Page 08）

**目标**：loss↓ 时加数据、加 grounding 正样本；用 replay / dropout / Augment / 改写保多样性。

**PV/BEV 组合表**（四象限都要见）：

| PV | BEV | 作用 |
|----|-----|------|
| ✓ | ✓ | 正常 |
| ✓ | ✗ | 缺 BEV 仍可答 |
| ✗ | ✓ | 缺 PV 仍可答 |
| ✗ | ✗ | 极端鲁棒 / 纯语言条件 |

**P1→P2 陷阱**：P2 的 MA-Traj 格式固化 → **继承不到 P1 多样性**；需人设/prompt/segment 与 π0.7 级泛化数据。

**Traj 质量**：把 RL reward 思路放进 SFT（似 loss weight），训练时区分好坏、推好样本。

**扰动集**：`{bev, pv, ViT 图像, crop, random}`。

**收敛诊断（Polaris2）**：选项高重叠低差异 → 看 smooth 是否降；可能 **lr 过大 / 任务过易导致过早收敛**；「数字对建模难」。

### 2. Agent 调用闭环（Page 09–10）

```
Telegram / Webhook / CLI / WebUI / Discord / 飞书 / 微信
  → MessageBus（Inbound / Outbound）
  → Agent Loop：选 session；加载 skills · memory · 历史
  → Context Builder：system prompt + 上下文
  → Agent Runner：LLM Provider + 工具循环（文件/shell/Web/MCP/Cron）
  → restore → compact → build → run → save → respond → done
```

| 组件 | 职责 |
|------|------|
| **Agent Loop** | 会话、工作流、上下文、Hook、进度推送 |
| **Agent Runner** | 模型调用、工具循环、流式输出、终止条件 |
| **Memory + Dream** | 长期记忆（手稿） |

### 3. ViT 族选型速记（Page 10）

| 模型 | 特点 | 适用画像 |
|------|------|----------|
| DINO | 无监督、单视角→再融合 | 表征预训练 |
| DA3 | 相对位置、多视角 3D | 几何/深度友好 |
| FastViT | BN+DW，偏部署 | 低延迟 |
| Qwen-ViT | 旋转 PE 多维、DeepStack、adapter | 原生多模 VLM |

注意：全局注意力在高分辨率代价↑↑；可学习绝对 PE 绑死分辨率。

### 4. Dropout、Spatial PE 与 CoC Reward（Page 11）

- **多模态 Dropout / patch drop**：2D 随机丢弃，逼跨模态。
- **Spatial PE**：把自车/目标几何显式写入 token。

**Reward Model 打分项**

1. **CoC 质量**：感知 / 推理 / 决策 各 0–5；可用 CoC 让 LLM 出 Yes/No 题。  
2. **CoC ↔ TBT 一致**。  
3. **决策 ↔ 轨迹一致**（车道线、障碍约束）。  
4. **CoC ↔ 决策**。  
5. **CoC 因果倒置**（惩罚把结果当原因）。

## 知识点校验

1. **Correct**：阶段格式固化导致多样性丢失、模态 dropout/组合训练提升鲁棒、CoC 多轴一致性做 reward，均与 [07](07-e2e-data-tricks.md)/[08](08-vla-ambiguity-coc.md)/[12](12-planning-dpo-rl.md) 同向。
2. **Caution**：C-EVAL「1.4%」或为「14%」笔迹；Polaris2 选项标记 `[Av]/[Ax]` 含义未定义；DA3 为推断命名。
3. **Caution**：Agent 骨架是通用 LLM 产品架构，**不等于车载实时 VLA 运行时**；勿混部署域。
4. **Expand**：P1 多样性如何课程式注入 P2；CoC 五维 reward 的加权与人标 IAA；Qwen-ViT `hwt→adapter` 与 4D 动态帧率对齐。

## 与其他主题的关联

- [07 端到端数据 Trick](07-e2e-data-tricks.md) — 多样性、配比、清洗
- [08 VLA 歧义与 CoC](08-vla-ambiguity-coc.md) — CoC / meta-action 口径
- [09 Alpamayo Reasoning-VQA](09-alpamayo-reasoning-vqa.md) — Reasoning-VQA
- [12 Planning DPO/RL](12-planning-dpo-rl.md) — reward 与偏好对齐
- [01 VLM 训练阶段](01-vlm-training-stages.md) — SFT/RL 阶段位置
- [17 持续学习](17-continual-learning-vla.md) — replay 抗遗忘
