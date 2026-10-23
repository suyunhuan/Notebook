# 自动驾驶对话知识 · 分类体系

> 基于 17 篇豆包对话梳理。分类按「学什么 / 建什么 / 怎么训 / 怎么落地」四层，便于按角色查阅。

## 分类总览

| 大类 | 子类 | 笔记 | 定位 |
|------|------|------|------|
| A. 感知表征 | A1 BEV 与深度 | [13](notes/13-bev-depth-models.md) | LSS vs Query 等路线对比 |
| | A2 PV→BEV 骨干 | [14](notes/14-pv-backbone-view-transform.md) | Backbone / Neck / View Transform |
| | A3 感知+规控栈全景 | [02](notes/02-bev-pnc-stack.md) | BEV + PNC + LLM 注入鸟瞰 |
| B. VLA / 决策 | B1 训练阶段管线 | [01](notes/01-vlm-training-stages.md) | VLM→VLA 分阶段解冻 |
| | B2 推理与因果链 | [09](notes/09-alpamayo-reasoning-vqa.md) · [08](notes/08-vla-ambiguity-coc.md) | Reasoning-VQA / CoC / 语义消歧 |
| | B3 因果与 shortcut | [05](notes/05-causal-modeling-trajectory.md) | C→A→T、do-探针 |
| | B4 动作表示 | [11](notes/11-trajectory-codebook.md) · [10](notes/10-action-expert-dit.md) | Codebook / DiT+adaLN |
| | B5 参考架构 | [16](notes/16-qwen3vl-vla-architecture.md) | Qwen3-VL-2B 插件式 VLA |
| C. 数据与对齐 | C1 场景构造/清洗 | [07](notes/07-e2e-data-tricks.md) | 长尾加权、构造、清洗闭环 |
| | C2 偏好与 RL | [12](notes/12-planning-dpo-rl.md) | 真值改造 → DPO → GRPO |
| | C3 蒸馏 | [03](notes/03-online-policy-distillation.md) | OPD / 强弱蒸馏 |
| | C4 持续学习 | [17](notes/17-continual-learning-vla.md) | Neural Thickets / 抗遗忘 |
| D. 训练与工程 | D1 优化器 | [06](notes/06-adamw-optimizer.md) | AdamW 原理与智驾调参 |
| | D2 训练效率 | [04](notes/04-training-efficiency.md) | FlashAttn / ZeRO / AMP |
| | D3 训练推理栈 | [15](notes/15-llm-train-infer-stack.md) | Swift / Megatron / vLLM |

## 按阅读角色

| 角色 | 建议顺序 |
|------|----------|
| 感知工程师 | 14 → 13 → 02 → 16（感知分支） |
| VLA / 多模态 | 01 → 09 → 08 → 11 → 10 → 16 |
| 规控 / 对齐 | 07 → 12 → 05 → 03 |
| 训练平台 | 15 → 04 → 06 → 17 |

## 知识依赖（简图）

```
感知基础 (A1/A2)
    ↓
BEV+PNC 栈 (A3) ──────────────┐
    ↓                         ↓
VLM 训练阶段 (B1) ←── 数据 Trick (C1)
    ↓                         ↓
推理/CoC (B2) ←── 因果 (B3)   DPO/RL (C2)
    ↓                         ↓
动作表示 Codebook/DiT (B4) ←── OPD (C3)
    ↓
完整 VLA 架构 (B5)
    ↓
持续学习 (C4) + 训练工程 (D*)
```

## 主题交叉矩阵（高频）

| | 数据配比 | 因果/CoC | Action Expert | 蒸馏/对齐 |
|--|---------|----------|---------------|-----------|
| 训练阶段 (01) | ● | ● | ● | ● |
| Alpamayo (09) | ○ | ● | ● | ○ |
| DPO 规控 (12) | ● | ○ | ● | ● |
| Qwen3-VL VLA (16) | ○ | ○ | ● | ○ |
| 数据 Trick (07) | ● | ○ | ○ | ○ |

● 强相关 · ○ 弱相关
