# 自动驾驶知识 · 分类体系

> 豆包 01–17 + 手写 18–30。按「学什么 / 建什么 / 怎么训 / 怎么落地」四层。

## 分类总览

| 大类 | 子类 | 笔记 | 定位 |
|------|------|------|------|
| A. 感知表征 | A1 BEV / 深度 / 多头 | [13](notes/13-bev-depth-models.md) · [21](notes/21-hand-multisensor-bev-pv.md) | LSS vs Query；融合多头 |
| | A2 PV→BEV 骨干 | [14](notes/14-pv-backbone-view-transform.md) | Backbone / View Transform |
| | A3 结构化中间层 | [02](notes/02-bev-pnc-stack.md) · [20](notes/20-hand-structured-perception.md) | Navi/Traffic/Ego；BEV+PNC |
| | A4 Token / Occ / Sparse | [26](notes/26-hand-token-arch-occ-sparse.md) · [30](notes/30-hand-fm-full-stack-architecture.md) | 多 tokenizer；Occ Flow |
| B. VLA / 决策 | B1 训练阶段管线 | [01](notes/01-vlm-training-stages.md) · [19](notes/19-hand-vl-4d-training-pipeline.md) | VLM→VLA；LLM→VL→4D |
| | B2 推理与因果 | [09](notes/09-alpamayo-reasoning-vqa.md) · [08](notes/08-vla-ambiguity-coc.md) · [22](notes/22-hand-ego-process-meta-action.md) · [27](notes/27-hand-opd-grpo-toggle-stl-cot.md) | CoC；STL；自适应 CoT |
| | B3 shortcut / 因果干预 | [05](notes/05-causal-modeling-trajectory.md) | C→A→T |
| | B4 动作表示 | [11](notes/11-trajectory-codebook.md) · [10](notes/10-action-expert-dit.md) · [28](notes/28-hand-model-tiers-fm-dit.md) | Codebook；DiT；档位 |
| | B5 系统架构 / WM | [16](notes/16-qwen3vl-vla-architecture.md) · [18](notes/18-hand-vla-l3l4-roadmap.md) · [24](notes/24-hand-scenegraph-streaming-4d.md) · [25](notes/25-hand-video-thinking-latent-wm.md) · [30](notes/30-hand-fm-full-stack-architecture.md) | L3/L4；流式 4D；Video Thinking |
| C. 数据与对齐 | C1 场景/多样性 | [07](notes/07-e2e-data-tricks.md) · [23](notes/23-hand-data-sft-agent-loop.md) | Trick；P1/P2；Agent |
| | C2 偏好与 RL | [12](notes/12-planning-dpo-rl.md) · [24](notes/24-hand-scenegraph-streaming-4d.md) · [27](notes/27-hand-opd-grpo-toggle-stl-cot.md) | DPO；Toggle；慢快 Rollout |
| | C3 蒸馏 | [03](notes/03-online-policy-distillation.md) · [27](notes/27-hand-opd-grpo-toggle-stl-cot.md) | OPD / MOPD |
| | C4 持续学习 | [17](notes/17-continual-learning-vla.md) | 抗遗忘 |
| D. 训练工程 | D1 优化器 | [06](notes/06-adamw-optimizer.md) | AdamW |
| | D2 效率 | [04](notes/04-training-efficiency.md) | FlashAttn / ZeRO |
| | D3 训推栈 / 缩放 | [15](notes/15-llm-train-infer-stack.md) · [29](notes/29-hand-swift-megatron-train.md) · [28](notes/28-hand-model-tiers-fm-dit.md) | Swift；Megatron；档位 |

## 按阅读角色

| 角色 | 建议顺序 |
|------|----------|
| 感知 | 14 → 13 → 21 → 20 → 26 → 30 → 02 |
| VLA / 多模态 | 18 → 01 → 19 → 09 → 08 → 22 → 11 → 10 → 25 → 16 → 30 |
| 规控 / 对齐 | 07 → 23 → 12 → 27 → 05 → 03 |
| 训练平台 | 15 → 29 → 04 → 06 → 28 → 17 |
| 系统架构师 | 18 → 30 → [整体体系](roadmap/INTEGRATED_SYSTEM.md) → 28 → 24 |

## 知识依赖（整合简图）

```
感知基础 (13/14/21)
    ↓
结构化中间层 (20) + Token/Occ (26)
    ↓
FM / 流式 4D (19/24/30) ──────┐
    ↓                         ↓
推理/CoC/STL (08/09/27)   数据/Agent (07/23)
    ↓                         ↓
动作 DiT/Codebook (10/11/28) ← OPD/DPO/Toggle (03/12/27)
    ↓
L3/L4 闭环与工程 (18/29/15)
```
