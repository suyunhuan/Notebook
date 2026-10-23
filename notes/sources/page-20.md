# 第 20 页：GRPO 目标与 Toggle Token 预算

- **原图**：[`images/01a169b3-24b2-76b1-80ba-c7131290f576.jpg`](images/01a169b3-24b2-76b1-80ba-c7131290f576.jpg)
- **主题标签**：GRPO, Toggle, Test-time scaling

## 纠错说明

- 词元截断缓解 off-policy；Toggle 交替压缩长度与放开推理

## 校正转录

K 响应；优势 r−r̄；Clip(πθ/πold,α,β)；τ(log ratio)² 惩罚
Toggle：phase0 在均值奖励阈值/budget 内收紧；phase1 放开
budget = 正确样本长度分位数；目标 ↓25%–30% tokens


## 摘要

组相对策略优化加长度 Toggle，在能力与简洁 CoT 间交替优化。
