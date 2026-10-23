# 第 23 页：多损失权重与 grad loss 监控

- **原图**：[`images/01a169b3-26fa-79e7-b69e-9582a51e6d68.jpg`](images/01a169b3-26fa-79e7-b69e-9582a51e6d68.jpg)
- **主题标签**：损失权重, 调参

## 纠错说明

- FM2LLM = FM→LLM 对齐损失
- AZ/AR/PP 为模块损失项

## 校正转录

llm 0.1；loss 0.1；FM2LLM 0.1；AZ 1；AR 1；PP 0.05；total≈5；grad loss≈1.6
示意图：DiT 条件化 FM


## 摘要

记录各头损失权重与当前梯度损失监控值。
