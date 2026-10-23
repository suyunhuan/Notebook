# 第 12 页：4D Scene Graph 与快慢 DPO

- **原图**：[`images/01a169b3-1f18-75e1-9aef-f99c4a014add.jpg`](images/01a169b3-1f18-75e1-9aef-f99c4a014add.jpg)
- **主题标签**：Scene Graph, DPO, VLM

## 纠错说明

- 通国 → 通过率
- TBT = Turn-By-Turn

## 校正转录

- 控制 RAG / 图 → VLM（盲区、四维、物体）→ 4D 时空动态打标
- 能力：物体语义、全局/局部、TBT 导航
- 慢/快 Rollouts → SFT → pairwise → DPO
- 指标：通过率 28.4%→74.31%；难例 35.9%→10.6%


## 摘要

用 4D 场景图增强 VLM 导航语义，经偏好优化显著提升通过率。
