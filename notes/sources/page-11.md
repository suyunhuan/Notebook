# 第 11 页：多模态 Dropout、Spatial PE、Reward(CoC)

- **原图**：[`images/01a169b3-1e55-7f1a-9cdf-72cd3d587bcc.jpg`](images/01a169b3-1e55-7f1a-9cdf-72cd3d587bcc.jpg)
- **主题标签**：Reward, PE, CoC

## 纠错说明

- CoC 保留（因果/决策链评分），勿一律改为 CoT
- TBT = Turn-By-Turn

## 校正转录

- 多模态 / patch dropout；Spatial PE：[x,y,dist,sin,yaw]
- Reward：CoC 质量（感知/推理/决策）；CoC+TBT 一致；决策·轨迹一致
- 裁判模型如 Qwen2.5-32B；注意 CoC 因果倒置


## 摘要

空间位置编码与基于 CoC 一致性的奖励模型设计。
