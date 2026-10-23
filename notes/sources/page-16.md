# 第 16 页：4D Encoder → MOT → Latent Reasoning

- **原图**：[`images/01a169b3-21e2-7305-8a27-377b26145800.jpg`](images/01a169b3-21e2-7305-8a27-377b26145800.jpg)
- **主题标签**：4D, MOT, Latent Reasoning, LWM

## 纠错说明

- God st → God state（特权状态）
- AR LWM = Auto-Regressive Large World Model

## 校正转录

① 4D Encoder → Depth / Ray Map / Semantic
② 多模4D+Lidar；Fisheye→Seg/Depth → MOT → Depth/Ray/PV seg/BEV(B,C,G)/Sparse BEV
③ SD+TBT + Vehicle Kinematics → BEV/PV/Sparse BEV/Action
Latent Reasoning：rollout 4 步约 2s；Large Transformer → World state（t+0.5…）→ Action Decoder → AR LWM


## 摘要

4D 感知经 MOT 与导航运动学到 Action，并用世界状态 rollout 做潜空间推理。
