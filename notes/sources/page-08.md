# 第 08 页：路线 VQA / SFT 与轨迹注入

- **原图**：[`images/01a169b3-1c2c-7b36-8b6c-7c6426c301f7.jpg`](images/01a169b3-1c2c-7b36-8b6c-7c6426c301f7.jpg)
- **主题标签**：SFT, VQA, trajectory

## 纠错说明

- C-EVAL → C-Eval
- T=0.7 为采样温度

## 校正转录

- 补 VQA / 路线问答；Data-driven；加强 grounding SFT
- P1+Q→A（多样）vs P2+Q→MA·Traj（固定，难继承 P1）
- traj 注入 reward 进 SFT；loss weight 区分好坏
- 抖动：BEV / PV crop；attention dropout；过早收敛与 lr 诊断


## 摘要

用多样问答提升泛化，同时把轨迹质量以加权损失注入 SFT。
