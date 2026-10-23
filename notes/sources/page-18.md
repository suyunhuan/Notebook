# 第 18 页：多 Tokenizer 端到端架构

- **原图**：[`images/01a169b3-234b-71c8-8597-973ccf875161.jpg`](images/01a169b3-234b-71c8-8597-973ccf875161.jpg)
- **主题标签**：E2E, tokenizer, Occ Flow

## 纠错说明

- Vehicle Kine → Kinematics
- Occ Flow = Occupancy Flow

## 校正转录

Fisheye→BEV token(504)；Camera→ResNet→PV token；SD+TBT→Navi(32)；Ego tokenizer
→ Backbone → BEV(BZL/GOD/COG/Navi) / PV(Seg/Depth) / Action Decoder
1 Occ Flow：5s + SDF；多帧间隔
2 Sparse BEV 长时序：6帧 2Hz
3 PV token 压缩：camera query 融合 Navi/Ego；隐式 traj
4 Scene token 长时序 64 @ 6帧 2Hz
动态 weight；小障碍物 tag；MetaAction；navi+道路类+距路口


## 摘要

四路 tokenizer 共享骨干，规定占用流与长时序 scene/BEV token 规格。
