# 第 27 页：环视/鱼眼/Lidar/Navi → FM → Diffusion

- **原图**：[`images/01a169b3-29e9-7932-b53e-c9f2e06bdd9e.jpg`](images/01a169b3-29e9-7932-b53e-c9f2e06bdd9e.jpg)
- **主题标签**：多传感器, FM, Diffusion

## 纠错说明

- DinoV2；高精导航大模型 distill
- FM ~0.4B×25 层

## 校正转录

环+鱼 ConvNet；前视 DinoV2+蒸馏；Lidar/Radar UNet；Navi token 蒸馏
flatten ≈1000 tokens + Ego → FM → BCG/PV&3D；PV2BEV
Diffusion(75M)：Ego/Obj traj；Noise+timestamp
FM block：Self-Attn+FFN ×25


## 摘要

量产向多传感器 token 化进 FM，再用扩散解码轨迹。
