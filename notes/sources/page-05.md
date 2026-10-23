# 第 05 页：BEV 点云检测：Anchor 与 Conv Head

- **原图**：[`images/01a169b3-1a0c-7ee8-8df2-7d073291a860.jpg`](images/01a169b3-1a0c-7ee8-8df2-7d073291a860.jpg)
- **主题标签**：BEV检测, heatmap, NMS

## 纠错说明

- VRU = Vulnerable Road User
- CenterNet 风格中心点检测

## 校正转录

- Anchor generator：网格 × 尺寸 × height × 方向
- 点云 → Backbone → BEV Feature → **Conv Head** 概率图
- Thresholding → local max → decode center → 回归 offset/size/rotation/vel/class
- 损失加权：低速/高速、动静、车辆/VRU rotation 加权


## 摘要

BEV 上以 heatmap 定位中心再回归属性，按速度与类别加权损失。
