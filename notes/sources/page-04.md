# 第 04 页：多传感器感知多任务大图

- **原图**：[`images/01a169b3-1950-75ae-9fb5-1242fa092ecb.jpg`](images/01a169b3-1950-75ae-9fb5-1242fa092ecb.jpg)
- **主题标签**：感知, BEV, Lane, GOD, Sparse Det

## 纠错说明

- pho/cyl：透视/柱面（鱼眼柱面）相机分支
- LSS = Lift-Splat-Shoot
- G+C+S：疑似 GRU/门控 + Conv + Sigmoid 时序块

## 校正转录

### Backbone
- Cam（pho 7×3×704×1280；cyl 4×3×640×512）→ fv-backbone → LSS → cam bev feat
- RL → voxelize → UNet → rl bev feat → **fusion bev feat**
- SD zoom → ISD fusion → PV Decoder → Lane query

### PV 任务
检测 / 分割（semantic freespace、路面边缘）；白名单：减速带、斑马线、车道线、lane merge、地标等

### 时序与认知
- 历史 hidden + 红绿灯特征 → 多尺度时序块
- Cognition：road intersection / lane head
- GOD temporal head；Sparse DetHead；Driveable area head
- concat → NMS → Box / 类别 / 背景


## 摘要

经典自动驾驶感知栈：多相机+点云 BEV 融合，多任务头与时序/实例跟踪后处理。
