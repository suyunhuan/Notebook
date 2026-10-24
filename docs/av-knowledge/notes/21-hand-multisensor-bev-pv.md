# 手写笔记：多传感器 BEV/PV 感知全链路与检测头

## 元信息

| 字段 | 内容 |
|------|------|
| 来源页码 | Handwriting Page 04–05（`_raw_batch_a.md` · a659-7b66 / a71a-727f） |
| 所属分类 | A. 感知表征 · A1/A2 BEV 深度与 PV→BEV |
| 笔记类型 | 手写精炼（非豆包对话） |
| 相关原始图 | `../assets/handwriting/01a169b1-a659-7b66-*` · `01a169b1-a71a-727f-*` |

## 一句话摘要

相机经 **LSS** 得 BEV、雷达/激光体素得 BEV，融合后挂 **Cognition / GOD / SparseDet / PV 多头**；检测侧用 CenterNet 式高斯热力图 + 分场景 loss 加权。

## 核心关键词

`LSS` · `BEV` · `PV` · `GOD` · `Cognition` · `SparseDet` · `CenterNet` · `Focal Loss` · `NMS` · `GRU` · `FPN`

## 缩写与术语详解

| 缩写/词 | 全称 / 含义 | 在本页语境 |
|---------|-------------|------------|
| **LSS** | Lift-Splat-Shoot | 按深度分布把 PV 特征抬到 3D 再投影（splat）到 BEV |
| **BEV** | Bird’s-Eye View | 鸟瞰栅格特征，手稿 cam/rl bev 约 `(64,352,320)` |
| **PV / FV** | Perspective / Front View | 透视特征；含 persp fv、cyl（柱面）fv |
| **RL（传感器）** | Radar 或 LiDAR（手稿未写死） | voxelize → UNet → rl bev，再与 cam bev concat |
| **GOD** | General Occupancy / Drivable 类占用头（工业口语，全称团队相关） | 时序特征 → GOD head → 稠密占用输出 `(704,448,5)` 量级 |
| **Cognition** | 远距道路认知头 | 路口分割、车道/属性/bound、lane instance |
| **SparseDet** | 稀疏 Query 检测 | det query `(556,256)` → instance + center |
| **FPN** | Feature Pyramid Network | PV 多尺度检测/分割 |
| **NMS** | Non-Maximum Suppression | 后处理去重框 |
| **GRU** | Gated Recurrent Unit | 三路时序（G+c+s）融合历史 hidden |
| **ISD** | 手稿缩写（融合前级，全称未确认） | SD zoom → ISD fusion → PV Decoder → lane query |
| **VRU** | Vulnerable Road User | 行人/骑行者；匝道处 rotation loss 加权 |
| **CenterNet 式** | 中心点热力图检测范式 | Gaussian heatmap + 回归 offset/size/rot/vel |

## 知识脉络

### 1. 多传感器融合主干（Page 04）

```
Camera → fv-backbone → cam2 feat
       → LSS → persp/cyl fv → cam bev feat (64,352,320)
RL     → voxelize → UNet → rl bev feat (64,352,320)
       → concat → fusion bev (128,352,320)

并行 PV：fv/cyl 多尺度 + pos emb + crop → fusion pv feature
```

车道支路示意：`SD zoom → ISD fusion → PV Decoder → lane query (48,512)` → boundaries / exist / vis。

### 2. 多任务头

| 头 | 时序输入示意 | 产出 |
|----|--------------|------|
| **Cognition（远程超高）** | T-1 `(128,88,80)` + 当前 | 路口 seg、lane/属性/bound、instance `(48,512)` |
| **GOD** | T-1 `(128,88,56)` + 当前 | GOD output `(704,448,5)` |
| **Sparse BEV Det** | det query `(556,256)` | instance feat、Obj center |
| **PV 任务** | FPN 多尺度 | 检测/分割；白名单、锥桶、车道、lane merge、2D 人车、freespace、湿滑[?] |
| **Driveable** | driveable query `(112,256)` | 可行驶区域 instance |

后处理：concat → NMS → Box / 类别 / 背景。  
辅助：obj prev instance、static RG_mask、obj time/mask；历史 hidden；灯态播报。

### 3. CenterNet 式检测头（Page 05）

```
点云/BEV Feature C×H×W
  → Conv Head → K×H×W 概率图
  → Threshold + local max → center (u,v)
  → 回归：offset、size、rotation、vel、class、traj
  → NMS
```

**Anchor 生成器（手稿）**：约 `2500×6`、`(50×50)`、2 尺寸 × 2 height × 2 方向 `(0°,90°)`；另有 `188×188×12 = 424128` 量级描述——**两套数字并存，以实码为准**。

**Loss 加权经验**

| 条件 | 权重 |
|------|------|
| 某量 `< 0.05` | ×1.5 |
| `v > 0.05` | ×2 |
| 动静 | ×3 |
| 通用 rotation（车） | ×3 |
| 匝道 VRU rotation | 另有更严阈值[?] |

训练目标：**BEV focal loss** + 高斯中心热图，再回归 attributes。

### 4. 工程 knowhow

- Cam/RL BEV **同分辨率再 concat**，避免插值伪影。
- Cognition 与 GOD **裁不同空间范围**（88×80 vs 88×56），远距与占用算力拆分。
- Sparse query + 稠密 GOD 并存：实例与场互补。
- Loss 加权必须绑定物理量定义（速度？置信？）；手稿 `0.05` 未写单位。

## 知识点校验

1. **Correct**：LSS 相机 BEV + 点云/雷达 BEV 特征融合、再挂多头，与工业 BEVFusion 族一致；CenterNet 热图+回归是成熟检测配方。
2. **Caution**：张量形状、query 数、`RL` 是雷达还是激光、GOD/ISD 全称、lane vis `48` vs `748` 均为手稿快照，**禁止当接口文档**。
3. **Caution**：loss 阈值 `0.05` 对应物理量未写明；照抄权重会导致错误偏置。
4. **Expand**：三路 GRU 与 Sparse 历史 instance 如何统一时间对齐；PV 黑白名单与 NMS 类别优先级策略。

## 与其他主题的关联

- [13 BEV 深度模型](13-bev-depth-models.md) — LSS vs Query 路线
- [14 PV Backbone / View Transform](14-pv-backbone-view-transform.md) — Backbone–Neck–LSS
- [02 BEV+PNC 栈](02-bev-pnc-stack.md) — Cognition / GOD / 检测头全景
- [20 结构化感知](20-hand-structured-perception.md) — 本链路特征如何进 Navi/Traffic/Ego
