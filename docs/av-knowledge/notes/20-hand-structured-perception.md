# 手写笔记：感知结构化特征（NaviNN / TrafficNN / Ego-HMF）

## 元信息

| 字段 | 内容 |
|------|------|
| 来源页码 | Handwriting Page 03（`_raw_batch_a.md` · a59b-7cb0） |
| 所属分类 | A. 感知表征 · A3 感知+规控结构化中间层 |
| 笔记类型 | 手写精炼（非豆包对话） |
| 相关原始图 | `../assets/handwriting/01a169b1-a59b-7cb0-*.jpg` |

## 一句话摘要

导航（TBT/EHP）与 BEV/PV 时序经 **NaviNN→TrafficNN→Ego-HMF** 串级，产出导航场、路口通行概率与轨迹指导，再喂给 FSD 规划与 world model。

## 核心关键词

`NaviNN` · `TrafficNN` · `Ego-HMF` · `TBT` · `EHP` · `BEV` · `PV` · `导航场` · `通行概率` · `World Model`

## 缩写与术语详解

| 缩写/词 | 全称 / 含义 | 在本页语境 |
|---------|-------------|------------|
| **TBT** | Turn-by-Turn | 逐步导航指令（左转/出口/车道建议等） |
| **EHP** | Electronic Horizon Provider（电子地平线） | 前方道路拓扑/限速/坡度等远距导航先验（约数百米） |
| **BEV** | Bird’s-Eye View | 鸟瞰特征 |
| **PV** | Perspective View | 透视相机视角特征 |
| **NaviNN** | Navigation Neural Net（手稿命名） | 静态导航场编码器–解码器 |
| **TrafficNN** | Traffic-light Neural Net | 红绿灯/路口通行概率支路；注入 Navi 特征 |
| **Ego-HMF / Ego-HMP** | Ego Hidden/Motion Feature（全称手稿未写死） | 融合结构化 + Navi/Traffic 特征，解码轨迹指导 |
| **FSD** | Full Self-Driving（产品口语） | 下游规划消费方 |
| **World Model** | 世界模型 | 占用/速度场等预测，与结构化场联动 |

## 知识脉络

### 1. 串级逻辑（远→近）

```
全局导航（EHP/TBT，远端 ~500m）
  → NaviNN：静态导航场
  → TrafficNN：路口通行概率（注入 Navi Feat）
  → Ego-HMF：轨迹指导
  → FSD 规划 / 轨迹预测 / 占据场 / 速度场 / 灯态概率
  →（可选）World Model
```

侧注：**全局→局部** 使轨迹监督更「有效」——先有可行廊道，再谈细轨迹。

### 2. 三模块 IO

| 模块 | 输入 | 输出 |
|------|------|------|
| **NaviNN** | 场景关系；BEV、PV 时序 | Encoder→Feat→Decoder(+query points)→**导航场** |
| **TrafficNN** | 灯态模糊特征；BEV、PV 时序；**Navi Feat** | **路口通行概率** |
| **Ego-HMF** | BEV/PV 时序；结构化；Traffic+Navi Feat | Ego 特征 → Decoder →**轨迹指导** |

### 3. 工程 knowhow

- Navi 特征注入 Traffic：灯态决策受导航廊道约束（该不该进交叉口）。
- Query points 解码导航场：类似稀疏查询，利于远距稀疏监督。
- 结构化中间量（场/概率）比直接回归轨迹更可解释、可校验。

## 知识点校验

1. **Correct**：导航先验（TBT/EHP）与感知时序解耦再串级融合，是工业栈常见分层。
2. **Caution**：Ego-HMF / Ego-HMP 全称与是否同一模块未确认；「孤立场景关系」或为「关联」笔误。
3. **Caution**：远端 500m、各 Feat 维度未给出实现规格，不可当 API 契约。
4. **Expand**：导航场与占据/速度场的坐标与分辨率对齐；灯态不确定度如何进入规划代价。

## 与其他主题的关联

- [02 BEV+PNC 栈](02-bev-pnc-stack.md) — cognition / 导航注入鸟瞰
- [09 Alpamayo Reasoning-VQA](09-alpamayo-reasoning-vqa.md) — 语言侧导航/决策条件
- [05 因果建模](05-causal-modeling-trajectory.md) — 结构化 A 再推 T
- [21 多传感器 BEV/PV](21-hand-multisensor-bev-pv.md) — 上游稠密感知特征从哪来
