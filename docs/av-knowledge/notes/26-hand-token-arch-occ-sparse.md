# 时序注意力、多路 Tokenizer 与 Occ / Sparse BEV

## 元信息

| 字段 | 内容 |
|------|------|
| 来源 | 手写笔记 Page 17–18（`handwriting/_raw_batch_c.md`） |
| 资产 ID | `b036-724c` · `b114-7a46` |
| 所属分类 | 智能驾驶核心技术 / 感知表征与 Token 架构 |
| 关联原始页 | Batch C：Cam 时序 Attn + 多模 4D；多 tokenizer → BEV/PV/Action |

## 一句话摘要

相机经 Frame→Temp→Spatial 注意力与 Cam–LiDAR 4D 融合，再由 Fisheye/Camera/Navi/Ego 多路 tokenizer 进共享 Transformer，输出 BEV/PV/Action，并以 Occ Flow 与 Sparse BEV 约定长时序规格。

## 核心关键词

`Frame/Temp/Spatial Attn` · `token compress` · `ray map` · `多模 4D` · `S0–S3` · `tokenizer` · `Occ Flow` · `Sparse BEV` · `Scene token` · `MetaAction` · `SD+TBT` · `DAgger`

## 知识脉络

### 1. 相机时序注意力管线

```
Cam → PE → Frame Attn → Temp Attn → token compress → Spatial Attn
         ↑ intrinsic / extrinsic
```

| 阶段 | 作用 |
|------|------|
| **PE** | 位置编码；常注入内外参（或 S1 阶段强化几何 PE） |
| **Frame Attn** | 单帧内 token 交互 |
| **Temp Attn** | 跨帧时序关联（运动、遮挡补全） |
| **token compress** | 降序列长度，控延迟与显存 |
| **Spatial Attn** | 空间结构 / 跨视角聚合 |

工程备注：

- 开源 **DA3（Depth Anything 3）** 缺几何语义时，补 **PVSeg**，并导出 **metric depth**。
- **内外参校验/约束** 与 **ray map 监督** 并列；笔记强调：用 ray map 解几何/pose 往往优于硬解某些量，但 **ray map 与 pose 联合优化易不稳定**。
- 流式 PoC 时长示例：0.3s → 1s；鱼眼相机单独接入。
- 训练数据侧：RL、**DAgger**、dense depth。

### 2. Ray Map（几何接口）

**Ray map**：把图像坐标映射为相机坐标系下的射线（方向向量，常再挂深度或占用 logit）。

- **用途**：监督深度/位姿、连接 PV 与 3D、作为 4D Encoder 输出之一（见 note 25）。
- **相对直接回归 pose**：监督信号更稠密、梯度路径更短；但与外参/pose 联合训时需锁标定或分阶段，否则振荡。
- **Know-how**：先 metric depth + 固定标定 → 再轻度微调外参相关项；避免一步端到端「全解」。

### 3. 多模 4D：Cam–LiDAR

```
Cam → Patch → Transformer → DPT Head → PVSeg
                ↑ (+)
   Lidar → Feat Extract
```

- LiDAR 特征加性/融合进视觉 Transformer，再出 PV 分割等。
- **训练阶段表（笔记）**：

| S0 | S1 | S2 | S3 |
|----|----|----|----|
| 预训练 | dense depth | BGC（B/G/C 多头） | E2E |

课程式：先几何与深度，再 BEV 业务头，最后端到端联训。

### 4. 多路 Tokenizer → 共享骨干

```
Fisheye → tokenizer @ ~20M ──→ BEV Token (504) ──PE─┐
Camera  → ResNet → PA3D @ ~0.4B → PV Token           │
          (约 220×2 + 60×5) ──PE─────────────────────┤→ Transformer
SD+TBT  → Navi tokenizer @ ~0.5M → Navi token 32 ────┤
Vehicle Kine → Ego tokenizer @ ~0.5M ────────────────┘
                                                      ├→ BEV { BZL, GOD, COG, Navi }
                                                      ├→ PV { Seg, Depth }
                                                      └→ Action Decoder
```

- **TBT**：Turn-by-Turn 导航；与 SD map 一起经 Navi tokenizer（约 32 token）。
- **Ego tokenizer**：车辆运动学状态（速度、加速度、转向等）离散/连续编码。
- **PV token 压缩**：camera token 作 query，融合 Navi、Ego；可隐式承载业务轨迹先验。
- **Navi 侧特征**：encode + rc / tsd / road class & dist 等做 cross（路口、标牌、道路等级与距离）。

### 5. Occ Flow、Sparse BEV、Scene token 规格

| 模块 | 规格（笔记） | 含义 |
|------|----------------|------|
| **Occ Flow** | 5s 窗口；Occ + Flow + **SDF**；0.6s 间隔 → **8 帧** | 占用 + 场景流 + 符号距离场，服务动态障碍与可通行性 |
| **Sparse BEV** | **6 帧 @ 2Hz** 长时序 | 稀疏 query/实例级 BEV，省显存、利长时关联 |
| **Scene token** | **64 个**；6 帧 @ 2Hz | 场景级摘要 token，供规划/LLM 条件 |

**Occ Flow 解释**：在占用栅格上同时预测占用状态、栅格/物体流动（Flow）与 SDF（到障碍表面的有符号距离），比单帧 Occ 更能支撑短时预测与碰撞约束。

**Sparse BEV**：相对稠密 BEV 网格，用少量 query 表征车道/障碍/地图要素，适合长时序与 VLA token 预算。

### 6. 数据与辅助信号

- 数据管线关键词：一阶段 / RL / DAgger / 小障碍物采集 / PNC。
- **动态 weight**；小障碍物结合 tag。
- **MetaAction 辅助**：结构化高层意图（如 keep/accel × 直行/变道），降低纯轨迹回归的歧义（与 note 08 的 meta-action 同族）。
- 与 note 25 的 SD+TBT、运动学旁路一致，形成「感知 token + 导航 token + ego token → Action」。

## 知识点校验

1. **Correct**：Frame→Temp→compress→Spatial 是典型的「先时序压缩再空间聚合」；多 tokenizer 共享骨干再分 BEV/PV/Action 头，符合多模态 VLA 感知侧工程结构。
2. **Caution**：PA3D 参数量有 `0.4B`/`0.9B`、PV token 计数有两种读法；BZL/GOD/COG 为内部头缩写，对外文档需映射正式任务名。
3. **Caution**：ray map 与 pose **联合训练不稳定**——不要默认「加 ray map 必涨点」；需分阶段与损失退火。
4. **Expand**：Occ Flow 的 SDF 与规划碰撞代价、STL 约束（note 27）如何共用同一几何场，值得显式打通。
5. **Expand**：Scene token 64 与 Navi 32、BEV 504 的注意力预算与车载延迟曲线未给出，需 profiling。

## 与其他主题的关联

- [02 BEV+PNC](02-bev-pnc-stack.md) — 时序 BEV、Occupancy、稀疏路线
- [07 数据 Trick](07-e2e-data-tricks.md) — DAgger、小障碍、动态权重
- [08 Meta-action / CoC](08-vla-ambiguity-coc.md) — MetaAction 语义落地
- [11 轨迹 Codebook](11-trajectory-codebook.md) — Action 离散化另一路径
- [13 BEV 深度](13-bev-depth-models.md) — LSS / Sparse4D / Occupancy
- [14 PV Backbone](14-pv-backbone-view-transform.md) — Backbone / View Transform
- [25 Video Thinking / LWM](25-hand-video-thinking-latent-wm.md) — 4D/Ray 与 latent rollout
- [30 全栈架构](30-hand-fm-full-stack-architecture.md) — 环视/前视/Lidar/Navi token 汇总
- [20 结构化感知 NaviNN](20-hand-structured-perception.md) — TBT/Navi 与本篇 Navi tokenizer
- [21 多传感器 BEV/PV](21-hand-multisensor-bev-pv.md) — GOD/Sparse 头与本篇 BEV 多头
- [23 数据/ViT 族/DA3](23-hand-data-sft-agent-loop.md) — DA3、FastViT、DAgger 数据侧
- [24 流式 4D](24-hand-scenegraph-streaming-4d.md) — 长时序 token 与 KV
