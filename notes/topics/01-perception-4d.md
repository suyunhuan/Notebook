# 主题：感知与 4D / 流式表征

关联页：04, 05, 13, 16, 17, 18, 27, 28

---

## 1. 问题定义

自动驾驶感知要从多相机、鱼眼、Lidar/Radar、导航信号中，得到**时空一致、可规划可用**的表征，并逐步从「多任务头堆叠」演进到「4D 流式基模 + 辅助任务」。

---

## 2. 经典栈（仍是落地基线）

```
Cam (透视/柱面) ──► FV Backbone ──► LSS ──► Cam BEV
RL/点云 ──► Voxel + UNet ──► RL BEV ─────────► Fusion BEV
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
                 Lane/边界                 GOD/Sparse Det           Driveable/Seg
```

要点：

- **PV 任务**：检测、语义 freespace、路面边缘；维护白名单（减速带、斑马线、车道线等）。
- **时序**：历史 hidden + 红绿灯特征；Cognition / GOD temporal。
- **BEV 检测**：CenterNet 式 heatmap → 中心解码 → offset/size/rot/vel；按速度与 VRU 加权。

---

## 3. 4D Encoder 与多模融合

4D = 空间 3D + 时间。笔记中的标准输出：

| 输出 | 作用 |
|------|------|
| Depth / Metric Depth | 几何尺度；质疑「仅 depth 是否足够」 |
| Ray Map | 射线/占据监督；训练可能不稳，常作正解辅助 |
| Semantic / PV Seg | 语义；开源 DA3 缺几何语义时需补 Seg |
| Scene Flow | 动态；与 Occ Flow / SDF 配合 |
| Sparse BEV | 长时序高效表示（如 6 帧 @ 2Hz） |

多模路径示例（页 27）：

- 环视+鱼眼：轻量 ConvNet + 3D PE → 降采样 token  
- 前视：DinoV2（可蒸馏）+ 3D PE  
- Lidar/Radar：UNet → DownConv  
- Navi SD+TBT：Navi Encoder（可从高精导航大模型蒸馏）  
- Flatten ≈ **1000 tokens** + Ego → FM  

---

## 4. 流式架构（页 13）

关键设计：

1. **2D adaptor**（文本/LiT 双视角）与 **4D adaptor**（多视角 streaming embedding）经 add&cat + Cross-Attn 融合。  
2. 多相机 Image Patch → Temporal Attn → Attn。  
3. 核心状态机：**Attn + SSM + State**，配合 **KV Cache**。  
4. 时间规格示例：`6s` 上下文，`2s` 段，`2Hz`。  
5. 任务头：Ray / Depth / Scene Flow / Camera / Scale / 规划相关头。

流式相对离线 clip 的优势：在线更新状态、复用 KV、对齐车端实时频率。

---

## 5. Token 压缩与长时序

| 机制 | 笔记规格 | 目的 |
|------|----------|------|
| PV token 压缩 | camera token 作 query，融合 Navi/Ego | 降序列长度，隐式监督 traj |
| Scene token | 64 tokens，6 帧 @ 2Hz | 场景级长时记忆 |
| Sparse BEV | 6 帧 @ 2Hz | 稀疏鸟瞰时序 |
| Occ Flow | 数秒窗口 + SDF，固定帧间隔 | 占据与距离场 |
| KV 28→3 | 中间层抽取 / 压缩 | 降访存 |

多阶段分辨率（具身/深度）：stage1 低分辨率定位 → stage2 高分辨率（如 1280）细化。

---

## 6. 演进路线（S0–S3）

```
S0 预训练 → S1 dense depth → S2 BGC/RGC 等 BEV 任务 → S3 E2E
```

并行的 VL+4D 大路线（页 02）：4D Encoder pretrain → projector → 2D/4D 混训 → SFT → RL。

---

## 7. 实践检查清单

- [ ] 内外参是否注入 PE / 矫正（鱼眼必备）  
- [ ] metric depth 是否可直出，还是仅相对深度  
- [ ] ray map 是否只做教师信号，避免拖垮主损失  
- [ ] 流式窗口与车端 Hz 是否一致（2Hz / 10Hz 配时）  
- [ ] 小障碍物是否有独立 tag / 动态 weight  
- [ ] token 预算是否落在 ~1k 量级以便进 FM/LLM  
