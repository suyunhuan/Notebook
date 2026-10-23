# Know-how：多模态 Token 化与融合

## 目标

把异构传感器变成 **长度可控、语义对齐、可进 Transformer/FM** 的 token 序列（笔记量级约 **1000**）。

---

## 四路 Tokenizer 模板（页 18）

| 源 | 模块 | 典型产出 |
|----|------|----------|
| Fisheye | 小 tokenizer（~20M） | BEV Token ~504 + PE |
| Camera | ResNet + 大注意力骨干 | PV Token（多相机×分辨率档）+ PE |
| SD + TBT | Navi tokenizer（~0.5M） | Navi token ~32 |
| Vehicle kinematics | Ego tokenizer（~0.5M） | Ego token |

共享 Encoder 后分叉：

- **BEV 头**：BZL / GOD / COG / Navi  
- **PV 头**：Seg / Depth  
- **Action Decoder**

---

## 融合模式

### 1. 早期融合（BEV）

LSS 把相机抬到 BEV，与 Lidar BEV 相加/拼接——适合检测与车道，但对语言接口不友好。

### 2. Token 级融合（进 FM/LLM）

Flatten 各模态 → 加模态/相机/时间 PE → Self-Attn。导航与 Ego 以**短 token**注入，避免淹没视觉。

### 3. Query 压缩（页 18 点 3）

用 **camera token 作 query**，交叉注意 Navi/Ego：

- 视觉决定「看什么」；  
- 导航/自车状态调制「怎么看」；  
- 可隐式监督轨迹，同时「压头」降算力。

### 4. 蒸馏对齐

- 前视 DinoV2 ← ConvNet 蒸馏；  
- Navi token ← 高精导航大模型蒸馏。  

教师提供几何/导航先验，学生保持部署尺寸。

---

## 位置编码要点

- **3D PE**：相机外参 + 深度/射线，使多相机可拼。  
- **Spatial PE**：`[x, y, dist, sin, yaw]`（页 11）。  
- **归一化 PE**（页 22）：跨分辨率/车型稳定。  
- 鱼眼与柱面分支分辨率不同，PE 必须可区分相机 ID。

---

## 长时序 Token

| 类型 | 规格例 | 用途 |
|------|--------|------|
| Sparse BEV | 6 帧 @ 2Hz | 动态障碍与车道记忆 |
| Scene token | 64 @ 6 帧 2Hz | 场景级摘要 |
| Occ Flow + SDF | 数秒、固定间隔 | 占据与可通行距离 |

Memory 策略：衰减但不删除关键事件（具身 Memory Bank 同理）。

---

## 常见坑

1. **Token 爆炸**：高分辨率多相机未压缩就进 LLM → 超延迟。  
2. **导航被淹没**：Navi 仅 32 token 但无足够 PE/损失权重。  
3. **时间错位**：2Hz token 与 10Hz 控制混用未标注时间戳。  
4. **只训 depth**：缺语义/雷达时，规划仍盲（页 14 的开放问题）。  

---

## 推荐顺序

1. 固定总 token 预算（如 1k）。  
2. 先通 Cam+Navi+Ego，再加 Lidar。  
3. 加压缩 query 与辅助 depth/BEV。  
4. 最后接 Action / DiT，用规划损失倒逼融合质量。  
