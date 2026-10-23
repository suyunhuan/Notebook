# Know-how：流式 4D 感知

## 为什么需要流式？

离线短 clip（如 2s）训练的模型在车端会长时运行。若不维护状态，每帧全量重算会导致：

1. 时序不一致（闪烁、ID 跳变）；  
2. 算力浪费；  
3. 无法利用「刚刚发生」的导航与灯态。  

流式方案用 **State + KV Cache +（可选）SSM** 把历史压成可更新状态。

---

## 核心组件

### 1. 2D / 4D Adaptor

- **2D adaptor**：承接图像–语言对齐（笔记中的 LiT 双视角），提供语义空间特征。  
- **4D adaptor**：多视角时空 embedding，携带几何与运动。  
- 融合：`add & cat` + **Cross-Attention**，让语言/语义 query 视觉，或反之。

### 2. Temporal → Spatial Attention

典型顺序（页 17）：

```
Cam → PE(+内外参) → Frame Attn → Temp Attn → token compress(+ST-PE) → Spatial Attn
```

先在时间维聚合，再压缩 token，最后做空间推理——比「直接全局 Attn」更省。

### 3. SSM 与 Attn 混用

- **Attn**：任意位置交互，适合关键帧、灯态突变。  
- **SSM**（如 Mamba 类）：线性复杂度长程状态，适合稳定背景与慢变几何。  
- **Router**：在专家/路径间分流（静态几何走 SSM，动态目标走 Attn 等）。

### 4. 时间规格

示例：`6s` 上下文，`2s` 段，`2Hz`。含义可以是：

- 状态保留约 6 秒历史；  
- 每 2 秒做一次较重的全局刷新；  
- 推理频率 2Hz（与 Sparse BEV / Scene token 一致）。  

必须与控制与传感器配时（10Hz/2Hz）对齐，否则会出现「特征旧半拍」。

---

## 多任务头

Ray / Depth / Scene Flow / Camera / Scale / 规划头：

- 几何头提供可度量空间（metric depth、ray）；  
- Flow 提供动态；  
- 规划头倒逼表征「可规划」。  

训练不稳时：ray map 可仅作教师，主任务用 depth+seg。

---

## 落地建议

1. 先做 **短窗流式 POC**（0.3s→1s），再拉长。  
2. 监控：时间一致性、ID switch、延迟、显存中 KV 大小。  
3. 鱼眼必须加内外参与合适 PE，否则 4D 几何漂。  
4. 与 FM 对接时，把流式输出变成固定长度 token（压缩），而不是变长原始特征。  
