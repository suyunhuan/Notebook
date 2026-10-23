# Know-how：扩散规划与逻辑约束

## 为什么规划用 Diffusion / DiT？

轨迹是**连续、多模态**的：同一路口可左转缓行或等待。回归均值易「抹平」多模态；离散 AR 对连续坐标又偏粗。扩散模型从噪声迭代去噪，能覆盖多模态分布，并自然接入条件（地图、灯态、ego）。

笔记中的形态：

```
ego / time / point / FM tokens
        → DiT × 2..N
        → waypoints (e.g. 5×2) 或 Ego/Obj Traj 25×2
```

条件含 Noise（区分 ego/obj）与 timestamp（页 27）。

---

## 与 FM / LLM 的接口

1. **FM** 提供场景 token（~1k）。  
2. **Cross-Attn（AZ×3）**：lm token 作 Q，FM 特征作 K/V → 语言/意图对齐。  
3. **DiT** 读条件向量生成连续轨迹（Diffusion Policy）。  
4. 损失权重：轨迹相关头权重大（AZ/AR≈1），辅助 PP 较小（0.05）。  

也可 **Action Decoder 接 AR LWM**：先 latent rollout 再解码（页 16）——与扩散可并存（离散意图 + 连续轨迹）。

---

## 逻辑约束：STL

**Signal Temporal Logic** 把「红灯不可过停止线」等写成时序逻辑公式，可计算**违例程度**（连续松弛）并回传梯度。

与 Classifier Guidance 配合：

- 训练期：违例惩罚进损失；  
- 采样期：用引导偏移去噪方向。  

文献线：Gen-Drive、DIFFAD、Diffusion Planner（页 21）。

---

## 评测维度

| 指标 | 含义 | 笔记提醒 |
|------|------|----------|
| Planning Acc | 是否接近专家/安全 | Action-Weighted / 时间加权 |
| Diversity | 是否多模态 | 避免坍缩 |
| 规则违例率 | STL/交通规则 | 引导是否生效 |
| 舒适性 | 加加速度等 | 纵向 ARDP 风险降速 |

---

## 训练阶段建议

1. **模仿**：专家轨迹扩散（稳定）。  
2. **联合**：加感知/占据辅助，避免「瞎规划」。  
3. **RL 后训练**：用安全与进度奖励微调（防 hacking）。  
4. **约束**：打开 STL/分类引导，压违例。  

---

## 常见坑

- 只拟合平均轨迹 → 多样性指标假好、路口僵死。  
- 条件未含灯态/导航 → 扩散「自由发挥」。  
- 与感知不同频 → 用过期 BEV 规划。  
- 权重失衡：语言 loss 压过轨迹，或反之。  
