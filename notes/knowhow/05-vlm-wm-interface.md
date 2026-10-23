# Know-how：VLM × World Model 接口

## 分工

| | VLM | World Model (WM) |
|--|-----|------------------|
| 擅长 | 语义、长程显式记忆、指令与问答 | 物理、动作后果、短时未来 |
| 表示 | 文本 / 离散 token / 显式 Memory | 连续 latent / 视频 / 状态 |
| 失败模式 | 物理幻觉、空间尺度错 | 语义盲目、目标误解 |

笔记主张二者经 **语义接口** 组成 **具身推理链**，并由 **Critic** 回写 Memory（页 14）。

---

## 三种对接模式

### 1. 文本接口

VLM 输出场景描述 / MetaAction / CoT → 作为 WM 或规划器的 prompt。  
实现简单，但带宽低，易损几何。

### 2. Hidden 接口（页 15）

开源 VLM forward 取 **text hidden**；MoT 放入 text Tower；场景 prompt 条件化 Video/Action Gen。  
保留更富语义向量，适合 Video Thinking。

### 3. Token 共享骨干（页 22, 27）

同一 FM 产 PV/BEV/Action；VLM 头与 WM/DiT 头共享下层。  
训练难，但部署统一。

---

## Video Thinking 工作流

```
hist video latent + state + action
        → Video Gen (e.g. Wan 2.2) ↔ Action Gen
        → 未来帧 + action
```

用途：

- 训练期：想象未来作辅助损失；  
- 推理期：对高难场景展开（配合 ThinkDrive）；  
- 数据期：生成难例增强。  

FSDrive 思路：**先预测未来帧，再规划**，减轻「一次压成单一向量」的信息损失。

---

## Memory 设计原则

1. **衰减 ≠ 删除**：分数衰减，事件可检索（Memory Bank Event）。  
2. **显式（VLM）+ 隐式（KV/SSM）** 双轨。  
3. 写入触发：红绿灯变化、接管、碰撞近乎、导航切换。  
4. 车端：Sparse BEV / Scene token @ 2Hz 即弱版 Memory。  

---

## Critic / Reward 对齐

CoC（笔记评分项）建议拆开：

- 感知正确？  
- 推理因果是否倒置？  
- 决策与轨迹 / TBT 是否一致？  

用强裁判模型（如 Qwen2.5-32B）做 pairwise / Yes-No，再蒸馏到车端小 Critic。

---

## 实践顺序

1. 固定 VLM 做标注与 VQA（4D Scene Graph）。  
2. 训独立 WM/Video 头。  
3. Hidden 对齐。  
4. 再考虑共享 FM 与联合 RL。  
