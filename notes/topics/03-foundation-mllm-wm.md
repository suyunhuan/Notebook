# 主题：基模、MLLM、World Model 与 VLA

关联页：01, 07, 12, 14, 15, 22, 28

---

## 1. 目标形态

**原生多模态大模型**：统一离散+连续表征，带 Memory / MoT，既能语言推理又能物理交互与规划，而非「感知模块 + 外挂 LLM」。

策略口号（页 01）：**先基本面 Transformer → MoT，再拓展**；评测靠数据 / 仿真 / 路测闭环。

---

## 2. 档位与参数（页 22）

| 档位 | 视觉骨干倾向 | FM 量级 | DiT | 备注 |
|------|--------------|---------|-----|------|
| Max | 共享 ResNet18，+~9M，无 BEV project | ~0.13B（768×18） | ~0.05B | 轻量 |
| Ultra | 更深 ResNet / FastViT 30M×7 | ~0.35B（1024×24） | ~0.1B | 可上到 ~0.7B |
| Ultimate | Text+instruct → FM→MLLM | FM/MLLM ~1.8B（2048×28） | ~0.2B | 40B token SFT 基模 |

输入配时示例：Vision 10–12Hz；LiDAR BEV 短时窗；Nav SDK/TBT。

---

## 3. 模块角色

| 模块 | 角色 | 笔记关键词 |
|------|------|------------|
| **FM** | 多传感器 token 骨干 | Self-Attn×N、归一化 PE |
| **MLLM / VLM** | 长程语义、显式记忆、问答/CoT | 理解→规划→Action |
| **WM** | 物理与动作前向模拟 | Video Thinking、rollout |
| **MoT** | 多模态混合 Transformer / 文本塔融合 | 与 MOT（跟踪）区分 |
| **DiT** | 连续轨迹/视频生成 | waypoints、Video Gen |
| **Critic** | 评价轨迹/推理，回写 Memory | 具身推理链 |

具身接口（页 14）：`VLM（语义记忆）+ WM（物理动作）→ 语义接口 / 具身推理链`。

---

## 4. Video Thinking 与 4D Scene Graph

**Video Thinking**（页 15）：

- 输入：hist video latent / state / action  
- 生成：Wan 2.2 Video Gen ↔ Action Gen  
- 条件：场景 prompt + MoT text Tower + 开源 VLM text hidden  
- 任务：过去 N 帧 → 未来 M 帧 + action  

**4D Scene Graph**（页 12）：

- VLM 做盲区、物体、四维动态打标  
- 快/慢 rollout（单帧关键词置信度）→ SFT → pairwise → DPO  
- 实证：通过率大幅上升、难例下降  

---

## 5. 中间层特征与 KV

- 常用抽取层：**0 / 9 / 18**（与 28 层级模型对应）。  
- KV Cache 压缩（如 28→3）服务车端访存。  
- Causal Mask：保证推理效率与自回归一致性。  

---

## 6. VLA / 原生多模态要点（页 01）

- **Causal 对齐**：隐空间交互流的预测质量。  
- **CFG**：离散+连续引导。  
- **离散+RL**：动作 token 可强化学习。  
- **快慢系统**：低频推理 + 高频控制。  
- **Data Efficiency + 闭环**：仿真与路测回流。  

对标：mindVLA-U1、Xiaomi OneVL（decoder/latent/RL）。

---

## 7. 与 LLM 的衔接（页 28）

```
Depth / 视觉 / 视频 → ViT / 4D Encoder → LLM
```

性能锚点：`7B → 3Hz`；`3B → 5–8Hz`；能力：流式视频 + 空间理解。  
辅助任务：depth、BEV、重建、生成——用生成式目标养世界模型能力。

---

## 8. 检查清单

- [ ] MoT 与 MOT 命名是否在文档中区分清楚  
- [ ] FM 与 DiT 参数量是否匹配 latency 目标  
- [ ] Memory 衰减策略：衰减但不丢关键事件  
- [ ] 是否同时保留离散 action token 与连续 waypoints  
- [ ] Scene Graph / VQA 数据是否覆盖盲区与 TBT  
