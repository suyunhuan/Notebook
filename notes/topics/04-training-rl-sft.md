# 主题：训练范式 — SFT、蒸馏、RL、偏好优化

关联页：02, 08, 11, 12, 17, 19, 20, 21

---

## 1. 总流水线

```
预训练 / 混训 → SFT（含 grounding、VQA、traj）→ 偏好优化(DPO) / 策略优化(GRPO/PPO)
                              ↘ DAgger 闭环模仿
                              ↘ Toggle（能力 ↔ token 预算）
```

VL+4D 版本（页 02）：`… stage2.5 混训 → 2D/4D-SFT → 2D/4D-RL`。

---

## 2. SFT 与数据技巧

| 技巧 | 说明 |
|------|------|
| Grounding 正样本 | 加强指代与路线对齐 |
| P1 vs P2 | 多样问答能力难直接继承到固定 MA·Traj 格式 |
| Traj 注入 | 把轨迹质量当 reward/loss weight 写进 SFT |
| 温度 T≈0.7 | 提升泛化 |
| Dropout / Augment | attention dropout；BEV/PV crop 抖动 |
| high overlap low difference | 近邻序列对比，逼模型抓因果差异 |
| 关键帧 / No-cut / CoT 指令 | 长视频与推理数据配方 |

诊断：过早收敛、lr 过大、数字/坐标建模难。

---

## 3. 蒸馏（OPD / MOPD）

页 19 技能树：

- **gt**：one-hot；**teacher**：soft 分布；**student**：解码 rollout。  
- SFT+RL 做大 teacher；多场景更强 expert。  
- **单词表/局部 KL** 替代全词表 KL → 降算力；提供 dense reward、减轻 reward hacking。  
- off-policy 单轨迹 vs LLM 多轨迹都需要 teacher 信号。  

---

## 4. Teacher Forcing vs Self Forcing

- Teacher forcing：训练用真值前缀，易 train/test 落差。  
- Self forcing：用自身采样；笔记称约 10% 设置可能不稳。  
- Meta Action 作 segmentation：对主网影响可能很小，需看 Freq / Grad L2。  

---

## 5. DAgger 与 RL 驾驶

- **DAgger**：专家纠偏聚合，配 dense depth / 几何。  
- 纵向 **ARDP**：交互推演 + 语义风险降速。  
- Reward Model（页 11）：CoC 感知/推理/决策分项；与 TBT、轨迹一致；防因果倒置。  

---

## 6. GRPO 与 Toggle（页 20）

**GRPO 核心思想**：对同一 prompt 采 K 条响应，用组内相对优势 \(r-\bar r\)，并对重要性比率做 clip，外加偏离惩罚。

**Token 截断**：log ratio 出 \([\alpha,\beta]\) 则清零梯度 → 稳 off-policy。

**Toggle 算法**：

- Phase 0：在平均奖励未达阈值或超 budget 时压制长度相关奖励。  
- Phase 1：放开，强化推理能力。  
- `budget(x)` = 正确样本长度的分位数。  
- 报告效果：token ↓ **25%–30%**。  

与 Test-time scaling：追求简洁 CoT，但要防「长度过拟合」（学会写短却不学会想对）。

---

## 7. 偏好与文献向方法（页 12, 21）

- 快慢 rollout → SFT → pairwise → **DPO**。  
- DriveLM / Reason2Drive：结构化推理与 RA/RD/MS。  
- ThinkDrive：难度估计 → 浅层 or 深度 CoT + 课程学习。  
- 扩散规划 + STL 约束 + RL 后训练。  

---

## 8. 检查清单

- [ ] SFT 是否区分「语言多样」与「动作格式」数据配比  
- [ ] KL/奖励是否 dense，能否观察到 hacking  
- [ ] GRPO 的 K、clip、τ 是否与序列长度匹配  
- [ ] Toggle 的 λ 与 percentile 是否按任务校准  
- [ ] DAgger 迭代是否同步更新 depth/几何教师  
