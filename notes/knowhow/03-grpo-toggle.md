# Know-how：GRPO 与 Toggle Token 预算

## GRPO 在做什么？

**Group Relative Policy Optimization**：对同一输入 \(x\) 采样 \(K\) 条输出 \(\{y_j\}\)，用组内相对奖励做优势，避免单独 Critic 网络。

定义（与笔记对齐）：

\[
\bar r(x)=\frac1K\sum_{j=1}^K r(x,y_j),\quad
N=\sum_j |y_j|
\]

目标大致为（示意）：

\[
\mathbb{E}_{x}\Big[\frac1N\sum_j\sum_i
\mathrm{Clip}\Big(\frac{\pi_\theta(y_i^j\mid x,y_{0:i}^j)}{\pi_{\mathrm{old}}(\cdot)},\alpha,\beta\Big)
\cdot\big(r(x,y_j)-\bar r(x)\big)
-\tau\big(\log\frac{\pi_\theta}{\pi_{\mathrm{old}}}\big)^2
\Big]
\]

要点：

1. **相对优势** \(r-\bar r\)：同组内比较，降奖励尺度敏感。  
2. **Clip 比率**：限制离策略步长。  
3. **平方 log-ratio 惩罚**：简化 KL 约束，抑漂移。  
4. **Token 截断**：log-ratio 出 \([\alpha,\beta]\) 则该 token 梯度清零 → 稳训练。  

监控：KL、温度、lr、reward hacking、collapse（页 24）。

---

## 为什么需要 Toggle？

纯 GRPO/RL 常出现：

- **冗长 CoT**：靠堆 token 刷分；  
- 或 **过度压缩**：变短但变笨。  

**Toggle** 在两相之间随机/周期交替：

| 相位 | 行为 | 目的 |
|------|------|------|
| Phase 0 | 仅当组平均奖励低于 λ **或** 长度在 budget 内时保留奖励（笔记公式）；强调预算 | 逼简洁 |
| Phase 1 | 正常奖励 | 保/提推理能力 |

\[
\mathrm{budget}(x)=\mathrm{Percentile}\big(\{|y_j|:r(x,y_j)=1\}\big)
\]

即：用**答对样本的长度分布**定义「合理长度」，而不是拍脑袋固定 max tokens。

报告效果：token 用量 ↓ **25%–30%**，同时维持推理目标。

---

## 与 Test-time Scaling 的关系

- Test-time 想「多想一会」提升准确率，但车端有延迟墙。  
- 训练期 Toggle 教模型：**能短则短，难例才长**（类似 ThinkDrive 的难度估计）。  
- 风险：**长度过拟合**——学会匹配 budget 分布，却不提升真实推理。需保留 Phase 1 与难例集。  

---

## 驾驶场景落地

1. \(r(x,y)\) 不要只用「格式对」：并入轨迹安全、TBT 一致、STL 违例。  
2. \(K\) 太小 → 相对优势噪声大；太大 → 贵。  
3. 对轨迹类连续输出，可先离散 MetaAction 再 GRPO，或对文本 CoT 做 GRPO、对 DiT 另训。  
4. 与 DPO 分工：DPO 适合成对偏好数据；GRPO 适合可自动打分的组采样。  

---

## 最小实验协议

1. 冻结 SFT 模型为 \(\pi_{\mathrm{old}}\)。  
2. 固定 K、clip、τ；只开 GRPO。  
3. 记录准确率–平均长度曲线。  
4. 再开 Toggle，调 percentile 与相位周期 \(m\)。  
5. 对比难例集：长度应升、简单题长度应降。  
