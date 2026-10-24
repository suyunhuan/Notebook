# 在线策略蒸馏（OPD）

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | 在线蒸馏（OPD）体系与智驾应用 |
| Thread ID | `xzgL8jt1IX592B2Gy` |
| 源链接 | https://www.doubao.com/thread/xzgL8jt1IX592B2Gy |
| 所属分类 | 特定技术方法 / 后训练与蒸馏 |
| 源文件 | `../sources/xzgL8jt1IX592B2Gy.md` |

## 一句话摘要

OPD（On-Policy Distillation）用学生自己采样的序列、教师 token 级 logits 做监督，缓解模仿学习曝光偏差，并可用 hint 自蒸馏、多专家蒸馏扩展到智驾 VLA 的通识、决策与难例能力。

## 核心关键词

OPD；On-Policy Distillation；Exposure Bias；Forward/Reverse KL；MOPD；OPSD；Hint；Strong-to-Weak；GKD；POPE；RLTF；VLA→T

## 知识脉络

### 1. 动机与定位

- **问题**：SFT / 离线 KD 在 teacher-forcing 下训练，推理时自回归误差累积 → exposure bias / train–infer mismatch；小模型难拟合教师全分布。
- **OPD 做法**：学生采样序列（on-policy）→ 教师对同一前缀给 logits → 通常用 **reverse KL**（及变体）做 token 级稠密监督；不把梯度回传到采样分布本身（teacher-forcing 打分）。
- **教师来源**：外部更强模型、多专家（MOPD）、或「带 hint 的自己」（OPSD）。

### 2. 后训练目标对照（精简）

| 方法 | 散度/损失 | 序列来源 | 特点 |
|------|-----------|----------|------|
| SFT | NLL | GT | 实现简单；训推分布易偏 |
| Seq-level KD | NLL | Teacher 序列 | 学教师行为，仍偏 off-policy |
| Supervised KD | Forward KL | Teacher logits | mode-covering；易在教师空白区浪费 |
| Rejection Sampling | NLL | 学生正样本 | on-policy 正强化 |
| On-policy imitation | NLL | 学生序列+专家标 | DAgger 类迭代 |
| **OPD** | 常 Reverse KL | 学生序列+教师 logits | 演化、抑低质生成；多样性下降 |
| Generalized KD | 混合 | 教师/GT/学生 | 最优散度任务相关 |

**KL 选择**

- Forward \(D_\mathrm{KL}(P\|Q)\)：mean-seeking / mode-cover，学平滑全分布。
- Reverse \(D_\mathrm{KL}(Q\|P)\)：mode-seeking，贴教师高概率模态。
- 任务相关可混合；与教师分布差异大时配比需实验（原文 TBD）。

### 3. 业界用法（原文归纳，需对照原报告）

- **MiMo / MOPD**：多专家蒸同一学生。
- **Qwen3 / Qwen3-VL**：先离线蒸馏获知识，再 online 蒸馏。
- **GLM-5**：用教师 RL 集采样 prompt，恢复多阶段 SFT+RL 技能。
- **DeepSeek 相关讨论**：全词表 OPD 降梯度方差；缓存教师末层 hidden、在线算 logits；多教师。
- **Black-box OPD**：判别器区分学生/教师输出（minimax），适用于无 logits 的 API 教师。

### 4. 智驾落地规则（决策树）

1. GT 质量高（高质量示范轨迹）→ **SFT**。
2. 存在更强外部模型或历史强 checkpoint → **OPD / MOPD**。
3. 通识 / 指令遵从 → 蒸馏通用大模型。
4. 端到端弱于专家 VA → 蒸馏 VA。
5. 原子能力 / 难例 → 先训大教师再蒸小；可嵌套大蒸小循环。
6. 可叠加安全、舒适等 **非准确率 reward**（on-policy 结果上加权，类似多任务 loss）。

### 5. 扩展：Hint 自蒸馏与课程

- **范式**：`q + hint + on-policy a → self-teacher logits`（特权信息教无特权版本）。
- **Hint 类型**：verified CoT（OPSD）、error feedback（环境/失败 case，类 SDPO）、partial answer、同 batch 成功路径。
- **与 VLA 课程联合**：`VLA→T`、`VL→AT`、指令 CoT→reasoning CoT；若 A 准则 T 一致性往往更高。
- **单轮↔多轮**：中间轮作助教 roll-in（POPE 思路）；多轮再蒸回单轮（RLTF）。
- **难题探索**：调熵、放宽 clip、pass@k、难易混合、慎用会封顶后续 RL 的强 off-policy 蒸馏。

## 知识点校验

1. **Caution**：原文「Deepseek V4 / GPT5 / GLM-5」等产品细节来自二手整理，具体是否采用 full-vocab OPD、教师数量等应以对应官方技术报告为准，笔记只保留方法学含义。
2. **Caution**：「OPD 根治曝光偏差」表述过强——可显著缓解，但容量差距、奖励黑客、多样性坍缩仍在。
3. **Expand**：智驾轨迹是连续/多模态的，token-level reverse KL 如何对齐离散语言 token 与连续轨迹头（或 Codebook token）需单独设计，不能直接套用纯 LLM OPD。
4. **Correct**：on-policy 蒸馏用学生采样 + 教师打分、reverse KL mode-seeking，与 MiniLLM / on-policy distillation 文献主线一致。
5. **Expand**：forward/reverse KL 在「教师与学生分布差异大」时的最优混合，原文明确 TBD，落地需按任务做 A/B。

## 延伸阅读建议

- *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*
- MiniLLM；GKD；KDRL
- OPSD；POPE；Black-Box On-Policy Distillation
- Qwen3 / Qwen3-VL / MiMo-V2-Flash Technical Report
- verl Async On-Policy KD Trainer；swift GKD 文档

## 与其他主题的关联

- [VLM 训练阶段](../notes/01-vlm-training-stages.md) — 垂域 OPD 数据配比
- [VLA 歧义与 CoC](../notes/08-vla-ambiguity-coc.md) — hint / 多层 CoC 课程
- [因果建模](../notes/05-causal-modeling-trajectory.md) — 多教师按瓶颈路由
- [数据 Trick](../notes/07-e2e-data-tricks.md) — 难例与脏数据影响蒸馏稳定性
- [AdamW](../notes/06-adamw-optimizer.md) — off-policy/难样本导致 loss spike 时的 β₂ 与 curriculum
