# OPD/MOPD、Toggle 预算、STL 约束与 CoT / 扩散规划

## 元信息

| 字段 | 内容 |
|------|------|
| 来源 | 手写笔记 Page 19–21（`handwriting/_raw_batch_c.md`） |
| 资产 ID | `b1e6-784c` · `b2c5-76a3` · `b39f-77f1` |
| 所属分类 | 特定技术方法 / 后训练、推理预算与约束规划 |
| 关联原始页 | Batch C：OPD 要点；词元 Clip + Toggle；STL / Latent-CoT / 扩散 |

## 一句话摘要

用 OPD/MOPD 做稠密 token 蒸馏逼近教师，用词元级 Clip 与 Toggle 在 test-time 预算下交替压推理长度，再用 STL/分类器引导与自适应 Latent-CoT 把规则与思考深度注入规划。

## 核心关键词

`OPD` · `MOPD` · `全词表/单词表 KL` · `GRPO` · `词元 Clip` · `Toggle` · `test-time 缩放` · `STL` · `Classifier Guidance` · `Latent-CoT` · `ThinkDrive` · `Diffusion Planner`

## 知识脉络

### 1. OPD / MOPD（蒸馏侧）

| 概念 | 含义 |
|------|------|
| **OPD** | On-Policy Distillation：学生自采样序列，教师给 soft logits，token 级 KL（常 reverse KL）监督 |
| **MOPD** | Multi-teacher OPD：多专家/多场景教师蒸同一学生 |

要点（与 note 03 对齐、此处补手写落地细节）：

- GT 常 **one-hot**；teacher 是 **soft**；student **解码 rollout**（on-policy）。
- **单词表 KL** 省算力；**全词表 KL** 降方差但贵。
- 相对 sparse RL：**无 reward hacking、稠密 reward**；可 off-policy 变体。
- 单轨迹 vs **VLM 多轨迹**；SFT+RL 做大教师后再蒸；目标 **性能逼近 teacher**。
- 指令跟随 / 物体 + **TBT** 等任务上做单配置最优方案搜索。

**训练超参备忘**：bf16；`num_workers=8`；data 不解压；`bs = bs/gpu × accum × GPUs = 1`；10 epochs；约 3k step 存 ckpt。  
**Special token**：unused `[0,1,30,31]`、ignore、unk、answer、长短类别标记等——训推必须同一套。

### 2. GRPO 与词元级 Clip（RL 目标）

**GRPO（Group Relative Policy Optimization）**：对同一 prompt 采 \(K\) 条响应，用组内相对奖励（相对均值）更新策略，常作 PPO 的简化变体（无独立 Critic 或弱化价值网络）。手写页把 **GRPO/PPO/DPO** 并列进监控清单（详见 note 28 监控项）。

设 \(K\) 条响应 \(\{y_j\}\)，\(\bar r(x)=\frac1K\sum_j r(x,y_j)\)，\(N=\sum_j|y_j|\)。词元级目标（笔记形式）：

\[
\mathbb{E}_{x\sim D}\Bigg[\frac{1}{N}\sum_{j=1}^{K}\sum_{i=1}^{|y_j|}
\mathrm{Clip}\Big(\frac{\pi_\theta(y^j_i\mid x,y^j_{0:i})}{\pi_{old}(y^j_i\mid x,y^j_{0:i})},\alpha,\beta\Big)
\cdot\big(r(x,y_j)-\bar{r}(x)\big)
-\tau\Big(\log\frac{\pi_\theta}{\pi_{old}}\Big)^2\Bigg]
\]

- **词元 Clip / 梯度掩码**：重要性比落在 \([\alpha,\beta]\) 外则梯度清 0 → 缓解 **off-policy 偏差**，抬训练稳定性。
- 奖励用 **组相对** \(r-\bar r\)，与 GRPO 同族。
- 二次项 \(\tau(\log\pi_\theta/\pi_{old})^2\)：额外约束策略漂移。

### 3. Toggle：test-time 预算交替优化

**问题**：test-time 缩放鼓励「想清楚」，但易 **长度过拟合**（废话 token 换分）。

**Toggle 算法**：在预算约束下 **交替两阶段** 优化同一策略——

- **Phase 0**（\(\lfloor t/m\rfloor \bmod 2=0\)）：仅当组均奖励 \(<\lambda\) **或** 响应长度 \(\le \mathrm{budget}(x)\) 时，保留原奖励；否则置零/掩码 → 逼迫「短而够用」。
- **Phase 1**：恢复 \(\tilde r=r\)；仅当平均准确率 \(>\lambda\) 时在 budget 内执行，并允许更大词元预算练能力。

\[
\mathrm{budget}(x)=\mathrm{Percentile}\big(\{|y_j|\mid r(x,y_j)=1\},\,\alpha\big)
\]

- 双目标：推理能力 ↔ 长度效率；笔记称可 **↓ 25%–30% token**。
- 与「简洁思维链」一致：先保对，再压长度，交替防塌到极短或极长。

### 4. STL、结构化推理与 Latent-CoT

**STL（Signal Temporal Logic，信号时序逻辑）**：

- 用带时间算子的逻辑公式描述规则（如「若前方 \(d\) 米内有行人，则在 \(T\) 秒内速度 \(<v\)」）。
- 可计算轨迹对公式的 **鲁棒度 / 违背程度**，作为可微或准可微代价回传 → **逻辑约束引导** 规划（笔记：Gen-Drive、DiAD 等扩散规划）。
- 与 **Classifier Guidance** 并列：分类器/规则网络引导采样轨迹落在安全/合规区域。

规划奖励维度：Planning Acc.；**Action-Weighted Reward**（如时间加权）；Planning Diversity；format。

**结构化推理数据**：

| 路线 | 焦点 |
|------|------|
| DriveLM | 感知→预测→规划逻辑链 |
| Reason2Drive | 对象 / 场景 |
| 评分 | Reasoning Alignment：准确 RA · 多样 RD · 匹配 MS |

**自适应 Latent-CoT（ThinkDrive 等）**：

- difficulty estimator：简单 → 浅推理直接出 action；高熵/不确定 → 加深 CoT。
- Soft/Step Budgeter；Fast DriveCoT 并行子任务；视觉 CoT（FSDrive：先预测未来帧再规划），减轻压成一维序列时的时空信息丢失。

**扩散规划三点**：① 生成评估 + RL 微调；② 感知–规划联合（Diffusion Planner）；③ **STL** 注入违背度梯度。

## 知识点校验

1. **Correct**：OPD 用学生 rollout + 教师 soft 分布、相对 sparse RL 更稠密；词元 Clip 与 PPO/GRPO 重要性采样截断同族；STL 鲁棒度可作规划约束是文献主线。
2. **Caution**：Toggle 的 \(\lambda\)、分位 \(\alpha\)、交替周期 \(m\) 强依赖任务；「↓25%–30% token」是笔记经验值，换域必复测。
3. **Caution**：全词表 OPD 与 VLM 多轨迹成本极高；未解决教师–学生容量鸿沟前，盲目全词表可能只烧算力。
4. **Expand**：STL 公式库谁写、如何覆盖中国交规与内部产品规则，原稿未给工程流程。
5. **Expand**：Latent-CoT 的 difficulty 标签是离线启发式还是在线熵，与 Toggle budget 是否共用同一长度控制器，需统一设计。

## 与其他主题的关联

- [01 VLM 训练阶段](01-vlm-training-stages.md) — SFT→蒸馏→RL 顺序
- [03 OPD](03-online-policy-distillation.md) — OPD/MOPD 方法学主文
- [08 CoC / Meta-action](08-vla-ambiguity-coc.md) — 推理–动作一致性
- [09 Reasoning VQA](09-alpamayo-reasoning-vqa.md) — 结构化推理数据
- [12 DPO/GRPO](12-planning-dpo-rl.md) — 规控偏好与在线 RL
- [10 DiT Action](10-action-expert-dit.md) — 扩散动作头
- [28 档位与 CrossAttn+DiT](28-hand-model-tiers-fm-dit.md) — GRPO 监控清单与轨迹头
- [18 VLA L3/L4 路线](18-hand-vla-l3l4-roadmap.md) — 闭环与因果对齐总纲
- [22 Meta-Action](22-hand-ego-process-meta-action.md) — 推理–动作切分
- [23 CoC Reward / Agent Loop](23-hand-data-sft-agent-loop.md) — CoT 质量 reward 与本篇 Toggle 互补
- [24 Rollout→SFT→DPO](24-hand-scenegraph-streaming-4d.md) — 偏好对齐前序
