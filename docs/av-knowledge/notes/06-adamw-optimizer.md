# AdamW 优化器（原理与智驾调参）

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | AdamW参数详解与对比 |
| Thread ID | `x4FdX61BhJAMV75Zq` |
| 源链接 | https://www.doubao.com/thread/x4FdX61BhJAMV75Zq |
| 所属分类 | 大模型技术 / 优化器 |
| 源文件 | `../sources/x4FdX61BhJAMV75Zq.md` |

## 一句话摘要

AdamW = Adam 的一/二阶矩自适应更新 + **解耦**权重衰减；智驾 VLM/VLA 调参重点在 β₂ 与 loss spike、模块级梯度不均衡、以及 bias/Norm 不加 decay。

## 核心关键词

AdamW；β₁ / β₂；解耦 Weight Decay；Bias Correction；Grad Clip；Loss Spike；分组学习率；Muon（对比）

## 知识脉络

### 1. 更新式（原理压缩）

对梯度 \(g_t\)：

\[
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
\]

偏差修正后：

\[
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}-\eta\lambda\theta_t
\]

- **β₁**：一阶动量 EMA，平滑梯度方向；大模型几乎固定 **0.9**。
- **β₂**：二阶矩 EMA，估方差做逐参数自适应步长；默认常 **0.999**（长记忆）。
- **λ（weight decay）**：每步直接缩参数，**不进梯度**——相对旧版 Adam 把 \(\lambda\theta\) 加进 \(g\)（等价 L2、被 \(1/\sqrt{v}\) 扭曲）是正确解耦。
- **惯例**：bias、LayerNorm（及同类 Norm）**不加** wd，只作用线性/卷积权重。

### 2. 为何 LLM/VLM 常用 AdamW

- 相对 SGD：逐参数自适应，对异构梯度尺度更稳，降低全局 lr 敏感度。
- 代价：额外存 \(m,v\)（约 2× 参数状态）；可用 8-bit Adam 等压显存。
- **lr 与有效 wd 耦合**：更新里衰减项含 \(\eta\lambda\)，改 lr 会连带改变衰减强度，调参要一起看。

### 3. 智驾 / VLA 相关调参要点

**（1）β₂ 与 loss spike（混合难例、DPO、脏数据常见）**

- β₂=0.999 → 有效窗口 ~1000 步：一次尖峰 \(g\) 抬高 \(v\) 后，等效 lr 被压很久，loss「假死」。
- 可试 **0.99** 折中，或 **0.95**（窗口 ~20 步）加快遗忘异常 \(v\)；副作用是自适应步长更噪，需 **grad clip**。
- **Clip 局限**：只限本步 \(g\)，已写入 \(v\) 的历史污染清不掉 → clip 与降 β₂ 互补，非替代。
- Spike 根因分流：偶发 → 查脏数据 / 难样本课程 / on-policy；频繁 → 查 loss 尺度、锐利 minima、wd/正则，勿只拧优化器。

**（2）多模块梯度不均衡（ViT + Adapter + LLM + Action Expert）**

- 预训练主干梯度小，随机初始化 adapter / AE 梯度大；**全局** `clip_grad_norm_` 易误伤主干或压不住 AE。
- 先做 **按模块 grad norm**（P50/P90/P99）诊断，再定 clip 与分组 lr。
- 稳妥流程：**warm-up 只训小 adapter 预对齐** → 再逐步解冻；若仍差一个数量级，用分组 lr decay。

**（3）常规配方起点**

| 场景 | betas | weight_decay | 备注 |
|------|-------|--------------|------|
| 通识预训练 | (0.9, 0.999) | 0.01 | 数据稳时默认 |
| 垂域 SFT | (0.9, 0.999→0.99) | 0.01 或 0.001 | 视过拟合 |
| 难例/偏好/OPD 混合 | (0.9, 0.95~0.99) | 维持 + 加强数据清洗 | 配 clip + 课程 |

### 4. 与 Muon 的边界（知晓即可）

- Muon：对 **2D 矩阵权重** 做动量矩阵极分解/谱归一，无 β₂；bias/Norm/Embedding 仍用 AdamW。
- 增益多见于预训练样本效率；VLA 微调是否划算需实测，**不能**全参数替换 AdamW。

## 知识点校验

1. **Caution**：β₂=0.95「建议」针对易 spike 的混合/离策略数据，**不是**替代所有 LLM 预训练的新默认；干净单域预训练仍常 0.999。
2. **Caution**：原文对 DeepSeek 等采用 Muon 混合优化器的叙述属二手信息，以实现仓库/官方报告为准。
3. **Expand**：VLA 各模块最优分组 lr 比（ViT : LLM : AE）高度依赖是否冻结与分辨率/序列长，需基于模块 grad 统计做网格，而非照搬单一经验值。
4. **Correct**：AdamW 解耦 weight decay、Norm/bias 通常不加 decay，是标准实践（Loshchilov & Hutter）。

## 延伸阅读建议

- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW)
- Adam 原论文；8-bit Optimizers
- Muon（Keller Jordan et al.）——矩阵参数优化补充阅读
- 大模型训练中 loss spike 处理实践（数据清洗 + clip + 矩超参）

## 与其他主题的关联

- [训练效率](../notes/04-training-efficiency.md) — ZeRO 分片优化器状态、8-bit Adam
- [VLM 训练阶段](../notes/01-vlm-training-stages.md) — 分阶段解冻与 adapter warm-up
- [在线蒸馏 OPD](../notes/03-online-policy-distillation.md) — off-policy/难样本触发 spike 时的课程与蒸馏
- [数据 Trick](../notes/07-e2e-data-tricks.md) — 脏样本是 spike 治本方向
