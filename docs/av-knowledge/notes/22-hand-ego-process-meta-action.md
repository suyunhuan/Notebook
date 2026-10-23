# 手写笔记：Ego 推演与 Teacher Forcing / Meta-Action / KV 读存

## 元信息

| 字段 | 内容 |
|------|------|
| 来源页码 | Handwriting Page 06–07（`_raw_batch_a.md` · a7d4-77eb / a892-746c） |
| 所属分类 | B. VLA / 决策 · B2/B4 推理–动作与 Action Expert |
| 笔记类型 | 手写精炼（非豆包对话） |
| 相关原始图 | `../assets/handwriting/01a169b1-a7d4-77eb-*` · `01a169b1-a892-746c-*` |

## 一句话摘要

Ego process 用 TTC/碰撞等做未来推演与风险降速；训练上用 **Teacher/Self Forcing** 管训推分布，**meta-action** 切语义，并从 VLM 中间层 **kvcache 读存** 条件化 **FM→DiT** 动作头。

## 核心关键词

`ego process` · `TTC` · `Teacher Forcing` · `Self Forcing` · `meta-action` · `kvcache` · `FM` · `DiT` · `PE` · `ARDP`

## 缩写与术语详解

| 缩写/词 | 全称 / 含义 | 在本页语境 |
|---------|-------------|------------|
| **TTC** | Time-To-Collision | 碰撞时间估计，未来推演安全指标 |
| **DAC** | 手稿缩写（δ 标注；或为 Distance/Decel 相关量） | 与 collision(δ) 并列的风险标量，全称未确认 |
| **PE** | Positional Embedding | 位置/时序编码 |
| **TempCompass** | 时序理解评测/模块名（手稿专名） | 与 granularities attention 并列 |
| **Teacher Forcing** | 训练时喂 **真值** 前缀，而非模型自身预测 | 加速收敛，但造成 **train/test 分布差**（exposure bias） |
| **Self Forcing** | 训练时部分步用 **模型自己的输出** 滚动 | 手稿：约 10% train「不好/不按常规」——用自生成轨迹缓解训推差 |
| **Meta-action (MA)** | 结构化语义动作（停/跟/变道等） | segmentation 切段；对网络影响小，**traj 影响大** |
| **kvcache / KV Cache** | Key-Value Cache | Transformer 已算层的 K/V 复用；此处强调从 VLM **中间层读出** 作动作条件 |
| **FM** | Flow Matching | 学速度场的连续生成目标，常接 Action Expert |
| **DiT** | Diffusion Transformer | 扩散/流匹配骨干，收 FM 与 KV 条件 |
| **ARDP** | 手稿缩写（全称未确认） | 与纵向交互/风险降速并列 |
| **GR0P / GRPO[?]** | 手稿模糊；或为 GRPO（Group Relative Policy Optimization） | 与 VLM 层读出、RL 对齐相关，**勿武断定名** |

## 知识脉络

### 1. Ego process：未来推演（Page 06）

```
输入（状态/感知时序）→ head
  → 未来推演指标：TTC、DAC(δ)、z-p(speed)、collision(δ)
  → 纵向策略：交互 / 感知推演；语义 / 风险降速
```

配套：PE、granularities attention、TempCompass；倒车等特殊工况；芯片/仓统一属工程备忘。

### 2. Teacher vs Self Forcing（Page 07）

| 模式 | 训练喂什么 | 风险 |
|------|------------|------|
| Teacher Forcing | GT 历史动作/token | 训推分布不一致 |
| Self Forcing | 模型滚动输出（手稿提 10% 特例） | 早期噪声大，需课程或混合比 |

### 3. Meta-action 与监督构造

- **MA segmentation**：按语义切段；MA 本身对网络扰动小，**轨迹监督权重大**。
- **gradient L2** + 构造 GT：用可微 L2 约束，并显式造真值数据。
- 示意曲线（klr[?]）提示不同损失项权重扫描。

### 4. VLM 中间层 → FM → DiT

```
VLM 层 {0, 9, 18}  ──读──→ 条件模块（GR0P[?] / ViT / FM）
                              │
KV cache 28 → 3（压缩读存）     │
                              ▼
                         FM → DiT（另有一路输入进 DiT）
```

工程要点：

- 不是只用最后一层：取 **浅/中/深** 层语义（0/9/18）分别对接。
- **kvcache 28→3**：层或 token 维压缩，控显存与延迟（与 [16](16-qwen3vl-vla-architecture.md) KV_cond 一致）。
- FM 隐层「90% attention」类备注：条件注入占比的经验观察，需实验复核。

## 知识点校验

1. **Correct**：Teacher Forcing 的 exposure bias、用 Self Forcing/混合滚动缓解，是序列决策常识；KV 中间层条件化 DiT/FM 与本库 Action Expert 设计同构。
2. **Caution**：DAC、ARDP、GR0P、z-p(speed)、「10% self forcing」比例与「Gemini 3 chip」均为未验证手稿项。
3. **Caution**：meta-action「对网络影响很小」是相对现象，**不等于可忽略 MA 标注质量**——[08](08-vla-ambiguity-coc.md) 强调 MA 可验证性。
4. **Expand**：Self Forcing 比例课程表；哪几层 KV 对短时控制 vs 长时语义最有效；TTC 推演头与规划代价如何闭环。

## 与其他主题的关联

- [10 Action Expert DiT](10-action-expert-dit.md) — adaLN / FM 条件注入
- [16 Qwen3-VL VLA](16-qwen3vl-vla-architecture.md) — Hook 中间层 KV → Action
- [08 VLA 歧义与 CoC](08-vla-ambiguity-coc.md) — meta-action 结构化
- [11 轨迹 Codebook](11-trajectory-codebook.md) — 离散动作与 traj
- [05 因果建模](05-causal-modeling-trajectory.md) — A→T 与风险指标
