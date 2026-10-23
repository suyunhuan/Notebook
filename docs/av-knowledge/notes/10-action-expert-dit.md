# Action Expert 中的 DiT 与 adaLN 条件注入

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | 自动驾驶Action Expert的DiT设计与adaLN解释 |
| Thread ID | `xA6rDOr9aEUJfSJ9L` |
| 源链接 | https://www.doubao.com/thread/xA6rDOr9aEUJfSJ9L |
| 所属分类 | 智能驾驶核心技术 / VLA Action 生成 |

## 一句话摘要

Action Expert 用条件扩散/流匹配在 DiT 上生成未来动作序列，并以 **adaLN-Zero** 把 ego 与 diffusion timestep 作为全局条件注入每一层 Transformer block。

## 核心关键词

`Action Expert` · `DiT` · `adaLN-Zero` · `Flow Matching` · `DDPM` · `ego conditioning` · `diffusion timestep` · `cross-attention 条件解耦`

## 知识脉络

### 1. Action Expert 在 VLA 中的角色

- 任务：条件生成未来 \(H\) 步动作（waypoint、速度、加速度、转角等），而非语言 token。
- 主干：Diffusion Transformer（DiT）；训练目标常见为 **DDPM 噪声预测** 或 **Flow Matching 速度场**（工程上后者采样步数更少，更常见于实时规控）。
- 可选 Latent DiT：先用 VAE/PCA 压到潜空间，缩短序列长度。

### 2. 输入与两类「时间」

| 信号 | 含义 | 注入位置 |
|------|------|----------|
| 噪声动作序列 \(x\in\mathbb{R}^{B\times H\times D}\) | 待去噪的动作 token | DiT 主输入 + **token-wise 位置编码**（horizon 内第 \(k\) 步） |
| diffusion timestep \(t\) | 去噪/ODE 进度，**非**真实世界时间戳 | 正弦编码→MLP→\(t_{emb}\)，进全局条件 \(c\) |
| ego state | 速度、加速度、yaw rate、转向、历史位姿等 | 独立 MLP→\(ego_{emb}\)，与 \(t_{emb}\) 融合为 \(c\) |
| 场景上下文（可选） | BEV pooled、route、目标点、agent 摘要 | 可并入 \(c\)，或留给 cross-attn |

**关键区分**：diffusion timestep → adaLN 全局条件；horizon 内真实时序位置 → action token 的 pos embedding。

### 3. 全局条件融合与 DiT Block

1. \(c = t_{emb}+ego_{emb}\)，或 \(c=\mathrm{MLP}(\mathrm{concat}(t_{emb},ego_{emb}))\)。
2. \(c\) 为 **per-sample 全局向量**，同一样本内所有 action token 共享。
3. adaLN-Zero 由 \(c\) 生成 6 组调制量：\(\gamma_1,\beta_1,g_1,\gamma_2,\beta_2,g_2\)：

```
x → LN(无仿射) → (1+γ)·x+β → MHSA → 残差×gate
  → LN → (1+γ)·x+β → MLP → 残差×gate
```

- **adaLN**：LN 的 \(\gamma,\beta\) 由条件动态生成（条件调制）。
- **adaLN-Zero**：额外 gate，初始化为 0，训练初期近似恒等，稳定性更好（DiT 原论文默认）。

### 4. 条件注入选型（工程惯例）

| 方式 | 适用 | 代价/风险 |
|------|------|-----------|
| adaLN / adaLN-Zero | ego、timestep、导航等**低维全局状态** | 几乎不增 attention 开销；首选 |
| Cross-Attention | 障碍物、地图等**结构化长序列** | 算力大；宜留给 World/场景专家 |
| add/concat 到 token | 弱条件 | 易冲掉动作时序信息；少作主方案 |

惯例：**全局状态走 adaLN，空间实体走 cross-attn**，避免全部塞进同一个 \(c\) 互相污染。

### 5. 工程要点

- ego 各量纲必须归一化，否则 \(\gamma/\beta\) 震荡。
- \(t_{emb}\) 与 \(ego_{emb}\) 维度对齐后再相加。
- ego 历史应先 pool 成全局向量，不要做成 per-token 塞进 adaLN。
- Flow Matching 的 \(t\) 是 ODE 时间，编码细节与 DDPM 不同，但注入范式不变。
- 输出可加动力学投影 / kinematic 校验。

## 知识点校验

- **Correct**：DiT 原作（Peebles & Xie）确以 adaLN-Zero 做类条件调制；「timestep ≠ 真实时间戳」的区分正确且易被忽略。
- **Caution**：原文将 SimWAM、AutoMoT 等作为「主流都用 adaLN」的论据——具体系统是否全部 adaLN-Zero、是否混用 cross-attn，需对照各论文实现，不宜当作统一事实。
- **Caution**：「Action Expert 几乎全部用 Flow Matching / adaLN-Zero」是工程概括，非统计结论；量产与论文中仍有回归头、离散 codebook、纯 cross-attn 等并存路线。
- **Expand**：连续动作上的 classifier-free guidance、多模态条件（language + BEV）如何与 adaLN/cross-attn 分层；与轨迹 Codebook 自回归路线的对比选型。

## 延伸阅读建议

- DiT: *Scalable Diffusion Models with Transformers*（Peebles & Xie）
- Flow Matching / Rectified Flow 系列（Lipman et al.；Liu et al.）
- 机器人/具身 VLA 中 diffusion policy、π0 等 Action Expert 设计（对照条件注入方式）
- 智驾侧条件扩散规划：对照各家 World-Action / VLA 开源实现中的 DiT block

## 与其他主题的关联

- 轨迹离散化与 Action Expert 路线对比：[`../notes/11-trajectory-codebook.md`](../notes/11-trajectory-codebook.md)
- 规控偏好对齐（DPO/GRPO）与动作生成后处理：[`../notes/12-planning-dpo-rl.md`](../notes/12-planning-dpo-rl.md)
- Qwen3-VL VLA 中 Flow Matching Action Expert：[`../notes/16-qwen3vl-vla-architecture.md`](../notes/16-qwen3vl-vla-architecture.md)
- VLA 决策与 CoC：[`../notes/08-vla-ambiguity-coc.md`](../notes/08-vla-ambiguity-coc.md)
- BEV 场景特征作为条件来源：[`../notes/13-bev-depth-models.md`](../notes/13-bev-depth-models.md)
