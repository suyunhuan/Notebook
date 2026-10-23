# 自动驾驶轨迹 Codebook 与 VQ-VAE

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | 自动驾驶轨迹Codebook设计与VQ-VAE原理实现 |
| Thread ID | `x3YpYhDmOsLZs5Ow7` |
| 源链接 | https://www.doubao.com/thread/x3YpYhDmOsLZs5Ow7 |
| 所属分类 | 智能驾驶核心技术 / VLA 动作离散化 |

## 一句话摘要

用 VQ-VAE 或预聚类把连续轨迹片段压成离散运动基元 token，使 VLA 能以自回归方式规划，并需额外施加动力学与车道可行性约束。

## 核心关键词

`轨迹 Codebook` · `VQ-VAE` · `运动基元` · `Straight-Through Estimator` · `commitment loss` · `码本坍缩` · `RQ-VAE` · `分层码本`

## 知识脉络

### 1. 为什么要轨迹 Codebook

- 目标：连续未来轨迹 → 离散整数 token，适配 LLM/VLA 的 next-token 预测，而非纯回归坐标。
- 代表思路（原文提及）：AutoVLA、LAMP、AnchorVLA、ReflectDrive 等——**具体设计以各论文为准**。

### 2. 预处理（决定码本质量）

1. **局部坐标系**：以当前自车为原点，用相对位移 \((\Delta x,\Delta y,\Delta\theta)\)；绝对坐标会污染位置信息，学不到运动模式。
2. **短窗切片**：如 0.5s/1s、固定 \(T\) 个 waypoint，编码运动基元而非整条长轨迹。
3. **归一化 + 过滤**：动力学不可行、碰撞、冲出车道样本剔除或修正。

### 3. 两条构造路线

| 路线 | 做法 | 要点 |
|------|------|------|
| A. 端到端 VQ-VAE | Encoder→最近邻量化→Decoder 重建 | \(K\) 常见 512/1024/2048；\(D\) 如 64/128；重建+VQ+commit |
| B. 预聚类初始化 | K-means / K-Disk 得原型轨迹 | 可冻结或再微调；原文称 AutoVLA 类 \(K=2048\)、约 0.5s 一段 |

### 4. 智驾特有约束（相对图像 VQ）

- **物理可行性**：曲率、加速度、jerk 可执行；解码前可做车道拓扑过滤。
- **多样性**：覆盖车道保持、跟车、变道、左转、避让等基元，避免码向量扎堆。
- **分层/残差**：高层行为 token + 底层残差修正（\(\tau = a_k + r_k\) 一类思路）。
- **距离度量**：不宜只用坐标 L2；可考虑 DTW、Frenet、轮廓距离等，更好反映航向/速度差异。

### 5. VQ-VAE 原理要点

- 码本 \(E=\{e_i\}_{i=1}^K\)；\(k=\arg\min_i\|z_e-e_i\|^2\)，\(z_q=e_k\)。
- **STE**：前向用 \(e_k\)，反向梯度直通到 \(z_e\)。
- 损失：\(L = L_{rec} + L_{codebook} + \beta L_{commit}\)（\(\beta\) 常取 0.25）；**无 KL 项**（潜变量是离散索引）。
- 相对普通 VAE：离散化利于自回归 Transformer，缓解后验塌陷一类问题（表述需结合原文语境）。

### 6. 训练与推理流水线

1. 学 Codebook：轨迹 ↔ 索引 ↔ 重建。
2. VLA 预训练：图像/文本 token + 轨迹 token 联合 next-token。
3. 推理：预测 token → 查表原型 → 拼接长轨迹；可加安全校验 / reflection 替换不安全 token。

### 7. 常见变种

- **CVQ-VAE**：条件于场景/地图再量化。
- **RQ-VAE**：多层残差量化，小码本换高精度。
- **NSVQ / Gumbel-VQ**：缓解码本坍缩（大量码永不被用）。

## 知识点校验

- **Correct**：VQ-VAE 三项损失 + STE 的标准形式表述正确；局部坐标与短窗切片是轨迹码本工程共识。
- **Caution**：原文点名的 AutoVLA / LAMP / AnchorVLA / ReflectDrive 的 \(K\)、窗口、是否分层等细节可能随版本变化，引用前需核对论文/开源，勿当作统一工业标准。
- **Caution**：「VQ-VAE 规避 posterior collapse」是相对连续 VAE 的常见说法，但 VQ 自身有 **codebook collapse** 问题，不能解读为「离散化就一劳永逸」。
- **Expand**：码本利用率监控、EMA codebook 更新、与连续 Flow Matching Action Expert 的混合（离散决策 + 连续残差）选型。

## 延伸阅读建议

- VQ-VAE（van den Oord et al.）；VQ-VAE-2；RQ-VAE
- 运动基元 / 轨迹离散化在规划与 VLA 中的代表性工作（以最新开源 VLA 规划论文为准逐篇核对）
- 码本坍缩对策：EMA、重置死码、Gumbel-Softmax / NSVQ 文献

## 与其他主题的关联

- 连续 Action Expert（DiT/Flow）对照：[`../notes/10-action-expert-dit.md`](../notes/10-action-expert-dit.md)
- Qwen3-VL VLA 中感知/轨迹 Codebook 与 MoE：[`../notes/16-qwen3vl-vla-architecture.md`](../notes/16-qwen3vl-vla-architecture.md)
- 规控真值与负样本构造：[`../notes/12-planning-dpo-rl.md`](../notes/12-planning-dpo-rl.md)
- VLA 决策歧义与标注一致性：[`../notes/08-vla-ambiguity-coc.md`](../notes/08-vla-ambiguity-coc.md)
- 因果建模与轨迹筛选：[`../notes/05-causal-modeling-trajectory.md`](../notes/05-causal-modeling-trajectory.md)
