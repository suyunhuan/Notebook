# 因果建模与决策–轨迹筛选

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | 因果建模与决策-轨迹筛选及提升方案 |
| Thread ID | `x0mTggTOwHEL0vzCy` |
| 源链接 | https://www.doubao.com/thread/x0mTggTOwHEL0vzCy |
| 所属分类 | 智能驾驶核心技术 / 因果与难例治理 |
| 源文件 | `../sources/x0mTggTOwHEL0vzCy.md` |

## 一句话摘要

VLA 直接图像→轨迹易学相关性与 shortcut；更稳妥的是建立 \(C \to A \to T\) 因果中介链，用高重叠偏好、意图条件生成、dropout/注意力正则与反事实自检分离真因，并用 do-探针把难例路由到 VQA / 慢思考 / 白盒等手段。

## 核心关键词

因果混淆；Shortcut；自车状态泄露；Meta-action / Intent \(z\)；High-overlap；CFG；Counterfactual；do-calculus；Bottleneck 分诊；\(\Delta_A\)

## 知识脉络

### 1. 两大病根

**多因混叠**

- 一条轨迹由道路结构、障碍物、自车状态、导航等叠加决定；端到端学的是联合共现，不是可干预因果。
- 典型：训练中「行人」常与「前车刹车」共现 → 测试仅行人时失效。

**Shortcut / 泄露**

- 图像隐含自车姿态、地平线、光流 → 偷映射速度/轨迹，换相机/平台即崩。
- 指路牌位置、指令–像素伪关联等虚假线索。

**后果**：不可解释、OOD 脆、仿真–实车 gap、难注入硬安全约束。

### 2. 分层解法组合（文献拼图，工程视角）

| 层 | 思路 | 作用 |
|----|------|------|
| 架构 | Intent \(z\) + 条件轨迹（CFG / flow-matching） | \(z\) 作因果中介，可 do(\(z\)) 探针 |
| 数据 | High-overlap 偏好对（仅关键决策点分叉） | 梯度聚焦分叉，抑共现污染 |
| 表征 | Prompt/元信息 dropout；注意力 RL 约束 | 打掉自车泄露与背景捷径 |
| 推理 | 反事实自省（移除对象重推） | 上线前筛「只学相关」case |
| 系统 | VLA 出结构化意图 + 带约束规划器 | 缩小连续轨迹上的混淆空间 |

### 3. \(C \to A \to T\) 分诊框架（核心可操作部分）

- **准则一（症状）**：\(P(T \mid \hat A, C)\) 差 → 进入难例池。
- **准则二（诊断）**：\(P(T \mid do(A{=}A^*), C) > P(T \mid do(A{=}\hat A), C)\)  
  \(\Delta_A\) 大 → **A-bottleneck**（投慢思考 / VQA / 白盒提升决策）；  
  \(\Delta_A\) 小且 \(P(T \mid A^*,C)\) 仍差 → **T-head 或 C 信息不足**。
- **平行探针**：do(reasoning)、do(whitebox)、do(VQA/knowledge) 得 \(\Delta_R,\Delta_W,\Delta_K\)，按 max\(\Delta\) 路由算力。
- **实现依赖**：intent-conditioned rollout（同一 \(C\) 下分别喂 \(\hat A\) / \(A^*\)）；flow 采样需多次取期望，避免把噪声当因果效应。

### 4. 强因果样本与训练闭环

- **强因果筛选**：仅 do(因) 后果显著变化的样本入池，按因果因子分组监控 loss。
- **High-overlap 中段失效**：公共前缀过强时中段因子被淹没 → 截断评估窗口、中间偏好标签、冻结后端防补偿、反事实预估过滤。
- **一致性 reward 陷阱**：决策与轨迹「都错但巧合对齐」会得正奖励 → 需分层 reward / 因果约束，避免强化错误链。
- **数据效率观**：少量高信息强因果难例微调，往往优于海量弱相关全参重训（与 lottery ticket / bits-that-count 叙事同向，需实证）。

## 知识点校验

1. **Caution**：对话引用的 Counterfactual VLA、Polaris2、MIND-VLA、DIAL、π0.7 等拼接方案是**工程假说组合**，不代表任一论文已验证整条智驾闭环；落地需分模块复现。
2. **Caution**：用观测条件概率差近似 \(do(\cdot)\) 仍可能被混淆；high-overlap 是缓解而非严格因果识别。
3. **Expand**：从示范轨迹反推 \(A^*\) 本身歧义大（一轨多意图），需置信度过滤，否则 \(\Delta_A\) 探针系统性偏置。
4. **Correct**：自车状态从图像泄露、共现 shortcut、意图–轨迹解耦，是端到端驾驶文献中反复确认的风险点。
5. **Expand**：分组 loss 监控存在跨组特征泄漏（「路口」同时影响多组），需交叉 held-out 检验独立性。

## 延伸阅读建议

- DAgger / 因果模仿学习经典讨论（exposure bias 与 confounding）
- Classifier-Free Guidance；Intent-conditioned planning / RL（如 DIAL 一类工作）
- Counterfactual reasoning in VLA；high-overlap preference 数据构造
- SOTIF 与可审计决策链（与 CoC / Reasoning-VQA 衔接）

## 与其他主题的关联

- [VLA 歧义与 CoC](../notes/08-vla-ambiguity-coc.md) — 决策可验证离散化与因果时序标注
- [Alpamayo Reasoning-VQA](../notes/09-alpamayo-reasoning-vqa.md) — 推理作动作条件、反事实反思
- [在线蒸馏 OPD](../notes/03-online-policy-distillation.md) — 按瓶颈选教师蒸馏
- [数据 Trick](../notes/07-e2e-data-tricks.md) — 交互难例挖掘与负决策样本
- [VLM 训练阶段](../notes/01-vlm-training-stages.md) — VLA→T 课程与 A–T 一致性
- [BEV+PNC](../notes/02-bev-pnc-stack.md) — 规划约束层承接意图
