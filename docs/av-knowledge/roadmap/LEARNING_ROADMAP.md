# 自动驾驶大模型知识体系 · Roadmap

目标：把 17 篇对话从「零散问答」变成可执行的学习与工程路线。  
假设读者：有深度学习基础，正在做或计划做 **端到端 / VLA 智驾**。

---

## 0. 你最终要建立的能力地图

| 能力层 | 要能回答的问题 | 对应笔记 |
|--------|----------------|----------|
| L1 感知表征 | PV 怎么变 BEV？深度估计在哪用？ | 13, 14, 02 |
| L2 多模态基座 | VLM 分几阶段接到车上？冻什么、解冻什么？ | 01, 09 |
| L3 动作接口 | 轨迹是回归、码本还是扩散？条件怎么注入？ | 11, 10, 08 |
| L4 因果与对齐 | 如何避免 shortcut？如何做偏好与蒸馏？ | 05, 07, 12, 03 |
| L5 系统落地 | 用什么框架训、推、防遗忘？ | 15, 04, 06, 16, 17 |

---

## 1. 四阶段学习路径

### Phase 1 · 感知与表征基础（先把「看见」打牢）

**产出**：能画出「多相机 → Backbone → View Transform → BEV → 检测/占用」框图，并说明 LSS 与 BEVFormer 差异。

1. [14 PV→BEV](../notes/14-pv-backbone-view-transform.md) — Backbone/Neck/View Transform
2. [13 BEV 深度模型](../notes/13-bev-depth-models.md) — LSS 系 vs Query 系
3. [02 BEV+PNC 栈](../notes/02-bev-pnc-stack.md) — 感知如何接到规控

**自检**

- [ ] 能解释 Lift / Splat / depth distribution
- [ ] 知道时序 BEV 为何要 ego-motion 补偿
- [ ] 能说出量产常见选 CNN backbone 的原因

**扩展练习**：用开源 BEVDet 或 BEVFormer 跑通 nuScenes 检测 demo，改一版 depth supervision。

---

### Phase 2 · VLM → VLA 主干（把「理解」接到「动作」）

**产出**：设计一版分阶段训练表（目的 / 可训参数 / 数据类型 / 配比）。

1. [01 VLM 训练阶段](../notes/01-vlm-training-stages.md)
2. [09 Alpamayo Reasoning-VQA](../notes/09-alpamayo-reasoning-vqa.md)
3. [08 决策歧义与 CoC](../notes/08-vla-ambiguity-coc.md)
4. [11 轨迹 Codebook](../notes/11-trajectory-codebook.md)
5. [10 Action Expert DiT](../notes/10-action-expert-dit.md)

**自检**

- [ ] 能区分 VLA→T / VL→AT / Reasoning→Action
- [ ] 能说明为何自由文本动作有歧义，meta-action / codebook 如何消歧
- [ ] 能解释 adaLN-Zero 如何注入 ego + diffusion timestep
- [ ] 能对比「连续回归轨迹」vs「离散码本」vs「Flow Matching」

**扩展练习**：用小规模轨迹数据训一个 VQ-VAE codebook；或实现一层 DiT block 的 adaLN 条件注入。

---

### Phase 3 · 数据、因果与对齐（把「学对」做扎实）

**产出**：一份数据闭环 SOP（挖掘 → 加权 → 构造 → 清洗 → 评测）+ DPO 负样本规范。

1. [07 端到端数据 Trick](../notes/07-e2e-data-tricks.md)
2. [05 因果建模](../notes/05-causal-modeling-trajectory.md)
3. [12 DPO / GRPO 规控](../notes/12-planning-dpo-rl.md)
4. [03 在线蒸馏 OPD](../notes/03-online-policy-distillation.md)

**自检**

- [ ] 能举 3 类智驾 shortcut（自车状态泄露、共现虚假因果等）及缓解
- [ ] 能写出 chosen/rejected **同场景配对**规则与弱负/强负比例原则
- [ ] 理解 OPD「学生采样 + 教师 token 监督」与离线蒸馏差异

**扩展练习**：在仿真里构造一对「行人单独出现 vs 行人+前车刹车」对照 clip，做干预式评测。

---

### Phase 4 · 工程落地与持续迭代

**产出**：选定训练/推理栈，画出「预训练 → SFT → DPO → 部署」流水线；明确防遗忘策略。

1. [15 训练推理栈](../notes/15-llm-train-infer-stack.md)
2. [04 训练效率](../notes/04-training-efficiency.md)
3. [06 AdamW](../notes/06-adamw-optimizer.md)
4. [16 Qwen3-VL VLA 架构](../notes/16-qwen3vl-vla-architecture.md) — 端到端拼装蓝图
5. [17 持续学习](../notes/17-continual-learning-vla.md)

**自检**

- [ ] 说清 Swift / Megatron / vLLM（或 TRT-LLM）各自阶段职责
- [ ] 知道 FlashAttention / ZeRO 解决的是算子还是显存切分
- [ ] 对「预训练邻域专家」与「智驾长尾仍需 replay」有边界意识

**扩展练习**：用 Swift 对 Qwen2/3-VL 做 LoRA SFT 小样本驾驶 VQA；或用 vLLM 压测同模型吞吐。

---

## 2. 推荐「最小可行项目」路径（MVP）

若目标是亲手跑通一条 VLA 雏形，建议顺序：

```
① 冻结 VLM + 训 Projector（驾驶 VQA）
      ↓
② 加轨迹监督头（先回归，后可选 codebook）
      ↓
③ 引入 meta-action / 结构化决策（消歧）
      ↓
④ Action Expert：Flow Matching / DiT + ego 条件
      ↓
⑤ 数据加权 + 难例清洗闭环
      ↓
⑥ 离线 DPO（同场景弱负）
      ↓
⑦ OPD / replay 防遗忘 + 推理侧 guardrail
```

对应笔记串联：`01 → 08 → 11/10 → 07 → 12 → 03/17 → 15`。

---

## 3. 知识缺口清单（对话覆盖不足，建议后续补）

| 缺口 | 为何重要 | 建议补读方向 |
|------|----------|--------------|
| 占用网络 Occupancy / 3D Gaussian | 规划常用中间表征 | OccNet、SurroundOcc、GaussianFormer |
| 预测（多智能体） | 模块化与 E2E 交界 | HiVT、MTR、Wayformer |
| 仿真与闭环评测 | DPO/RL 前提 | CARLA、NeuroNCAP、NAVSIM、Bench2Drive |
| 安全与功能安全 | 上车硬约束 | SOTIF、运行时 monitor、规则兜底 |
| 量产部署（MCU/域控） | 延迟与量化 | TensorRT、量化感知训练、传感器时间同步 |
| 高精地图 / 无图导航 | 定位与 route 条件 | MapTR、LaneSegNet、导航 token |

---

## 4. 每周节奏建议（可裁剪）

| 周次 | 焦点 | 交付物 |
|------|------|--------|
| W1 | Phase 1 感知 | BEV 路线对比一页纸 |
| W2–W3 | Phase 2 VLA | 训练阶段表 + 动作接口选型 |
| W4 | Phase 3 数据/因果 | 数据 SOP + shortcut 清单 |
| W5 | Phase 3 对齐 | DPO 负样本规范草案 |
| W6 | Phase 4 工程 | 训推流水线 + MVP 里程碑 |

---

## 5. 使用本知识库的方式

1. 先读 [分类体系](../taxonomy.md) 定位自己缺口  
2. 按本 Roadmap 阶段推进，每篇笔记底部有「关联」可横向跳转  
3. 原文全文在 [sources/](../sources/)，笔记是校验后的精炼版——**以笔记为准做体系，以原文做细节回溯**  
4. 笔记中的 **Caution** 优先复查；数字配比一律当经验区间  

原始对话索引见 [README](../README.md)。
