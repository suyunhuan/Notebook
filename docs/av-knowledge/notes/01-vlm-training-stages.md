# VLM 智驾训练阶段

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | VLM智驾模型训练阶段整理 |
| Thread ID | `xFP6Nlw5PGBuoWjtD` |
| 源链接 | https://www.doubao.com/thread/xFP6Nlw5PGBuoWjtD |
| 所属分类 | 智能驾驶核心技术 / VLM–VLA 训练管线 |
| 源文件 | `../sources/xFP6Nlw5PGBuoWjtD.md` |

## 一句话摘要

把通用 VLM 接到智驾 VLA，需按「视觉对齐 → 4D/环视对齐 → 垂域增训 → 难例后训练 → Planner 对齐与联训」分阶段解冻参数，并配合动态帧率与 replay 配比控制遗忘。

## 核心关键词

VLM；VLA；4D 编码器；连接器 / Projector；Action Expert (AE)；DriveCore；OPD；CoC；动态帧率（6s2Hz / 2s10Hz）；replay；指令控车（VLA→T / VL→AT）

## 知识脉络

### 1. 总体设计原则

- **先对齐再扩能力**：早期只训连接器，保护 LLM / ViT；再逐步解冻视觉编码器、4D 编码器、LLM、Planner。
- **通识与垂域交替**：垂域增训后必须用通用决策 / VQA replay，避免 catastrophic forgetting。
- **语义决策可离散化**：后训练把语义决策落到 special token / meta-action，再与轨迹监督对齐。

### 2. 分阶段要点（目的 / 可训参数 / 数据）

| 阶段 | 目的 | 可训练参数（要点） | 数据与配比要点 |
|------|------|-------------------|----------------|
| 1 视觉理解 | 图文高层语义对齐，不破坏 LLM/ViT | 连接器 | 简单自然图、大字文档 |
| 2 预训练+SFT | 细粒度图文/视频–文本对齐 | ViT+连接器，LLM | 复杂图像与文档 |
| 3 4D 对齐 VLM | 环视时空编码器与 VLM 对齐 | 4D+连接器，ViT+连接器，LLM | 环视图–文本 |
| 4.1 FM/4D→LLM（可选） | 外接 FM/4D 补空间理解 | 4D 连接器为主 | VFLM（环视–文本）、VFLA（环视–决策） |
| 4.2 智驾垂域增训 | 保持 VLM，新增智驾决策 | 4D 连接器、ViT+连接器、LLM | 动态帧率；DriveCore ~550K；VLA 视频 ~100w；OPD 中文分析/决策；CODA-LM |
| 5 VLM(+AE) 后训练 | replay 通用决策；加 CoC、难例、轨迹监督 | 时空编码器+连接器、ViT+连接器、LLM LoRA | 2s10Hz；VIL→AP ~630w；VIL→RAP ~30w（含难例） |
| 6.1 VLA 对齐 | 对齐 VLM(4&5) 与 AE(5) | planner 连接器 + planner | VLAT ~144w |
| 6.2.x 联训 | 指令导航控车、数据扩量、replay 固化 | 逐步解冻 proj / LLM / planner（常冻 ViT/FM） | 单指令 L3 + 复合 L2；VFSL→AT 扩量；RAT 难例；warm-up 前置 VLA→T |

### 3. 工程要点

- **动态帧率**：长时语义用 6s@2Hz；短时控制/难例用 2s@10Hz。
- **课程式指令**：warm-up 先做 `VLA→T`（给定动作学轨迹），再做 `VL→AT` / reasoning 控车，降低联合难度。
- **联训两段式**：大量 AT 主训后，用有限预算 replay AT + RAT，推理侧可从 VLAT 切到 VLRAT。
- **规模信号（原文内部评测）**：DriveBench V4.5 上 2B/4B/8B 分数约 61/68/72——仅作内部 scaling 参考，非公开基准结论。

### 4. 关键概念速记

- **连接器 / Projector**：视觉（或 4D）token → LLM 词嵌入空间的桥。
- **AE (Action Expert)**：轨迹/动作头，常与 DiT / Flow Matching 联用。
- **OPD**：在线策略蒸馏，用更强 teacher 或历史强版本补齐分析/决策（见 OPD 笔记）。

## 知识点校验

1. **Caution**：`2012：VLM 多模态模型训练` 等备注疑似内部项目代号/笔误，不能当作 2012 年公开技术时间线。
2. **Caution**：DriveBench V4.5 分数量级、数据配比（550K / 100w / 630w 等）均为对话内工程口径，外推到其他团队或公开榜需重新验证。
3. **Expand**：阶段 4.2 与 5「是否 merge」未决——工程上可对比「垂域 SFT → 难例后训」两段 vs 混合单段，用通用 VQA + 难例闭环双指标做消融。
4. **Correct**：先冻主干只训连接器、再逐步解冻，是 LLaVA / InstructBLIP 类多模态训练的主流稳定策略。
5. **Expand**：`VLA→T` / `VL→AT` / `VLRAT` 命名需与动作码本、CoC 标注口径统一，否则联训阶段会出现标签不一致（关联 VLA 歧义笔记）。

## 延伸阅读建议

- LLaVA / InstructBLIP（连接器对齐与两阶段训）
- Qwen2-VL / Qwen3-VL Technical Report（强弱蒸馏与视频–文本对齐）
- DriveLM、CODA-LM（驾驶 VQA / 难例语言监督）
- DiT + Flow Matching Action Expert、轨迹 Codebook / VQ-VAE（AE 侧）

## 与其他主题的关联

- [在线蒸馏 OPD](../notes/03-online-policy-distillation.md) — 垂域阶段 OPD 数据与大蒸小
- [VLA 歧义与 CoC](../notes/08-vla-ambiguity-coc.md) — 语义决策 special token / 标注一致性
- [Alpamayo Reasoning-VQA](../notes/09-alpamayo-reasoning-vqa.md) — CoC 作为推理条件而非事后解释
- [端到端数据 Trick](../notes/07-e2e-data-tricks.md) — 难例与配比工程
- [因果建模与轨迹筛选](../notes/05-causal-modeling-trajectory.md) — A→T 链路与瓶颈分诊
- [BEV+PNC 技术栈](../notes/02-bev-pnc-stack.md) — Planner / DiT 侧结构
- [训练效率](../notes/04-training-efficiency.md) / [AdamW](../notes/06-adamw-optimizer.md) — 多阶段大规模训练落地
