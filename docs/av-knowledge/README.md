# 自动驾驶知识体系

把 **17 篇豆包对话** + **28 页手写笔记** 整理为：精炼笔记（校验+扩展）→ 分类 → Roadmap → 整体体系分析。

| | |
|--|--|
| 交互总览 | [自动驾驶知识体系 Roadmap（Canvas）](https://cursor.com/canvas/store-76ea721c-6842-482e-a020-029a6137026c/5078d7c4-19f8-46fe-8707-2e041b120d6b) |
| 笔记 | [`notes/`](notes/) · **01–17** 豆包 · **18–30** 手写 |
| 分类 | [`taxonomy.md`](taxonomy.md) |
| 学习路径 | [`roadmap/LEARNING_ROADMAP.md`](roadmap/LEARNING_ROADMAP.md) |
| **整体体系分析** | [`roadmap/INTEGRATED_SYSTEM.md`](roadmap/INTEGRATED_SYSTEM.md) |
| 豆包原文 | [`sources/`](sources/) |
| 手写转录与页索引 | [`handwriting/`](handwriting/) · [`handwriting/INDEX.md`](handwriting/INDEX.md) |

> 配比、张量形状、内部评测分数多为工程快照，笔记中已标 **Caution**；落地前以代码/配置复验。

---

## 文档索引

### 豆包精炼（01–17）

| # | 笔记 | 分类 | 原对话 |
|---|------|------|--------|
| 01 | [VLM 智驾训练阶段](notes/01-vlm-training-stages.md) | VLA 管线 | [链接](https://www.doubao.com/thread/xFP6Nlw5PGBuoWjtD) |
| 02 | [BEV + PNC 技术栈](notes/02-bev-pnc-stack.md) | 感知/规控全景 | [链接](https://www.doubao.com/thread/x2KWll9wqjafJJp2B) |
| 03 | [在线蒸馏 OPD](notes/03-online-policy-distillation.md) | 蒸馏对齐 | [链接](https://www.doubao.com/thread/xzgL8jt1IX592B2Gy) |
| 04 | [训练效率优化](notes/04-training-efficiency.md) | 训练工程 | [链接](https://www.doubao.com/thread/x2oYTj7lKbhtwQJaA) |
| 05 | [因果建模与轨迹](notes/05-causal-modeling-trajectory.md) | 因果/Shortcut | [链接](https://www.doubao.com/thread/x0mTggTOwHEL0vzCy) |
| 06 | [AdamW 优化器](notes/06-adamw-optimizer.md) | 训练工程 | [链接](https://www.doubao.com/thread/x4FdX61BhJAMV75Zq) |
| 07 | [端到端数据 Trick](notes/07-e2e-data-tricks.md) | 数据闭环 | [链接](https://www.doubao.com/thread/xoqQSgCUH1RcXmBYT) |
| 08 | [VLA 歧义与 CoC](notes/08-vla-ambiguity-coc.md) | 决策语义 | [链接](https://www.doubao.com/thread/x77gV8d56RaJ3cJ5W) |
| 09 | [Alpamayo Reasoning-VQA](notes/09-alpamayo-reasoning-vqa.md) | 推理对齐 | [链接](https://www.doubao.com/thread/xvmTCVw3wgaz5As68) |
| 10 | [Action Expert DiT](notes/10-action-expert-dit.md) | 动作生成 | [链接](https://www.doubao.com/thread/xA6rDOr9aEUJfSJ9L) |
| 11 | [轨迹 Codebook / VQ-VAE](notes/11-trajectory-codebook.md) | 动作表示 | [链接](https://www.doubao.com/thread/x3YpYhDmOsLZs5Ow7) |
| 12 | [规控 DPO / GRPO](notes/12-planning-dpo-rl.md) | 偏好对齐 | [链接](https://www.doubao.com/thread/xMGx5DJo7VzWmYCZ6) |
| 13 | [BEV 深度模型对比](notes/13-bev-depth-models.md) | 感知 | [链接](https://www.doubao.com/thread/xgeM04177ZmZm4AnV) |
| 14 | [PV Backbone 与 View Transform](notes/14-pv-backbone-view-transform.md) | 感知 | [链接](https://www.doubao.com/thread/xTGPZXuruwC5E92Xm) |
| 15 | [训练推理全栈](notes/15-llm-train-infer-stack.md) | 工程栈 | [链接](https://www.doubao.com/thread/xw7zUWEkYkxGOoTJG) |
| 16 | [Qwen3-VL VLA 架构](notes/16-qwen3vl-vla-architecture.md) | 系统拼装 | [链接](https://www.doubao.com/thread/xuayeFKhBhI8yo8yX) |
| 17 | [持续学习抗遗忘](notes/17-continual-learning-vla.md) | 持续学习 | [链接](https://www.doubao.com/thread/x06hNIFYKk3nhcd6V) |

### 手写精炼（18–30 ← Page 01–28）

| # | 笔记 | 分类 | 来源页 |
|---|------|------|--------|
| 18 | [VLA 选型与 L3/L4 路线](notes/18-hand-vla-l3l4-roadmap.md) | VLA / 落地 | 01 |
| 19 | [LLM→VL→4D 训练管线](notes/19-hand-vl-4d-training-pipeline.md) | VLA 管线 | 02 |
| 20 | [结构化感知 Navi/Traffic/Ego](notes/20-hand-structured-perception.md) | 感知中间层 | 03 |
| 21 | [多传感器 BEV/PV 与检测头](notes/21-hand-multisensor-bev-pv.md) | 感知全链路 | 04–05 |
| 22 | [Ego 推演 / Teacher Forcing / KV](notes/22-hand-ego-process-meta-action.md) | 决策–动作 | 06–07 |
| 23 | [数据 SFT / Agent / CoC Reward](notes/23-hand-data-sft-agent-loop.md) | 数据与对齐 | 08–11 |
| 24 | [Scene Graph / 流式 4D / Memory](notes/24-hand-scenegraph-streaming-4d.md) | 4D 与具身 | 12–14 |
| 25 | [Video Thinking / Latent WM](notes/25-hand-video-thinking-latent-wm.md) | 世界模型 | 15–16 |
| 26 | [Token 架构 / Occ / Sparse BEV](notes/26-hand-token-arch-occ-sparse.md) | 感知时序 | 17–18 |
| 27 | [OPD / Toggle / STL / CoT](notes/27-hand-opd-grpo-toggle-stl-cot.md) | 蒸馏与约束规划 | 19–21 |
| 28 | [模型档位 FM↔DiT](notes/28-hand-model-tiers-fm-dit.md) | 系统缩放 | 22–24 |
| 29 | [Swift / Megatron 训练](notes/29-hand-swift-megatron-train.md) | 训练工程 | 25–26 |
| 30 | [FM 全栈架构总览](notes/30-hand-fm-full-stack-architecture.md) | 端到端系统 | 27–28 |

---

## 怎么用

1. 先读 [整体体系分析](roadmap/INTEGRATED_SYSTEM.md) 建立五层地图  
2. 按 [Roadmap](roadmap/LEARNING_ROADMAP.md) 或体系文第 6 节顺序推进  
3. 用 [taxonomy](taxonomy.md) 按角色查阅  
4. 手写页级回溯：[handwriting/INDEX.md](handwriting/INDEX.md)

## 整理原则

- **精炼**：笔记不复述全文/全页，保留可复用结构与工程要点  
- **校验**：Correct / Caution / Expand  
- **纠错**：手写缩写多义（MOT、CoC/COC）在 INDEX 与各篇注明  
- **关联**：豆包原理 ↔ 手写工程双向链接  
