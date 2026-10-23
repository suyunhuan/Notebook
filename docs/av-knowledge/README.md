# 自动驾驶对话知识体系

把 17 篇豆包技术对话整理为：**精炼笔记（校验+扩展）→ 分类体系 → 学习 Roadmap**。

| | |
|--|--|
| 交互总览 | [自动驾驶知识体系 Roadmap（Canvas）](https://cursor.com/canvas/store-76ea721c-6842-482e-a020-029a6137026c/5078d7c4-19f8-46fe-8707-2e041b120d6b) |
| 笔记 | [`notes/`](notes/) · 17 篇 |
| 分类 | [`taxonomy.md`](taxonomy.md) |
| Roadmap | [`roadmap/LEARNING_ROADMAP.md`](roadmap/LEARNING_ROADMAP.md) |
| 原文存档 | [`sources/`](sources/) |

> 笔记中的配比数字、内部评测分数多为对话工程口径，已用 **Caution** 标注；落地前请用自有数据复验。

---

## 文档索引

| # | 笔记 | 分类 | 原对话 |
|---|------|------|--------|
| 01 | [VLM 智驾训练阶段](notes/01-vlm-training-stages.md) | VLA 管线 | [xFP6Nlw5PGBuoWjtD](https://www.doubao.com/thread/xFP6Nlw5PGBuoWjtD) |
| 02 | [BEV + PNC 技术栈](notes/02-bev-pnc-stack.md) | 感知/规控全景 | [x2KWll9wqjafJJp2B](https://www.doubao.com/thread/x2KWll9wqjafJJp2B) |
| 03 | [在线蒸馏 OPD](notes/03-online-policy-distillation.md) | 蒸馏对齐 | [xzgL8jt1IX592B2Gy](https://www.doubao.com/thread/xzgL8jt1IX592B2Gy) |
| 04 | [训练效率优化](notes/04-training-efficiency.md) | 训练工程 | [x2oYTj7lKbhtwQJaA](https://www.doubao.com/thread/x2oYTj7lKbhtwQJaA) |
| 05 | [因果建模与轨迹](notes/05-causal-modeling-trajectory.md) | 因果/Shortcut | [x0mTggTOwHEL0vzCy](https://www.doubao.com/thread/x0mTggTOwHEL0vzCy) |
| 06 | [AdamW 优化器](notes/06-adamw-optimizer.md) | 训练工程 | [x4FdX61BhJAMV75Zq](https://www.doubao.com/thread/x4FdX61BhJAMV75Zq) |
| 07 | [端到端数据 Trick](notes/07-e2e-data-tricks.md) | 数据闭环 | [xoqQSgCUH1RcXmBYT](https://www.doubao.com/thread/xoqQSgCUH1RcXmBYT) |
| 08 | [VLA 歧义与 CoC](notes/08-vla-ambiguity-coc.md) | 决策语义 | [x77gV8d56RaJ3cJ5W](https://www.doubao.com/thread/x77gV8d56RaJ3cJ5W) |
| 09 | [Alpamayo Reasoning-VQA](notes/09-alpamayo-reasoning-vqa.md) | 推理对齐 | [xvmTCVw3wgaz5As68](https://www.doubao.com/thread/xvmTCVw3wgaz5As68) |
| 10 | [Action Expert DiT](notes/10-action-expert-dit.md) | 动作生成 | [xA6rDOr9aEUJfSJ9L](https://www.doubao.com/thread/xA6rDOr9aEUJfSJ9L) |
| 11 | [轨迹 Codebook / VQ-VAE](notes/11-trajectory-codebook.md) | 动作表示 | [x3YpYhDmOsLZs5Ow7](https://www.doubao.com/thread/x3YpYhDmOsLZs5Ow7) |
| 12 | [规控 DPO / GRPO](notes/12-planning-dpo-rl.md) | 偏好对齐 | [xMGx5DJo7VzWmYCZ6](https://www.doubao.com/thread/xMGx5DJo7VzWmYCZ6) |
| 13 | [BEV 深度模型对比](notes/13-bev-depth-models.md) | 感知 | [xgeM04177ZmZm4AnV](https://www.doubao.com/thread/xgeM04177ZmZm4AnV) |
| 14 | [PV Backbone 与 View Transform](notes/14-pv-backbone-view-transform.md) | 感知 | [xTGPZXuruwC5E92Xm](https://www.doubao.com/thread/xTGPZXuruwC5E92Xm) |
| 15 | [训练推理全栈](notes/15-llm-train-infer-stack.md) | 工程栈 | [xw7zUWEkYkxGOoTJG](https://www.doubao.com/thread/xw7zUWEkYkxGOoTJG) |
| 16 | [Qwen3-VL VLA 架构](notes/16-qwen3vl-vla-architecture.md) | 系统拼装 | [xuayeFKhBhI8yo8yX](https://www.doubao.com/thread/xuayeFKhBhI8yo8yX) |
| 17 | [持续学习抗遗忘](notes/17-continual-learning-vla.md) | 持续学习 | [x06hNIFYKk3nhcd6V](https://www.doubao.com/thread/x06hNIFYKk3nhcd6V) |

---

## 怎么用

1. **建立体系**：先看 [Roadmap](roadmap/LEARNING_ROADMAP.md) 四阶段路径  
2. **按主题查**：用 [分类体系](taxonomy.md)  
3. **深挖一篇**：打开对应 `notes/`，关注文末 **知识点校验** 与 **关联**  
4. **回溯原文**：`sources/<thread_id>.md`

## 整理原则

- **精炼**：笔记不复述全文，保留可复用结构与工程要点  
- **校验**：Correct / Caution / Expand 三类标注  
- **关联**：跨笔记双向链接，避免孤岛知识点  
- **缺口**：Roadmap 第 3 节列出对话未覆盖的 Occupancy、仿真闭环、功能安全等
