# 手写笔记页码索引

28 页手写照片 → 纠错转录（`_raw_batch_*.md`）→ 主题精炼笔记（`notes/18–30`）。

> 原图体积较大（约 120MB），未入库；本地路径见下方「原图哈希」。精炼笔记为准，转录供回溯。

## 页码 → 精炼笔记

| 页 | 原图哈希前缀 | 主题（纠错后） | 精炼笔记 |
|----|--------------|----------------|----------|
| 01 | a3f9-7d92 | VLA 选型 / L3–L4 / MOT 离散–连续 | [18](../notes/18-hand-vla-l3l4-roadmap.md) |
| 02 | a4bb-7ae7 | LLM→VL→4D 多阶段训练 | [19](../notes/19-hand-vl-4d-training-pipeline.md) |
| 03 | a59b-7cb0 | NaviNN / TrafficNN / Ego-HMF | [20](../notes/20-hand-structured-perception.md) |
| 04 | a659-7b66 | 多传感器 BEV/PV 全链路多头 | [21](../notes/21-hand-multisensor-bev-pv.md) |
| 05 | a71a-727f | BEV CenterNet 检测头与 loss 加权 | [21](../notes/21-hand-multisensor-bev-pv.md) |
| 06 | a7d4-77eb | Ego 推演 / TTC / 风险降速 | [22](../notes/22-hand-ego-process-meta-action.md) |
| 07 | a892-746c | Teacher Forcing / meta-action / KV→FM | [22](../notes/22-hand-ego-process-meta-action.md) |
| 08 | a954-756b | VQA/轨迹 SFT 多样性与固化 | [23](../notes/23-hand-data-sft-agent-loop.md) |
| 09 | aa0d-72b9 | MessageBus Agent Loop | [23](../notes/23-hand-data-sft-agent-loop.md) |
| 10 | aad4-705f | ViT 族对比 + Agent 骨架 | [23](../notes/23-hand-data-sft-agent-loop.md) |
| 11 | abac-72e0 | 模态 Dropout / CoC Reward | [23](../notes/23-hand-data-sft-agent-loop.md) |
| 12 | ac7a-7c16 | Scene Graph + 慢快 Rollout→DPO | [24](../notes/24-hand-scenegraph-streaming-4d.md) |
| 13 | ad35-7f5e | 流式 4D：KV + Router + SSM | [24](../notes/24-hand-scenegraph-streaming-4d.md) |
| 14 | adf6-7dae | 具身 FM：VLM+WM+Critic / Memory | [24](../notes/24-hand-scenegraph-streaming-4d.md) |
| 15 | aeb1-783e | Video Thinking / Action Gen | [25](../notes/25-hand-video-thinking-latent-wm.md) |
| 16 | af72-7ba9 | 4D Encoder→MoT；Latent Reasoning | [25](../notes/25-hand-video-thinking-latent-wm.md) |
| 17 | b036-724c | 时序 Attn + 多模 4D；S0–S3 | [26](../notes/26-hand-token-arch-occ-sparse.md) |
| 18 | b114-7a46 | 多 tokenizer；Occ/Sparse/Scene | [26](../notes/26-hand-token-arch-occ-sparse.md) |
| 19 | b1e6-784c | OPD/MOPD + special token 训参 | [27](../notes/27-hand-opd-grpo-toggle-stl-cot.md) |
| 20 | b2c5-76a3 | 词元 Clip + Toggle 预算优化 | [27](../notes/27-hand-opd-grpo-toggle-stl-cot.md) |
| 21 | b39f-77f1 | STL / CoT / 扩散规划 | [27](../notes/27-hand-opd-grpo-toggle-stl-cot.md) |
| 22 | b483-74fe | Max/Ultra/Ultimate 档位 | [28](../notes/28-hand-model-tiers-fm-dit.md) |
| 23 | b56b-7052 | 多任务 lr/loss 权重；FM↔DiT | [28](../notes/28-hand-model-tiers-fm-dit.md) |
| 24 | b66b-7f2b | GRPO 监控；CrossAttn→waypoints | [28](../notes/28-hand-model-tiers-fm-dit.md) |
| 25 | b73a-75bf | Swift LoRA &lt;7B | [29](../notes/29-hand-swift-megatron-train.md) |
| 26 | b7ef-77c8 | Megatron 并行 + 训评流水线 | [29](../notes/29-hand-swift-megatron-train.md) |
| 27 | b8af-781e | 四路传感器 → FM×25 + Diffusion | [30](../notes/30-hand-fm-full-stack-architecture.md) |
| 28 | b978-77d7 | ViT/4D→LLM 感知规划总览 | [30](../notes/30-hand-fm-full-stack-architecture.md) |

## 转录文件

- [`_raw_batch_a.md`](_raw_batch_a.md) — Page 01–07  
- [`_raw_batch_b.md`](_raw_batch_b.md) — Page 08–14  
- [`_raw_batch_c.md`](_raw_batch_c.md) — Page 15–21  
- [`_raw_batch_d.md`](_raw_batch_d.md) — Page 22–28  

## 常见缩写纠错备忘

| 手写/笔误 | 纠正 / 释义 |
|-----------|-------------|
| TraficcNN | TrafficNN |
| boundres | boundaries |
| answrr | answer |
| nums_works | num_workers |
| hiffen | hidden |
| valvo | Volvo?（存疑，未升格） |
| MOT（语境 A） | Mixture of Transformers / Tokens（离散–连续） |
| MOT（语境 B） | Multi-Object Tracking（检测跟踪）——**按上下文区分** |
| CoC vs COC | 因果推理链 vs 通信/计算优化——**按上下文区分** |
| TBT | Turn-by-Turn 导航 |
| GOD | General Obstacle Detection（内部口径） |
| FM | Foundation Model |
| WM | World Model |
