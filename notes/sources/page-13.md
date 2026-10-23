# 第 13 页：流式多视角 4D 架构

- **原图**：[`images/01a169b3-1fc7-7909-b0ea-f377c8e22674.jpg`](images/01a169b3-1fc7-7909-b0ea-f377c8e22674.jpg)
- **主题标签**：流式, 4D, SSM, KV Cache

## 纠错说明

- LiT：Language-Image Tuning / Locked-image Tuning 类双视角适配
- P/B/G.：感知/行为/引导类任务头，待项目确认

## 校正转录

- 文本 → LiT（双视角）→ 2D adaptor；多视角 4D embedding → 4D adaptor → add&cat + Cross Attn
- m-cam → Image Patch → 时间 Attn → Attn
- 核心：Router / Attn / SSM ↔ State；KV Cache；流式 6s、2s、2Hz
- 头：Ray、Depth、Scene Flow、Camera、Scale、P/B/G.、planning


## 摘要

流式多相机+文本经 2D/4D adaptor 与 SSM/Attn 状态机，输出几何与规划多头。
