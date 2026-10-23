# 自动驾驶 × 多模态基模 × 训练工程 — 手写笔记知识库

将 28 页手写笔记识别、纠错、分类后，整理为可检索的 Markdown 知识体系。

## 快速入口

| 文档 | 内容 |
|------|------|
| [知识体系 Canvas](https://cursor.com/canvas/store-76ea721c-6842-482e-a020-029a6137026c/ff24b74b-2ba7-4ced-a650-709b8cacb2c9) | 可交互总览（分层 / 主线 / 文档映射） |
| [知识体系总览](notes/00-knowledge-system.md) | 整体架构、能力分层、技术主线 |
| [逐页转录索引](notes/sources/INDEX.md) | 28 页原图对照 + 纠错转录 |
| [主题笔记](notes/topics/) | 按感知 / 规划 / 基模 / 训练 / 工程分类 |
| [Know-how 详解](notes/knowhow/) | 核心知识点拓展与落地建议 |

## 主题地图

```
感知层 ──► 表征层 ──► 推理层 ──► 行动层 ──► 训练/工程层
  │           │          │          │            │
  PV/BEV     FM/MLLM    CoT/WM     Action/DiT   SFT/RL/Megatron
  4D/流式    MoT/KV     Latent     Diffusion    DAgger/GRPO
  Lidar/Navi Token压缩  SceneGraph Planning     Swift/LoRA
```

## 目录结构

```
notes/
├── 00-knowledge-system.md      # 整体知识体系分析
├── sources/
│   ├── INDEX.md                # 逐页索引
│   ├── page-01 … page-28.md    # 逐页转录与纠错
│   └── images/                 # 原图
├── topics/
│   ├── 01-perception-4d.md
│   ├── 02-planning-action.md
│   ├── 03-foundation-mllm-wm.md
│   ├── 04-training-rl-sft.md
│   ├── 05-infra-agent.md
│   └── 06-embodied-vla.md
└── knowhow/
    ├── 01-streaming-4d.md
    ├── 02-token-fusion.md
    ├── 03-grpo-toggle.md
    ├── 04-diffusion-planner.md
    ├── 05-vlm-wm-interface.md
    └── 06-training-stack.md
```

## 使用建议

1. 先读 **知识体系总览**，建立全局图景。
2. 按工作需要进入对应 **主题笔记**。
3. 需要公式、原理、落地清单时查 **Know-how**。
4. 核对原文时回 **逐页转录** 与 `sources/images/`。
