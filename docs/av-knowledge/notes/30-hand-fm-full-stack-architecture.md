# 环视/前视/Lidar·Navi → FM 与 ViT/4D/LLM 全栈总览

## 元信息

| 字段 | 内容 |
|------|------|
| 来源 | 手写笔记 Page 27–28（`handwriting/_raw_batch_d.md`） |
| 资产 ID | `b8af-781e` · `b978-77d7` |
| 所属分类 | 智能驾驶核心技术 / 端到端系统架构 |
| 关联原始页 | Batch D：传感器编码→~1000 token→FM×25 + Diffusion；ViT/4D→LLM 总览 |

## 一句话摘要

环视/后视 ConvNet、前视 DINOv2、Lidar/Radar UNet、Navi（SD+TBT）四路压成约 1000 token，经 0.4B 级 FM（×25 块）与 75M Diffusion 轨迹头出 ego/obj 轨迹；上层 ViT/4D→LLM 用混训、MOPD、RL 与 CoT 数据闭环。

## 核心关键词

`环视/前视` · `DINOv2` · `UNet Lidar` · `Navi Encoder` · `TBT` · `FM` · `PV2BEV` · `Diffusion 75M` · `ViT/4D` · `MOT` · `MOPD` · `LSS→BEV/BCG` · `no-cot / cot`

## 知识脉络

### 1. 四路传感器编码（Page 27）

**① 环视 + 后视**

| 项 | 规格 |
|----|------|
| 输入 | `5×(3×352×640)` + `4×(3×256×384)` |
| 骨干 | **ConvNet 10–25M** + 3D PE |
| 中间 | `384×22×40` · `384×16×24` |
| 时/空 | 10/2 Hz Fusion & Downsample |
| 输出 | `5×1024×5×10` · `4×1024×6×4` |

**② 前视 2 路**

| 项 | 规格 |
|----|------|
| 输入 | `2×(3×704×1280)` |
| 骨干 | **DINOv2（50–200M，patch 14）** + 3D PE；可用 ConvNet→distill |
| 中间 | `384×44×80` → Fusion & Downsample |
| 输出 | `2×1024×11×20` |

**③ Lidar / Radar**

- 各 `64×288×112` → 合 `128×288×112` → **UNet** → `64×288×112` → DownConv → `1024×36×14`。

**④ Navi**

- **Navi SD + TBT** → Navi Encoder → Navi tokens；可 distill 自高精导航大模型。
- **TBT（Turn-by-Turn）**：逐步导航语义，与 SD 几何/拓扑互补。

**Flatten**：①②③④ → 约 **1000 tokens**（量级备忘：PV / Lidar / Navi 另有 `?` 估算）。

### 2. FM + PV2BEV + Diffusion 轨迹头

```
~1000 tokens → FM (~0.4B, emb 1024)
                 Self-Attn(~4.19M) → FFN(~8.39M) ×25
                 吃 BCG · PV & 3D

PV2BEV: 64×288×112 → BCG

Diffusion (~75M):
  Ego state · Noise(ego/obj) · timestamp
  → Linear & Reshape → EgoTraj + Path · Obj Traj (25×2)
```

- **FM**：中等容量融合 Transformer，重复 **×25** 层块；表征服务 BCG/PV/3D。
- **Diffusion 头**：条件含 ego、噪声（自车/他车）、时间戳；输出自车轨迹+路径与他车轨迹。
- 与 note 28 的「FM→DiT waypoints」同族：本页给出更贴近传感器分辨率的 **全栈尺寸账本**。

### 3. ViT / 4D → LLM 总览（Page 28）

**感知入口**

- ViT → 环视感知；**4D** 训基模。
- 帧率档：`7B → ~3Hz`；`3B → 5–8Hz`；流式视频 + 空间理解。
- 结构：Depth · 视觉 · 视频 → **ViT/4D** → **LLM**。
- 辅助：depth、BEV、重建/生成；高效 token 压缩；形成 **MOT** 等融合。

**Planner**

- **DP / ARDP** 等（扩散策略 / 自回归扩散规划变体）；显隐 hidden 与 KV 管理。

**数据与训练**

| 维度 | 内容 |
|------|------|
| 原子能力 | 分技能数据 |
| CoT 形态 | `no-cot` · `cot reasoning` · `cot 指令` |
| 时序标注 | 关键帧 · causal factor；high overlap · low difference |
| 训练手法 | scaling law；混训；model merge；**MOPD**；RL |
| 底托 | **LSS → BEV〔BCG〕** |

环视鱼眼 PV tokenization；不同层接 **DPT head**（与 note 26 多模 4D 一致）。

### 4. 与前序笔记的「一条链」

```
传感器编码 (本篇)
  → Tokenizer / Occ·Sparse (26) · Video/LWM (25)
  → FM 档位 Max–Ultimate (28)
  → OPD/Toggle/STL/CoT (27)
  → Swift/Megatron 训 (29)
  → Action：DiT / Diffusion / ARDP
```

## 知识点校验

1. **Correct**：前视用更强 ViT（DINOv2）、环视用轻量 ConvNet、点云 UNet、导航独立 tokenizer，再 FM 融合 + 小 Diffusion 头，是可落地的算力分配；CoT 分级数据 + MOPD/RL 与既有后训练叙事一致。
2. **Caution**：FM `0.4B/0.3B`、各张量形状与「约 1000 tokens」均为手写估算；实现时应用真实 flatten 脚本核对，避免对齐错维。
3. **Caution**：**BCG**、**ARDP** 全称未在原稿展开，引用时需映射内部任务名，勿当成公开标准缩写。
4. **Expand**：7B@3Hz vs 3B@5–8Hz 的精度–延迟帕累托未给实测；与 Ultimate 1.8B MLLM（note 28）如何同车部署需产品切档。
5. **Expand**：Diffusion 75M 的 obj traj `25×2` 与 Occ Flow 8 帧、Sparse BEV 6 帧（note 26）时间网格应对齐。

## 与其他主题的关联

- [02 BEV+PNC 栈](02-bev-pnc-stack.md) — 系统三层鸟瞰
- [13 BEV 深度](13-bev-depth-models.md) — LSS→BEV
- [14 PV Backbone](14-pv-backbone-view-transform.md) — 环视/前视骨干分工
- [16 Qwen3-VL VLA](16-qwen3vl-vla-architecture.md) — LLM+动作插件对照
- [03 OPD](03-online-policy-distillation.md) · [27 OPD/Toggle/STL](27-hand-opd-grpo-toggle-stl-cot.md) — MOPD/RL
- [08 CoC](08-vla-ambiguity-coc.md) · [09 Reasoning](09-alpamayo-reasoning-vqa.md) — cot 数据形态
- [25 LWM](25-hand-video-thinking-latent-wm.md) · [26 Token/Occ](26-hand-token-arch-occ-sparse.md) · [28 档位](28-hand-model-tiers-fm-dit.md) · [29 训练](29-hand-swift-megatron-train.md)
- [18 L3/L4 路线](18-hand-vla-l3l4-roadmap.md) — 系统目标
- [19 VL→4D](19-hand-vl-4d-training-pipeline.md) — 训练阶段如何喂满本篇架构
- [20–21 感知结构化 / 多传感器](20-hand-structured-perception.md) — 中间层与检测头细节（另见 [21](21-hand-multisensor-bev-pv.md)）
- [22–24](22-hand-ego-process-meta-action.md) — Ego、数据、流式 4D 与本篇 Planner/Memory 衔接
