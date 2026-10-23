# Video Thinking、Latent Reasoning 与 World Model Rollout

## 元信息

| 字段 | 内容 |
|------|------|
| 来源 | 手写笔记 Page 15–16（`handwriting/_raw_batch_c.md`） |
| 资产 ID | `aeb1-783e` · `af72-7ba9` |
| 所属分类 | 智能驾驶核心技术 / 世界模型与潜空间推理 |
| 关联原始页 | Batch C：Video Thinking；4D Encoder → MoT → Latent Reasoning |

## 一句话摘要

历史视频 latent / state / action 经视频生成（如 WAN2.2）与 Action Gen 交互预测未来帧与动作，并用 AR Latent World Model 做短时域（约 2s）rollout，再解码为可执行 action。

## 核心关键词

`Video Thinking` · `WAN2.2` · `Action Gen` · `Latent Reasoning` · `AR LWM` · `World Model rollout` · `4D Encoder` · `Ray Map` · `SD+TBT` · `MoT` · `Vehicle Kinematics`

## 知识脉络

### 1. Video Thinking：历史→未来帧+动作

**任务定义**：给定过去 \(N\) 帧（及可选 caption），预测未来 \(M\) 帧视觉内容与对应 action。

```
hist Video latent
hist state
hist action
        ↓
   [encoder / latent block]
        ├──→ Video Gen（如 WAN2.2） ┐
        │         ↕ 交互            ├──→ 未来帧 + action 预测
        └──→ Action Gen             ┘
```

- **Video Gen**：在 latent 空间生成未来帧；笔记点名 **WAN2.2**（开源视频生成模型 Wan 系），与 Action Gen **双向交互**（视觉一致性约束动作；动作约束画面运动）。
- **Action Gen**：与视频分支共享历史条件，输出可执行控制/轨迹 token。
- **场景条件三路**：① MoT（Mixture of Transformers / 感知 MoT 特征）进 text Tower；② 显式场景 prompt；③ 开源 LLM forward 的文本 hidden。三者把「看懂场景」从纯像素接到语言语义。

工程要点：Video Thinking 不是单独的「看视频写 caption」，而是 **latent 级世界状态 + 动作闭环**；caption / prompt 只作条件，不替代几何与动力学。

### 2. 感知侧前置：4D Encoder → MoT 多头

在进入 WM / Action 之前，手写稿给出感知压缩管线：

1. **4D Encoder** → `{ Depth, Ray Map, Semantic }`  
   - **Ray Map**：每像素（或每 token）对应一条相机射线（方向 + 可选深度/占用），把 2D 观测接到 3D 几何；相对「只回归 pose」，用 ray map 监督常更稳、可微性更好（详见 note 26）。
2. **多模 4D + LiDAR** 与 **Fisheye → {Seg, Depth}** 汇入 **MoT**，再分头：  
   Depth / Ray / PV seg / BEV `{B, C, G}`（常对应车道/认知/可行驶等 BGC 族头）/ Sparse BEV。
3. **导航与运动学旁路**：  
   `SD + TBT` 与 `Vehicle Kinematics` 进共享 backbone，产出 BEV/PV/Sparse BEV，并旁路到 Action。  
   - **TBT（Turn-by-Turn）**：逐步导航指令（下一路口左转、保持车道等），与 SD map 一起提供「该往哪走」的语义/拓扑，而非仅视觉。

### 3. Latent Reasoning：短时域 AR World Model

```
Large Transformer → World state Reasoning
                      ├ t+0.5
                      ├ t+1.0
                      ├ t+1.5
                      └ t+2.0/2.5  → Action Decoder
```

- **Rollout 规格（笔记）**：约 **4 步、2s 未来**；可用 GOD 等 BEV/占用真值监督 world state。
- **AR LWM（Autoregressive Latent World Model）**：在潜空间自回归推进「世界状态」token，再接 **action decoder**；避免每步都解码全分辨率视频。
- **与 Video Thinking 关系**：Video Thinking 侧重「看得见的未来帧 + 动作」；Latent Reasoning 侧重「短 horizon 状态机式推理」，算力更可控，适合车载闭环。

### 4. 工程 Know-how

| 项 | 建议 |
|----|------|
| 条件注入 | MoT / prompt / LLM hidden 分层注入，避免全部 concat 冲掉几何 |
| 运动学旁路 | 车辆运动学直接约束 Action，减轻 WM 幻觉导致的不可达控制 |
| Horizon | 2s 短 rollout 优先保一致性；长 horizon 再叠稀疏 BEV / Occ Flow（note 26） |
| 训推一致 | 历史 latent 用教师强迫或有限 teacher-forcing + 自回归混合，防曝光偏差 |
| 帧率 | 感知可 10Hz 入，规划侧常抽到 2Hz；与 FM/DiT 档位设计对齐（note 28） |

## 知识点校验

1. **Correct**：历史 latent + 视频生成与动作生成交互、再用 AR 潜空间短时域 rollout 接 action decoder，是「世界模型辅助规划」的常见拆分（像素/视频级 vs 潜状态级）。
2. **Caution**：笔记中末步写作 `t+2.5`，相对 0.5s 步长可能笔误（更合理为 `t+2.0`）；落地以实际采样网格为准，勿照抄时间戳。
3. **Caution**：WAN2.2 / 开源 LLM 作条件时，开源权重许可、延迟与车规部署需单独评估；「能生成视频」≠「可在线闭环」。
4. **Expand**：Video Gen↔Action Gen 的交互是共享 attention、互为条件、还是交替优化，原稿未闭合；需实验对照。
5. **Expand**：GOD gt 监督 LWM 时，占用/可行驶域与轨迹一致性如何加权，与 Occ Flow、Sparse BEV 长时序（note 26）应对齐设计。

## 与其他主题的关联

- [02 BEV+PNC 栈](02-bev-pnc-stack.md) — 感知多头与 PNC 外壳
- [05 因果建模](05-causal-modeling-trajectory.md) — 状态→动作因果链
- [08 VLA 歧义与 CoC](08-vla-ambiguity-coc.md) — Meta-action / 关键帧与推理一致性
- [09 Alpamayo Reasoning](09-alpamayo-reasoning-vqa.md) — 场景理解与 VQA 条件
- [10 Action Expert DiT](10-action-expert-dit.md) — 动作生成另一主线（扩散）
- [13 BEV 深度](13-bev-depth-models.md) — Depth / 几何前置
- [16 Qwen3-VL VLA](16-qwen3vl-vla-architecture.md) — 多模态条件进 VLA
- 可能的 **18+**（手写前序）：若存在 Video/WM 或 4D 预训练笔记，应交叉链到本篇的 encoder 与 rollout 规格
- 本系列后续：[26 Token/Occ](26-hand-token-arch-occ-sparse.md) · [28 档位与 DiT](28-hand-model-tiers-fm-dit.md) · [30 全栈](30-hand-fm-full-stack-architecture.md)
