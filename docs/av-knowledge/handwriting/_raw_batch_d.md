# Handwriting Raw Batch D

> 来源：7 张手写笔记照片（附件顺序 = Page 22–28）。转录已做笔误/缩写还原；模糊处标 `[?]`。

## Page 22 — b483-74fe

**主题**: Max / Ultra / Ultimate 档位骨干与参数规模，以及 Vision/Lidar/Navi → FM → DiT / MLLM 融合规格。

**转录（纠错后）**:

（横版双页：左 Max/Ultra/Ultimate 档位，右 Ultimate-MLLM 规格）

**左页 — 档位与骨干**
- **Ultra**：`ResNet-18 → ResNet-101[?]` ↑深度 / ↑通道
- **FastViT**：混合 CNN + ViT；参数 `30M × 7`
- **Max**：共享 BG / 采用 BG[?] 共享 `ResNet-18`；参数 `9M`；无 BEV project（无 BEV 投影）

**左页 — 输入 → FM → DiT**
- ① Vision（PV）→ `10Hz` → `2Hz` → **FM** → **DiT**（旁注：优化位置编码）
- ② Lidar BEV `0.3s` → FM → DiT
- ③ Navi（SD / Navi）→ FM → DiT

**左页 — 规模表**（channel / layer / 参数 | DiT）
- Noisy[?]：`512 · 3 · 8M`｜`512 · 3 · 8M`
- **Max**：`768`｜`18`｜`0.13B`｜DiT `768 · 3`｜`0.05B`
- **Ultra**：`1024`｜`24`｜`0.35B`｜DiT `1024 · 3`｜`0.1B`
- （备选）`1536`｜`20`｜`~0.7B`

**左页 — FastViT / Ultimate**
- FastViT：token 压缩；参数 ↑
- FastViT `30M × 7`；`1920×1080`；vision pretrain 基模加载 / vision 蒸馏加速[?]
- **Ultimate**：
  - Text · `cam × 2` · instruct → **[FM → MLLM]** → DiT
  - ② `20Hz` · `2Hz` → [FM → MLLM]
  - ①③ 汇入 FM

**右页 — MLLM / WM**
- （圈）`PV · BEV · Action → MOT`；`WM 推理[?]/OCR`；`FM 2048 · 28 · 1.8B`
- DiT：`2048 · 3 · 0.2B`（原文 `1024` 划掉改 `2048`）
- **MLLM** / VLM：`2048 · 28 · 1.8B`（旁注 VL-[?]B）
  - `40B token` SFT 基模
  - 辅助驾驶 · FM 能力
  - 理解 · 规划 · **COC / CoT 推理** → action
- MOT 融合 VL · FM · WM
- Causal Mask（原文 Casual Mask）· 推理效率[?]提高

**不确定项**:
- ResNet 目标深度：`101` 与 `114` 两种读法；取标准 ResNet-101 并标疑。
- Max「共享/采用 BG」：BG 全称未写（疑 Backbone / 与 BCG 相关）。
- Vision 支路中间方框字迹不清；帧率取 `10Hz→2Hz`（亦见 `12Hz` 误读可能）。
- `WM 推理 / OCR`：亦可能读作「推广 / OCR」。
- VLM 前缀（VL-2B / VL-UB 等）笔迹不清。
- 页缘残字 `…etrain` / `…域` 及 DiT 旁 `…8 3 0.05B` / `…4 3 0.1B` 可能为左页表延续透写。


## Page 23 — b56b-7052

**主题**: 多任务 loss / lr 权重配置，以及 FM↔DiT 与帧率（10Hz→5Hz）示意。

**转录（纠错后）**:

```
lr
llm      0.1     loss 0.1
FM2LLM   0.1     0
AZ       1       { AR 1
                   PP 0.05 }

total 5
uvp
grad loss 1.6
```

**图示**:
- 左侧：曲线分布（类扩散/噪声日程）
- 中：`DiT` ↔ 竖框 `FM`（双向箭头）
- 右：（圈）`(10Hz) → (5Hz)`；下方模糊 `250ms[?]`

**不确定项**:
- `AZ`：与后页 `AZx3 Cross Atten` 同族模块/损失项；全称未写。
- `PP 0.05`：规划点 / Path Planning loss 权重，或其它缩写；与后页 `PP (5,2) waypoints` 可能同源。
- `uvp`：与资料中 CaDDN/UVP 多视角→BEV 损失相关；本页仅作权重项名。
- `FM2LLM` 右侧 `0`：对应 loss 列置零还是另一分支权重，未标注全称。
- 图右 `250ms` 笔迹极淡。


## Page 24 — b66b-7f2b

**主题**: GRPO/PPO/DPO 训练监控清单，与 Fm_feat → MLP/RMSNorm → AZx3 Cross-Attn + DiT 出 waypoints。

**转录（纠错后）**:

**RL 监控**
- 标题：`GRPO · PPO · DPO`
- 清单：配置；常见问题；监控项；调试
- 监控项括号：`KL`；`温度[?]/难度`；`lr`；`hacking`（reward hacking）；`collapse`

**特征支路**
- （圈）`Fm_feat`：`B, N, 768`
  - → MLP `(768 → 1536 → 2048)` → RMSNorm `(2048)`
  - 去/走[?]特殊 tokens 嵌入/输入
- `hidden [0, 9, 18]`
  - → MLP `(2048 → 1536 → 1536)` → RMSNorm `(1536)`

**融合 / 出轨迹**
- **AZx3 Cross Atten**
  - K, V：（圈）`(3, B, seq, 1536)` ← 上两路特征（3 层 hidden）
  - Q：`lm token Q` · `512 × 1536`
  - 出：`lm-head` / m-head `512 × 1536`
- **Condition** ↑：`lat + vision` · `point` · `time` · `ego`
  - → **DiT ×2** → **PP** `(5, 2) waypoints`

**不确定项**:
- 监控项第二字：`温度` 与 `难度` 两读并存。
- 「去特殊 tokens」vs「走特殊 token 输入」：笔迹近，按数据流两解皆可。
- Condition 首项：`lat+vision` 亦可能读作 `1st+sign.[?]`。
- `PP`：Path/Planning Points；与 `(5,2) waypoints` 对齐。


## Page 25 — b73a-75bf

**主题**: Swift <7B LoRA/多模态上限与 >7B Megatron 并行（TP/PP/DP/EP/CP）及通信优化。

**转录（纠错后）**:

**上：Swift Transformer <7B**
- LoRA：`rank=8`，`alpha=32`；`~1% param train`
- target modules：`all-linear`；`q+v`；`qkv o`；`qkv o gate up down`
- `aligner` · `merger`
- `max pixel: 16777216`
- `max img tokens: 16384`
- `video 50[?]/500`
- （圈）`DDP/FSDP`（原文 DPP → 纠为 DDP）
- 精度：`fp32 / bf16 / fp16`
- `channel loss`；损失 `CE loss`（原文或作「激活 CZ loss」→ CE）

**下：Swift Megatron 大模型 >7B–100B**
- 并行：`TP / PP / DP / EP / CP`
- `micro batch size` / `global batch size`
- `tensor model parallel`
- `pipeline model parallel`
- `sequence parallel`
- `packing`
- `flash attention`
- `recompute`
- `COC` 通信优化（Communication-Overlap-Computation[?]）
- `lazy tokenize`：延迟 tokenize ↑速度
- 精度旁注：`bf16 / fp16`

**不确定项**:
- `video 50` vs `500`：位数不清。
- 「激活 CZ loss」优先还原为 **CE loss**；若确为 CZ 则保留专名。
- 本页 `COC` 偏通信优化缩写，与智驾「COC reasoning」同形异义，勿混。


## Page 26 — b7ef-77c8

**主题**: Swift+Megatron 训练超参、ModuleWrapper 职责与 env→DataLoader→训练→评估流水线。

**转录（纠错后）**:

**超参**
- `swift · megatron · amp[?]`（Automatic Mixed Precision；亦读 uxp）
- `lr = 2e-6`
- `max length 8192`
- `bfloat16`
- `image token 1150`
- `video token 128`
- `max frames 16`
- `gradient clip` 梯度裁剪 / `gradient ckpt` 梯度检查点[?]

**ModuleWrapper**
- `networks`
- `sample` 数据采样
- `net output` 前向
- `update` 更新权重

**流程**
```
env 初始化 → 加载模型 → Dataset · preprocess · DataLoader
  → 训练 → 保存和评估
```
- 训练内：`forward · loss · backward · param update`
- `DiT loss`

**不确定项**:
- 首行第三词：`amp`（混合精度）更贴训练语境；`uxp` 为备选读法。
- `gradient clip` vs `gradient ckpt`：页上/下两处可能分写裁剪与 checkpoint。
- `video token 128`：位数略挤，亦可能 `188`。


## Page 27 — b8af-781e

**主题**: 环视/前视/Lidar·Radar/Navi 编码进 ~1000 token，0.4B FM（×25）与 75M Diffusion 轨迹头。

**转录（纠错后）**:

**① 环 + 后**
- 输入：`5×(3×352×640)` · `4×(3×256×384)`
- → **ConvNet (10–25M)** + 3D PE
- 中间：`384×22×40` · `384×16×24`
- `10/2 Hz` 时/空 → Fusion & Downsample
- 出：`5×1024×5×10` · `4×1024×6×4`

**② 前 2 路**
- 输入：`2×(3×704×1280)`
- → **DINOv2 (50–200M, patch 14)** + 3D PE（ConvNet → distill）
- 中间：`384×44×80`
- → Fusion & Downsample → `2×1024×11×20`

**③ Lidar / Radar**
- 各 `64×288×112` → 合 `128×288×112`
- → **UNet** → `64×288×112` → DownConv → `1024×36×14`

**④ Navi**
- Navi SD + TBT → **Navi Encoder** → Navi tokens
- distill → 高精导航大模型[?]

**Flatten / Token**
- ①②③④ flatten；约 **1000 tokens**
- 量级备忘（带 ?）：PV `(7×720×1280)?`；Lidar `(512×1024)?`；Navi `(32×1024)?`

**FM · PV2BEV · Diffusion**
- **FM `0.4B[?]`**：吃 BCG · PV & 3D
- PV2BEV：`64×288×112` → BCG
- **Diffusion `75M`**：Ego state · Noise（ego/obj）· timestamp
  - Linear & Reshape；EgoTraj + Path；Obj Traj（`25×2`）

**FM 层块**
- emb `1024` → Self-Attn（`4.19M`）→ Add&Norm → FFN（`8.39M`）→ Add&Norm
- 重复 **×25**

**不确定项**:
- FM 规模：`0.4B` / `0.3B` 两读。
- 骨干：`DINOv2 patch 14`（标准）；亦见 V3/patch 16 误读可能。
- 「高精导航大模型」原文「高线导大模型」类草写。
- `BCG`：内部任务/表征缩写（与 LSS→BEV/BCG 任务呼应），全称未写。


## Page 28 — b978-77d7

**主题**: ViT/4D→LLM 感知–规划总览：帧率档、CoT 数据、混训/MOPD/RL 与 LSS→BEV。

**转录（纠错后）**:

**感知入口**
- ViT → 环视（感知）
- 4D → 训基模
- `7B → 3Hz`；`3B → 5–8Hz`
- 流式视频；空间理解

**结构示意**
- Depth · 视觉 · 视频 →（圈）**ViT/4D** → 上接 **LLM**

**感知架构**
- 4D ViT（环 · Video）
- 接入形成：MOT 等
- 高效压缩 token
- 辅助任务：depth；BEV…；重建、生成

**Planner 架构**
- DP · ARDP…
- 显/隐 hidden · KV

**数据**
- 原子能力
- `no-cot`；`cot reasoning`；`cot 指令`
- 关键帧 · causal factor
- high overlap · low difference

**环视 / 头**
- 环视鱼眼 PV tokenization
- 不同层 → **DPT head**

**训练**
- scaling law；混训；model merge；**MOPD**；RL

**底**
- LSS → BEV〔 BCG / 任务

**不确定项**:
- `ARDP`：与既有笔记一致，全称仍未展开（Auto-Regressive DP[?]）。
- `MOPD`：按知识库还原为 Multi-teacher On-Policy Distillation。
- 「接入形成」亦可能「接入生成」。
- BCG 与 Page 27 同缩写，任务名未写全。
