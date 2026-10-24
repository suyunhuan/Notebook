# 大模型训练效率优化

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | 大模型训练效率优化方法 |
| Thread ID | `x2oYTj7lKbhtwQJaA` |
| 源链接 | https://www.doubao.com/thread/x2oYTj7lKbhtwQJaA |
| 所属分类 | 大模型技术 / 系统与训练工程 |
| 源文件 | `../sources/x2oYTj7lKbhtwQJaA.md` |

## 一句话摘要

训练效率优化分三路：注意力算子省访存（SDPA/FlashAttention）、分布式消冗余显存（ZeRO/FSDP）、以及 AMP/检查点/并行/数据管道等通用手段；工业常用组合是 BF16 + FlashAttn + ZeRO-2 + Gradient Checkpointing。

## 核心关键词

SDPA；FlashAttention；ZeRO-1/2/3；FSDP；AMP / BF16 / FP8；Gradient Checkpointing；梯度累积；TP / PP；算子融合

## 知识脉络

### 1. 注意力算子：SDPA vs FlashAttention

- **瓶颈**：标准注意力物化 \(O(N^2)\) 矩阵，HBM 带宽受限。
- **SDPA**：PyTorch `scaled_dot_product_attention` 接口，自动选 Flash / Memory-Efficient / math 后端；支持 causal、dropout。
- **FlashAttention**：分块 + SRAM 计算 + 重计算 softmax；显存约 \(O(N)\)；v3 支持 FP8、更长上下文。
- **关系**：SDPA = API；FlashAttention = 常用高性能后端（非并列替代品）。
- **相关**：Memory-Efficient Attention；推理侧 PagedAttention（训练少用）；长序列 Ring / Blockwise Attention。

### 2. 分布式显存：ZeRO-2 重点

| Stage | 分片对象 | 特点 |
|-------|----------|------|
| ZeRO-1 | Optimizer states | 省显存有限 |
| **ZeRO-2** | Optimizer + Gradients | 参数每卡完整；通信小于 ZeRO-3，速度更好 |
| ZeRO-3 | + Parameters | 显存最优，通信最大 |

- 与 DP 互补：消除 DP 冗余，不是 TP/PP 的替代。
- **FSDP**：`SHARD_GRAD_OP` ≈ ZeRO-2；`FULL_SHARD` ≈ ZeRO-3。
- **选型粗标**：中等规模（原文称 <10B 量级）偏 ZeRO-2；更大再上 ZeRO-3 ± TP。

### 3. 其他主流手段

- **AMP**：FP16/BF16 前反向，关键累加 FP32；FP8 进一步降带宽。
- **梯度累积**：不减单步显存，抬有效 batch。
- **Gradient Checkpointing**：丢激活、反传重算；长序列刚需；Flash 内部已有轻量重算。
- **TP / PP**：模型切分，常与 ZeRO 组合。
- **优化器状态压缩**：8-bit Adam 等。
- **数据管道**：prefetch、共享内存、离线预处理，避免 GPU 等 CPU。

### 4. 智驾场景联想（由通用方法外推）

- 环视多帧 + 长 CoC / VQA 序列 → 优先 FlashAttn + checkpoint。
- VLA 多模块联训显存紧 → ZeRO-2/FSDP + 分阶段解冻（少同时激活大模块）。
- 超长 BEV 时序或世界模型视频 → 考虑序列并行类注意力，而非只堆卡。

## 知识点校验

1. **Caution**：「模型 <10B 用 ZeRO-2、几十 B 用 ZeRO-3」是经验阈值，实际取决于序列长度、激活、Adam 状态精度与并行拓扑，不能当硬阈值。
2. **Caution**：PagedAttention 主要服务推理 KV 管理；写进「训练效率」清单易误导，训练侧应弱化。
3. **Expand**：智驾多模态（图像 token ≫ 文本）下，激活显存常大于优化器状态，需单独 profiling；仅开 ZeRO-2 可能不够，要配合 checkpoint / 冻结 ViT。
4. **Correct**：SDPA 与 FlashAttention 的接口–后端关系、ZeRO 三级分片对象划分，与 PyTorch / DeepSpeed 文档一致。

## 延伸阅读建议

- FlashAttention / FlashAttention-2/3 论文
- ZeRO（Rajbhandari et al.）；PyTorch FSDP 文档
- Megatron-LM（TP/PP）；Ring Attention
- NVIDIA Transformer Engine（FP8）

## 与其他主题的关联

- [AdamW](../notes/06-adamw-optimizer.md) — 优化器显存与 8-bit 变体
- [VLM 训练阶段](../notes/01-vlm-training-stages.md) — 多阶段解冻降低同时训练参数量
- [Alpamayo Reasoning-VQA](../notes/09-alpamayo-reasoning-vqa.md) — 多相机 token 膨胀与压缩
- [BEV+PNC](../notes/02-bev-pnc-stack.md) — 长时序 / Transformer 规划侧算力
