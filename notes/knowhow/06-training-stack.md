# Know-how：SWIFT / Megatron 训练栈与复现配置

## 选型规则

```
参数量 < 7B  ──► SWIFT + LoRA/FSDP（快迭代）
参数量 ≥ 7B  ──► SWIFT + Megatron 5D 并行（可扩展到 ~100B）
需要高吞吐推理评测 ──► 配 vLLM
```

不要对小模型一上来全开 TP/PP——通信开销可能超过收益。

---

## LoRA 速查（&lt;7B）

| 项 | 笔记值 | 说明 |
|----|--------|------|
| rank | 8 | 越大越接近全参 |
| alpha | 32 | 常设为 2×rank |
| target | q+v → qkvo → +gate/up/down | 逐步加 |
| 精度 | bf16 优先 | 数值稳 |
| 并行 | DDP / FSDP | 手写 DPP 应为 DDP |

多模态：限制 `max pixel` / `max img tokens` / video 帧数，防止视觉 token 撑爆。

---

## Megatron 旋钮

1. **micro batch** vs **global batch**：梯度累积对齐全局。  
2. **TP**：单卡装不下某一层。  
3. **PP**：层数极深。  
4. **CP**：序列极长（8k+）。  
5. **EP**：MoE 专家。  
6. **sequence parallel + packing + FlashAttention + recompute**：吞吐与显存。  
7. **lazy tokenize**：降低启动与 CPU 预处理阻塞。  

---

## 一份可抄的小配置（页 19/26 综合）

```text
precision: bf16
lr: 2e-6          # 视阶段调整；蒸馏单机可能不同
max_length: 8192
num_workers: 8
save_every: 3k steps
epochs: 10
image_tokens: ~1150
video_tokens: ~128
max_frames: 16
gradient_checkpoint: true
effective_batch: bs_per_gpu * grad_accum * num_gpus
```

Special tokens：预留 unused / ignore / unk / answer；区分 long_cat / short_cat 数据格式。

---

## 损失与监控

多头时显式权重（页 23）：

- 主任务（AZ/AR）≈ 1  
- 语言/对齐（llm, FM2LLM）≈ 0.1  
- 辅助（PP）≈ 0.05  
- 盯 `grad loss` 是否爆炸或长期平台  

RL 阶段另盯：KL、温度、长度分布、hacking、collapse。

---

## ModuleWrapper 最小职责

| 方法 | 职责 |
|------|------|
| networks | 构建/包装模型 |
| sample | 数据采样与打包 |
| forward / net output | 前向 |
| update | 反传与优化器步进 |

流水线：`env → load → dataset → train → save/eval`。

---

## 复现检查清单

- [ ] 随机种子与数据版本钉死  
- [ ] 有效 batch 与笔记公式一致  
- [ ] 视觉 token 上限与分辨率匹配  
- [ ] ckpt 频率足够回滚 Toggle/GRPO 实验  
- [ ] 日志含各 loss 分量而非只看 total  
