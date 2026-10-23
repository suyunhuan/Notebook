# 第 25 页：Swift LoRA 与 Megatron 5D 并行

- **原图**：[`images/01a169b3-2877-7a48-a5bd-45361c646250.jpg`](images/01a169b3-2877-7a48-a5bd-45361c646250.jpg)
- **主题标签**：Swift, LoRA, Megatron

## 纠错说明

- DPP → DDP
- Swift = ModelScope SWIFT
- CP = Context Parallel

## 校正转录

<7B：LoRA rank8 α32；target qkv/o/gate/up/down；max pixel/img tokens；DDP/FSDP；bf16
>7B–100B：TP/PP/DP/EP/CP；micro/global bs；sequence parallel；packing；FA；recompute；通信优化；lazy tokenize


## 摘要

按模型规模切换 PEFT 与 Megatron 多维并行的工程备忘。
