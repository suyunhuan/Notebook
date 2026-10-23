# 第 26 页：训练流水线 ModuleWrapper

- **原图**：[`images/01a169b3-2931-7cff-8938-41aabf8ba5d7.jpg`](images/01a169b3-2931-7cff-8938-41aabf8ba5d7.jpg)
- **主题标签**：训练流水线, DiT loss

## 纠错说明

- uvp → vllm
- DFT loss → DiT loss（语境更吻合）

## 校正转录

swift / megatron / vllm；lr=2e-6；max len 8192；bf16；image token 1150；video 128；max frames 16；grad ckpt
ModuleWrapper：networks / sample / forward / update
env→load→dataset→train(forward/loss/backward/update)→save&eval；DiT loss


## 摘要

多模态训练超参与标准训练环封装。
