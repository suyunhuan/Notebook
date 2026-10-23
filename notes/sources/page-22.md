# 第 22 页：Max / Ultra / Ultimate 规格表

- **原图**：[`images/01a169b3-2647-7460-befe-1b17bc74470c.jpg`](images/01a169b3-2647-7460-befe-1b17bc74470c.jpg)
- **主题标签**：型号规格, FM, DiT, MLLM

## 纠错说明

- COC → 多处应为 CoT
- PV Hz 喂入 FM→DiT

## 校正转录

Max：共享 ResNet18，无 BEV project；Ultra：更深/FastViT
Vision/LiDAR BEV/Nav → FM → DiT
规格例：Max 768×18≈0.13B+DiT0.05B；Ultra 1024×24≈0.35B；FM/MLLM 2048×28≈1.8B
Ultimate：Text/instruct → FM→MLLM→DiT；MOT 融合 VL/FM/WM；Causal Mask


## 摘要

产品档位参数与 FM→MLLM→DiT 的统一接口草图。
