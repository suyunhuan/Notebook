# 第 07 页：Teacher/Self Forcing、KV Cache、FM→DiT

- **原图**：[`images/01a169b3-1b78-7491-989c-14bdeb6afd91.jpg`](images/01a169b3-1b78-7491-989c-14bdeb6afd91.jpg)
- **主题标签**：训练, KV Cache, DiT

## 纠错说明

- kvache → KV cache
- Meta Action；SRPP 为下游模块名

## 校正转录

- Teacher forcing vs Self forcing（分布差；10% self 不稳定）
- Meta action as segmentation；对网络影响小；Freq / Gradient L2
- FM：L0 hidden 高 attention；取层 0/9/18
- **KV cache 28→3**（降访存）
- 结构：VLM/ViT/FM → SRPP；FM → DiT


## 摘要

对比 forcing 策略，压缩 KV，并从中间层取特征接到规划/扩散头。
