# 第 24 页：RL 算法监控 + AZ Cross-Attn + DiT waypoints

- **原图**：[`images/01a169b3-27ba-7409-b927-57d3f6299959.jpg`](images/01a169b3-27ba-7409-b927-57d3f6299959.jpg)
- **主题标签**：GRPO, Cross-Attn, Diffusion Policy

## 纠错说明

- 无特殊 tokens 放入（按笔迹）
- DP = Diffusion Policy
- (5,2) waypoints

## 校正转录

GRPO/PPO/DPO：监控 KL、温度、lr、hacking、collapse
Fm-feat B,N,768 → MLP→RMSNorm(2048)；hidden[0,9,18]→MLP→RMSNorm(1536)
AZ×3 Cross-Attn：Q=lm token，K/V=FM；→ lm-head
Condition(lat+token/point/time/ego)→ DiT×2 → (5,2) waypoints


## 摘要

RL 调试清单与 FM 特征交叉注意力接扩散航点头的结构。
