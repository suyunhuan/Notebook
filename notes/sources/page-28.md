# 第 28 页：感知接 LLM 的策略总览

- **原图**：[`images/01a169b3-2aab-7688-9439-483ff1750215.jpg`](images/01a169b3-2aab-7688-9439-483ff1750215.jpg)
- **主题标签**：策略, Scaling, CoT 数据

## 纠错说明

- MOPD 内部方法名
- DPT head；LSS→BEV

## 校正转录

ViT 环视；4D 训基模；7B@3Hz / 3B@5–8Hz；流式视频+空间理解
Depth/视/视频 → ViT/4D → LLM
感知：token 压缩；MOT；辅助 depth/BEV/重建/生成
数据：原子能力；CoT；关键帧；high overlap low difference
训练：Scaling、混训、model merge、MOPD、RL
落地：鱼眼 PV tokenize；多层→DPT；LSS→BEV→BCG


## 摘要

把性能目标、数据与训练策略收束到「4D ViT→LLM」主线上。
