# 第 17 页：DAgger RL 与多模 4D 阶段

- **原图**：[`images/01a169b3-2298-7039-81b0-a05efd60ac97.jpg`](images/01a169b3-2298-7039-81b0-a05efd60ac97.jpg)
- **主题标签**：DAgger, depth, 多模4D

## 纠错说明

- Dagger → DAgger
- inthustic → intrinsic
- PPT/DPT Head：稠密预测头

## 校正转录

Cam→PE→Frame Attn→Temp Attn→token compress(ST-PE)→Spatial Attn；注入内外参
改进：DA3 补 PV Seg；内外参矫正；metric depth；stream 0.3s→1s；ray map；鱼眼
多模4D：Cam Patch Transformer ⊕ Lidar Feat → PV Seg
S0 预训练 → S1 dense depth → S2 BGC/RGC → S3 E2E


## 摘要

带几何的时空注意力管线 + DAgger，按预训练到端到端分阶段推进。
