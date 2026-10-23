# 第 19 页：算法技能树：OPD/MOPD 与单机配置

- **原图**：[`images/01a169b3-23fd-78a3-97e1-95fd8e5b1ee6.jpg`](images/01a169b3-23fd-78a3-97e1-95fd8e5b1ee6.jpg)
- **主题标签**：蒸馏, KL, 配置

## 纠错说明

- num_works → num_workers
- 单词表 KL = 单 token/局部词表 KL

## 校正转录

OPD / MOPD；gt one-hot vs teacher soft；student rollout；SFT+RL→大 teacher
KL：dense reward、无 reward hacking；off-policy 单轨迹 vs LLM 多轨迹
单机：bf16；workers=8；不解压；有效 bs 公式；10 epochs；3k ckpt
special token：unused/extra/ignore/unk/answer；long_cat/short_cat


## 摘要

蒸馏+KL 奖励经验与可复现的单机训练配置清单。
