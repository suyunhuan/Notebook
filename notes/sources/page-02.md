# 第 02 页：VL + 4D 分阶段训练路线

- **原图**：[`images/01a169b3-17b2-7042-b1ad-2336c7367bcd.jpg`](images/01a169b3-17b2-7042-b1ad-2336c7367bcd.jpg)
- **主题标签**：训练路线, 4D Encoder, SFT, RL

## 纠错说明

- 「离线」划掉改为「路线」
- AR = Autoregressive；与 Diffusion 并列建模范式

## 校正转录

### 三大支柱
- 网络结构 → **Transformer**
- 规模：model / data / compute
- 建模范式：**AR / Diffusion**

### VL 路线
`LLM → VL-stage1 → VL-stage2 → VL-stage2.5`

### 并行 4D 路线
1. **4D Encoder pre-train**（~100M；流式类 CLIP）
2. **4D-Stage1**：Point 4D projector；~10M 4D
3. **4D-Stage2**：Point 4D encoder & projector；~100M 4D

### 汇合
- **2D/4D stage 2.5**：2D+4D 全类混训、全参（约 150M 2D + 40M 4D）
- → **2D/4D-SFT** → **2D/4D RL**


## 摘要

标准 VL 多阶段与 4D Encoder 预训练并行，在 2.5 阶段混训后接 SFT 与 RL。
