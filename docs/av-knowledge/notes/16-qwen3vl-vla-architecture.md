# 基于 Qwen3-VL-2B 的智驾 VLA 架构（感知、Codebook、MoT、Action Expert）

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | Qwen3-VL-2B VLA架构升级(含Codebook与MoE) |
| Thread ID | `xuayeFKhBhI8yo8yX` |
| 源链接 | https://www.doubao.com/thread/xuayeFKhBhI8yo8yX |
| 所属分类 | 大模型技术 / 智驾 VLA 系统设计 |

## 一句话摘要

以 Qwen3-VL-2B 为理解底座，插件式接入 4D 感知、VQ Codebook、（可选）MoT 路由与 Flow Matching Action Expert，并建议分阶段训练而非一次端到端打通。

## 核心关键词

`Qwen3-VL` · `VLA` · `4D 感知` · `KV conditioning` · `Flow Matching` · `VQ-VAE Codebook` · `MoT` · `State Management` · `ms-swift`

## 知识脉络

### 1. 目标形态（插件式，非全量重训）

```
Camera/LiDAR/(VGGT)
  → 4D 感知特征 F_percep
  → [可选 VQ] 离散感知 code tokens
  → [可选 MoT] 与 ViT/几何 token 路由融合
  → Qwen3-VL LLM（LoRA）场景理解
  → Hook 中间层 KV → 压缩为 KV_cond
  → Flow Matching Action Expert → 轨迹/控制
  → Planner（QP 硬约束或可学习软约束）
```

Swift：负责 Qwen 加载、LoRA、数据、分布式；感知/VQ/MoT/Action 为外部 `nn.Module`，wrapper 内聚合多 loss。

### 2. 模块要点

**4D 感知**  
- Camera：BEVFormer / BEVDet4D 等；LiDAR：PointPillars/CenterPoint；VGGT：冻结几何特征。  
- 策略：感知先单独训到可用再冻结，VLA 阶段只训投影或小 LR 联调。

**VLM**  
- 图像 + 驾驶文本（caption/VQA/指令）；ViT 冻结或轻量 LoRA；LLM QLoRA。  
- KV：取倒数第 2–4 层一类语义层；**必须降维/池化** 成固定条件，防显存爆炸与变长问题。

**Action Expert**  
- Flow Matching 学速度场；条件 = KV_cond ± 感知全局特征。  
- 与回归单一轨迹相比，分布建模更适合多模态驾驶意图（需数据与延迟验证）。

**Planner**  
- MVP 优先：模型轨迹为初值 + QP 硬安全约束。  
- 可学习 Planner 增加碰撞/平滑 soft loss。

### 3. Codebook 与 MoT 升级

| 组件 | 作用 | 风险 |
|------|------|------|
| **VQ-VAE on BEV** | 连续感知 → 离散 code，贴近 LLM token 空间；可单独预训练 | **code collapse**；序列过长 |
| **MoT**（Transformer 专家，异于 FFN-MoE） | 图像 / 感知 code / 几何分专家，减轻 concat 抢注意力 | **路由坍缩**；需负载均衡 loss |
| 接入 | code embedding→MoT；慎直接扩 Qwen 词表 | 词表膨胀伤推理 |

推荐：Codebook K≈1024/2048；MVP 用 2 专家、MoT 放在原生 ViT 之后改动最小。

### 4. 4D 主干 + State Management（综合定稿方向）

原文后期收敛的设计取向（方案级，非已验证 SOTA）：

1. **4D 时空编码器做视觉主干**，替换/弱化原生 2D ViT 主导地位（环视视频动态）。  
2. **2D ViT 降为冻结语义补强**（灯色、标识纹理）。  
3. 融合顺序：4D 时空收敛 → 2D 语义交叉注意力注入 → **4D-VQ** 得时空 CodeToken。  
4. **State Management**：跨帧维护占用/关键目标/导航状态，约束时序一致性，缓解纯帧级抖动。

### 5. 分阶段训练（防崩）

| Stage | 内容 |
|-------|------|
| 0 | 预训练 VQ-VAE（冻结 codebook） |
| 1 | 4D 感知监督至收敛并冻结 |
| 2 | MoT 路由预热（负载均衡） |
| 3 | Qwen 驾驶域 LoRA（先理解，不接 Action） |
| 4 | 冻结底座，只训 KV 投影 + Flow Action |
| 5 | 可选小范围联合微调（高风险） |

数据：图像/点云/BEV/轨迹/文本必须对齐；nuScenes 等缺自然语言时需自动 caption。

### 6. MVP 路径

纯视觉 VGGT 或轻量 BEV → 文本摘要进 Qwen → KV→Flow 出短时域轨迹 → 仿真闭环；再加 VQ/MoT/Planner。

## 知识点校验

- **Correct**：分阶段、KV 条件需压缩、VQ/MoT 各自坍缩风险、Planner 硬约束兜底——工程逻辑合理。
- **Caution**：整套「Qwen3-VL-2B + MoT + 4D-VQ + State Management + Flow」是**对话中的架构设想**，非已发表统一基准；性能宣称需自建实验，勿当作既有论文结论。
- **Caution**：原文 Flow Matching 损失公式排版混乱处，应以标准 conditional flow matching 文献为准，勿照抄变形公式。
- **Expand**：车载实时性下 Flow 采样步数与蒸馏；State 与占用网络/记忆库的具体接口；感知 code 与轨迹 code 是否共用词表空间。

## 延伸阅读建议

- Qwen2/Qwen3-VL 技术报告与 ms-swift 多模态文档  
- Flow Matching；Diffusion Policy / 具身 VLA Action Expert  
- VQ-VAE；Mixture-of-Experts / Mixture-of-Transformers 相关工作  
- BEVFormer、VGGT；端到端 VLA 开源项目（对照其真实模块边界）

## 与其他主题的关联

- Action Expert DiT/adaLN：[`../notes/10-action-expert-dit.md`](../notes/10-action-expert-dit.md)
- 轨迹 Codebook：[`../notes/11-trajectory-codebook.md`](../notes/11-trajectory-codebook.md)
- BEV/4D 感知：[`../notes/13-bev-depth-models.md`](../notes/13-bev-depth-models.md)
- Swift 训练推理栈：[`../notes/15-llm-train-infer-stack.md`](../notes/15-llm-train-infer-stack.md)
- VLM 训练阶段：[`../notes/01-vlm-training-stages.md`](../notes/01-vlm-training-stages.md)
- 持续学习抗遗忘：[`../notes/17-continual-learning-vla.md`](../notes/17-continual-learning-vla.md)
