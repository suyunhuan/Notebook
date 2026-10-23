# BEV + PNC 技术栈

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | 智驾BEV+PNC AI+LLM技术梳理 |
| Thread ID | `x2KWll9wqjafJJp2B` |
| 源链接 | https://www.doubao.com/thread/x2KWll9wqjafJJp2B |
| 所属分类 | 智能驾驶核心技术 / 感知–规划栈 |
| 源文件 | `../sources/x2KWll9wqjafJJp2B.md` |

## 一句话摘要

工业智驾栈可拆为「多模态 BEV 感知（投影/时序/多头）→ PNC 轨迹生成（真值改造 + DiT/Codebook + RL/DPO）→ LLM 语义特征注入」三层，VLA 往往是后两者的统一外壳。

## 核心关键词

BEV；Lift-Splat / CaDDN；时序融合；Feature Fusion / Gate；Occupancy；PNC；真值改造；DPO/IPO；VQ-VAE Codebook；DiT；AdaLN；RoPE；GQA/MQA

## 知识脉络

### 1. BEV 感知层

**Backbone / 投影**

- 图像→BEV：CaDDN、Lift-Splat、CV3D、UVP 等；多尺度用 FPN / FCN / SSFA。
- 前视 FV：RegNet / ResNet(+FPN)；再经 fv2bev 与多相机、LiDAR、Radar、时序融合。
- 融合经验：鱼眼重叠大常用 average；普通相机多用 concat；远距依赖时序增强。

**时序与多传感器**

- `bev_temporal`：坐标补偿（自车运动）+ 交叉注意力 / LSTM / GRU / Transformer+RNN / Add。
- `bevfpn`：视频流 + 内部记忆，追求长时序；用 teacher-forcing + replay buffer 缓解训推不一致。
- `feature_fusion_layer`：各模态先各自 LN/GN，再门控或双向 cross-attn；LiDAR 可 random drop / 局部 mask；局部 3×3 注意力控算力。

**任务头（示例）**

- 检测：Anchor 或 Center 热图；近处/静态可加大 loss。
- Depth：DDN 深度分布；车道/路沿 cognition；GOD（可行驶域 + Occupancy + height/flow/semantic）。
- 其他：roadmark、seg、sparse4d、周边 trajectory_head、track（StreamPETR 等流式 DETR）。

### 2. PNC AI 层

**真值 / 采样 Trick（工程高频）**

- 曲率限速、舒适加速度、回正轨迹（直道多车道、>30km/h，按车速与横偏调回正时间 + 碰撞检查）。
- 启停：连续 5 帧速度为 0 则清空未来轨迹；底盘延迟约 400ms；速度阈值 ~0.05m/s；有效起步位移 ≥0.6m。

**优化范式**

- 监督：多目标 loss + 动态权重。
- RL reward：舒适性（acc/jerk 时间衰减）、动静态碰撞与风险、效率、跟车、goal/GT、车道/导航、红绿灯等。
- 偏好对齐：DPO / IPO；负样本可为 s 方向缩放、碰撞、换道、横向风险轨迹。
- 在线：PPO / GRPO；ego 扰动增强。
- 离散化：VQ-VAE 轨迹 Codebook，便于 LLM/VLA token 化。

**主模型形态（原文）**

- Decoder + RoPE +（GQA/MQA）→ 取末段 token hidden → Diffusion / Flow。
- DiT：ego + time embedding 经 AdaLN 条件注入；噪声轨迹作 QKV。

### 3. LLM 层

- 数据与结构服务于语义/导航/场景理解；MOT 可作为特征注入。
- 典型数据流：多模态 BEV → 检测/占用/车道/他车轨迹 → DiT(+Codebook) 自车轨迹 → LLM 高层语义条件。

## 知识点校验

1. **Caution**：原文模块名（`pdpn_nets`、`sdp_llm`、`UVP`、`RGS` loss 等）偏内部代号，公开文献中未必有一一对应实现，引用时需对照真实代码/论文。
2. **Caution**：「理论无限时序」依赖记忆机制；实际仍受显存、漂移与训推分布限制，teacher-forcing + replay 只能缓解而非消除 inconsistency。
3. **Expand**：BEV 稠密栅格 vs Sparse4D / query-based 稀疏表示的算力–召回权衡，原文未系统对比，落地需按算力预算单独立项。
4. **Correct**：Lift-Splat / CaDDN、CenterPoint 式热图头、Occupancy 多任务 loss，属于业界常见组合，方向正确。
5. **Expand**：DPO 负样本与 GRPO 在线优化如何与轨迹 Codebook 联合（离散 token 上做偏好），需另见 DPO / Codebook 专题。

## 延伸阅读建议

- Lift-Splat-Shoot；CaDDN；BEVFormer / BEVDet / BEVDepth
- PointPillars；StreamPETR / Sparse4D
- UniAD / VAD（感知–预测–规划联合）
- DiT（Peebles & Xie）；AdaLN；轨迹扩散 / Flow Matching 规划工作
- DPO / IPO；VQ-VAE

## 与其他主题的关联

- [VLM 训练阶段](../notes/01-vlm-training-stages.md) — AE / Planner 联训位置
- [因果建模](../notes/05-causal-modeling-trajectory.md) — 意图–轨迹拆分与 shortcut
- [数据 Trick](../notes/07-e2e-data-tricks.md) — 与真值改造、场景加权互补
- [AdamW](../notes/06-adamw-optimizer.md) — DiT/LLM 训练优化器
- [训练效率](../notes/04-training-efficiency.md) — 长时序 BEV / Transformer 显存
