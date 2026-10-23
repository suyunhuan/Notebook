# 手写笔记：VLA 选型与 L3/L4 工程路线

## 元信息

| 字段 | 内容 |
|------|------|
| 来源页码 | Handwriting Page 01（`_raw_batch_a.md` · a3f9-7d92） |
| 所属分类 | B. VLA / 决策 · B5 参考架构 / L3–L4 工程落地 |
| 笔记类型 | 手写精炼（非豆包对话） |
| 相关原始图 | `../assets/handwriting/01a169b1-a3f9-7d92-*.jpg` |

## 一句话摘要

L3/L4 落地的关键不是「再堆一个大模型」，而是用 **MOT 把离散语义与连续控制打通**，再配齐闭环仿真、数据效率、调度与统一表征空间。

## 核心关键词

`VLA` · `MindVLA` · `OneVLA` · `CFG` · `MOT` · `Memory` · `World Model` · `闭环` · `3DGS` · `因果对齐` · `L3/L4`

## 缩写与术语详解

| 缩写/词 | 全称 / 含义 | 在本页语境 |
|---------|-------------|------------|
| **VLA** | Vision-Language-Action | 视觉–语言–动作一体策略，输出可控车动作/轨迹 |
| **CFG** | Classifier-Free Guidance（亦见 Conditional Flow / 配置口语） | 手稿并列「离散+连续」；智驾里常指对条件引导的生成控制，或「配置」口语——**以「离散/连续双通路」读法为主**，勿与纯扩散 CFG 公式强行等同 |
| **MOT** | Mixture of Transformers（异于 Multi-Object Tracking） | 本知识库与 [16](16-qwen3vl-vla-architecture.md) 一致：多专家 Transformer 路由，打通离散 token 与连续轨迹支路 |
| **Memory** | 跨帧/跨 episode 状态记忆 | 原生多模态 + 记忆，支撑长时闭环 |
| **WM** | World Model | 世界模型：预测未来状态/占用，服务闭环与仿真 |
| **3DGS** | 3D Gaussian Splatting | 用于数据生成/场景重建，补稀有工况 |
| **RL** | Reinforcement Learning | 离散决策侧或闭环策略精调 |
| **L2 / L3 / L4** | SAE 自动驾驶等级 | L2 偏感知–推行–IG；L3/L4 强调多任务调度、数据效率、闭环与 WM |

> 手稿不确定项（Volvo/valvo、harness engine、curl、IG、主机贴付等）**不升格为定论**，仅作备忘。

## 知识脉络

### 1. 选型维度（手稿①②③）

1. **因果对齐 + 隐空间**：理解交通流与规则，而非只拟合轨迹像素。
2. **数据与策略双轨**：3DGS 扩数据；reasoning → policy（手稿「Volvo[?] policy」存疑）。
3. **架构候选对照**  
   - **MindVLA-U1**：CFG 思想下离散+连续；MOT；Memory。  
   - **self 路线**：离散 + RL，偏 AGI/长程。  
   - **harness / 快慢系统**：3D ViT + coding（1/2）分工。  
   - **XiaoMi OneVLA**：decoder / latent；RL。

### 2. 能力分层：L2 vs L3/L4

```
L2：物理层 { 感知、推行[?]、IG[?] } —— 开环/弱闭环够用
L3/L4：
  ├─ 多任务 + 调度器负载 + data efficiency
  ├─ 训练码 / 软件栈 / World Model
  └─ 真闭环（仿真↔实车）
```

「左脚/右脚」「物理抽离」：感知与决策交替迭代、或物理仿真与学习策略解耦再耦合——工程上即 **开环指标 ≠ 闭环安全**。

### 3. 工程落地清单（手稿结论）

| 支柱 | 要点 |
|------|------|
| 表征 | 统一表征空间；语言能力编码进特征；原生多模态 + Memory |
| 打通 | **MOT：离散语义 ↔ 连续控制** |
| 部署 | 高效部署、共系[?]软件栈 |
| 评测 | 数据平台 + 仿真闭环 + 评测体系三者齐备 |
| 基本面 | 大模型 Transformer 底座有基本面，再拓展 MOT / 闭环 |

### 4. 工程 knowhow

- 先定 **离散–连续接口**（meta-action / codebook / MOT 路由），再谈 AGI 叙事。
- L3/L4 瓶颈常在 **调度与数据效率**，不在单点 SOTA 感知分数。
- 评测若不闭环，L4 声称无效。

## 知识点校验

1. **Correct**：离散语义与连续控制需显式打通（MOT / codebook / meta-action），否则 VLA 易停在「会说不会开」。
2. **Caution**：手稿型号名（MindVLA-U1、OneVLA、Volvo policy、Gemini 等跨页混写）多为内部/口语，**不能当作公开产品规格**；MOT 在本库指 Mixture of Transformers，勿与 Multi-Object Tracking 混淆。
3. **Caution**：CFG 在扩散文献与手稿「离散+连续」并列时语义可能漂移——落地前必须对齐团队内部定义。
4. **Expand**：L3/L4 的「调度器负载」应落到具体多任务共卡策略（优先级、时分、算力预算）与闭环 KPI（接管率、碰撞近失、规则违规）。

## 与其他主题的关联

- [16 Qwen3-VL VLA 架构](16-qwen3vl-vla-architecture.md) — MOT / Codebook / Flow Action 插件式设计
- [01 VLM 训练阶段](01-vlm-training-stages.md) — 分阶段解冻与垂域增训
- [08 VLA 歧义与 CoC](08-vla-ambiguity-coc.md) — 离散可验证动作
- [11 轨迹 Codebook](11-trajectory-codebook.md) — 离散动作表示
- [02 BEV+PNC 栈](02-bev-pnc-stack.md) — 感知–规划工业分层
