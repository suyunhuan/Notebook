# Handwriting Raw Batch A

> 来源：7 张手写笔记照片（附件顺序 = Page 01–07）。转录已做笔误/缩写还原；模糊处标 `[?]`。

## Page 01 — a3f9-7d92

**主题**: VLA/MindVLA–OneVLA 架构选型：离散–连续打通、闭环与 L3/L4 工程落地要点。

**转录（纠错后）**:
- ① causal 对齐；隐空间；交通流 / 规则的理解
- ② 3DGS → 数据生成
- ② reasoning → Volvo [?] policy model
- MindVLA-U1 | CFG：离散 + 连续 | MOT | Memory
- self | 离散 + RL | AGI
- harness engine [?] → 快慢；3D ViT；coding（1/2）
- XiaoMi OneVLA | decoder / latent 思路；GPT 开天 [?]；RL
- curl [?]
- ② 闭环模型
- 物理抽离完美；（Shift）；左脚 / 右脚
- L2：物理 { 感知、推行 [?]、IG [?] }
- L3/L4：
  - ① 多任务、调度器负载；data efficiency
  - 主机贴付 [?] / 训练码 / 软件 / WM（World Model）
  - 闭环
- 编码能力 → 特征 { Language }
- ② 原生多模态、带 memory 等（A-）
- ③ 高效部署、统一的表征空间、共系 [?]
- ⇒ MOT → 离散、连续打通
- 评测；工程落地：数据、仿真平台、评测
- 大模型原生：有基本面；Transformer — MOT；再去拓展

**不确定项**: Volvo（原文 valvo）；harness engine；curl；IG；主机贴付；共系；开天；推行。


## Page 02 — a4bb-7ae7

**主题**: LLM→VL→4D 多阶段训练管线（pretrain / SFT / RL）及 2D/4D 数据规模。

**转录（纠错后）**:
- 网络结构 → Transformer
- 规模：model / data / compute
- 建模范式：AR / Diffusion
- ~~离线 → 在线~~（划掉）
- 主链路：LLM → VL-stage 1 → VL-stage 2 → VL-stage 2.5
- 自 VL-stage 2 并行 4D：
  - 4D Encoder pre-train（100M；离线类 CLIP）
  - → 4D-stage 1（仅训 [?] 4D projector；10M 4D）
  - → 4D-stage 2（100M 4D；仅训 [?] 4D encoder & projector）
- 后期：2D/4D stage 2.5 → 2D/4D-SFT → 2D/4D RL
- 2D、4D 全类混训，全量
- 数据规模：150M 2D；110M 4D

**不确定项**: 「仅训」原文似 Pi ml / 冻结类笔迹，按阶段惯例推断为仅训 projector/encoder；「全量」原文略似「全录」。


## Page 03 — a59b-7cb0

**主题**: 感知结构化特征：NaviNN / TrafficNN / Ego-HMF 串级产出导航场、通行概率与轨迹指导。

**转录（纠错后）**:
- 标题：感知结构化特征
- 导航信息输入：TBT、EHP
- **NaviNN（静态导航场）**
  - 输入：孤立场景关系；BEV、PV 时序特征
  - → NaviNN Encoder → NaviNN Feat → NaviNN Decoder（+ query points）→ 导航场
- **TrafficNN（红绿灯）**
  - 输入：红绿灯模糊特征；BEV、PV 时序特征
  - → TrafficNN Encoder → TrafficNN Feat（原文 TraficcNN）→ TrafficNN Decoder → 路口通行概率
  - NaviNN Feat 注入 TrafficNN Feat
- 侧注（串级）：全局导航（远端约 500m）→ 到局部 → 轨迹更有效
- **Ego-HMF（感知隐层特征）**
  - 输入：BEV、PV 时序 [?]；结构化；Traffic Feat + NaviNN Feat
  - → Ego-HMF Encoder → Ego 特征 → Ego-HMP Decoder → 轨迹指导
- 下游：FSD 规划；轨迹预测；占据场；速度场；导航场；红绿灯通行概率
- 引入 world model

**不确定项**: Ego-HMF / Ego-HMP 全称；「B.P. 时」按同页写法还原为 BEV、PV 时序；孤立场景关系是否为「孤立」或「关联」场景。


## Page 04 — a659-7b66

**主题**: 多传感器 BEV/PV 感知全链路：融合特征、Cognition/GOD/SparseDet/PV 多头与后处理。

**转录（纠错后）**:
- SD zoom → ISD fusion → PV Decoder → lane query (48, 512)
  - Linear → boundaries (48, 256, 2)（原文 boundres）
  - Linear → lane exist (48, 256)
  - Linear → lane vis (48 [?], 256)（原文似 748）
- cam → fv-backbone（特征提取）→ cam2 feat (512, 44, 80)
  - LSS 投影 → persp [?] fv (64, 88, 160)、cyl fv (64, 80, 64) → cam bev feat (64, 352, 320)
- RL（雷达/激光 [?]）→ voxelize → UNet → rl bev feat (64, 352, 320)
- cam bev + rl bev → concat → fusion bev feat (128, 352, 320)
- fv feat (256, 22, 80)×7；cyl fv feat (256, 20, 16)×4；pos emb；w/ crop → fusion pv feature (192, 125 [?], 8, 256)
- **Cognition（远程超高）**：T-1 feature (128, 88, 80) → cog temporal → road intersection seg head；lane head / attributes / lane / lane bound；lane instance feat (48, 512)；当前帧时序特征 (128, 88, 80)
- **GOD**：T-1 feature (128, 88, 56) → GOD temporal → GOD head → GOD output (704, 448, 5)；当前帧时序特征 (128, 88, 56)
- **Sparse BEV**：det query (556, 256) → sparse DetHead → instance feature (556, 256)；Obj center (x,y) (556, 2)
- **PV 任务**：FPN 多尺度 (128,88,160)/(256,44,80)/(512,22,40)；PV 预测 → 检测 / 分割；白名单、成排锥桶、车道线、lane Merge、指标、黑名单、2D 人车；semantic freespace；路面湿滑 [?]
- 历史帧特征 hidden state；红绿灯播报
- 三路 GRU + conv + sigmoid（G+c+s）：128×88×160 → 140×(NC+5)；128×44×80 → 3520×(NC+5)；128×22×40 → 880
- driveable query (112, 256) → driveable area head → instance feat（可行驶区域）
- concat → 后处理 NMS → Box / 类别 / 背景（原文后景）
- 辅助：obj prev instance feat (556, 10, 256)；static RG_mask (100, 11)；obj time (556, 11)；obj mask (556, 11)

**不确定项**: RL 传感器具体指 radar 还是 lidar；pho fv→persp fv；lane vis 748 vs 48；fusion pv 维度 125；路面湿滑/遥测；ISD 全称。


## Page 05 — a71a-727f

**主题**: BEV CenterNet 式检测头：anchor、高斯热力图中心、回归项与分场景 loss 加权。

**转录（纠错后）**:
- anchor generator：2500×6；(50×50)；2 尺寸 [? ]×；2 height、2 方向 (0, 90)；188×188×12 = 424128
- BEV focal loss；Gaussian heatmap 中心点；再回归 attributes（原文 attribution）
- 点云输入 → Backbone → BEV Feature C×H×W
- → Conv Head：K×H×W 概率图
- → Thresholding → local max suppression → decode center (u, v)
- 回归：offset、size、rotation、vel、class；traj
- NMS
- loss 加权：
  - &lt; 0.05 → 1.5× loss
  - v &gt; 0.05 → 2× loss
  - 动静 ×3
  - 通用 [?] rotation car ×3
  - 匝道处 VRU rotation &gt; 3 [?]
- Conv2d head

**不确定项**: 「2 尺寸」措辞；「通用」原文似通效；匝道处 VRU 加权条件书写不完整；阈值 0.05 对应物理量（速度？）未写明。


## Page 06 — a7d4-77eb

**主题**: ego process：未来推演（TTC/碰撞）、PE/时序注意力与纵向交互–风险降速策略。

**转录（纠错后）**:
- ego process
- 未来推演：TTC、DAC（δ）、z-p(speed)、collision（δ）
- head
- 输入 / 输出 / 时序；倒车；Gemini 3 chip
- PE（positional embedding）；Granularities attention
- TempCompass
- 周一、二：代码仓统一
- 纵向 { 交互 / 感知推演；语义 / 风险降速 }
- ARDP

**不确定项**: DAC 全称；δ 标注含义；z-p(speed) 记号；Gemini 3 chip 是否硬件型号；ARDP 全称；Granularities attention 是否专有名词。


## Page 07 — a892-746c

**主题**: Teacher/Self Forcing、meta-action 与 VLM 中间层读存（kvcache）对接 FM→DiT。

**转录（纠错后）**:
- teacher forcing；train/test 分布
- self forcing：10% train 不好，不按常规
- meta action segmentation
- （示意曲线，标 klr [?]）
- MA 对网络影响很小；traj 大
- gradient L2；构造 ground[-truth] 数据
- FM；L0 [?] hidden 90% attention
- 0、9、18 层 → 到对应部分
- kvcache 28 → 3；读存
- [图] VLM（层 0 / 9 / 18）—读→ GR0P [?]；ViT / FM
- [图] FM → DiT（另有一路输入进 DiT）

**不确定项**: klr；L0（或 Lo / to）；GR0P（或 GRDP / GRPO）；meta 字迹较草（按语境定为 meta action）。
