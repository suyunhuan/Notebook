# 智驾BEV+PNC AI+LLM技术梳理

Thread ID: x2KWll9wqjafJJp2B
URL: https://www.doubao.com/thread/x2KWll9wqjafJJp2B

---

bev
backbone model
bev_backbone
caddn、uvp；多视角转 bev，FPN
FCN、FPN、SSFA 多尺度融合
bev_temporal
时序融合，坐标补偿和变换，融合方式：bev 交叉注意力、LSTM 融合、transformer+rnn 融合、GRU、Add
bevfpn_backbone
多尺度 bev；视频流 + 内部记忆，理论来讲无限时序，用 teacher-forcing 和 replay buffer 对齐分布解决训推不一致问题
bevfpn_nets
lidar radar，用 bevfpn 提取特征，空间位置编码，特征降维聚合 flattern pool，
feature_fusion_layer
lidar layernorm、groupnorm，image layernorm，再融合；可以用 gate 来融合，lidar 有时候要 drop 或者局部 mask 一下做数据增强；共享层提取共性，独立层提取特性，image 和 lidar 双流门控融合；cross attention，双向注意力，局部 33 的注意力避免计算爆炸
fv_backbone
regnet：Facebook AI Researcher；down/up sample，cat
resnet
改进版 resnet，结合了 FPN
fv_nets
fv backbone 和 fv head 的结合
fv2bev
liftsplat、caddn、cv3d 等投影方法，多视角多帧融合（多摄像头视角融合投影 bev，多传感器融合 lidar radar cam，多时序融合，上采样）；鱼眼重叠比较多一般用 average，其他一般是 cat，远处的一般保留时序增强
lidar_backbone
voxel（point pillar 等）、unet
lidar_det
lidar_nets
map_backbone
modal_bridge
moe_gate
pdpn_nets
radar_backbone
radar_nets
sdp_llm
streamdetr3d
streampetr
temporal_module
track
bev task/head
det_head
预先设计了一些 anchor（不同尺寸和朝向，grid），而不是直接预测，anchor 目标分配，分析正负样本，iou 交并比
bev heatmap 中心点预测，基于中心点的 offset、属性、速度、rotation 等预测，有不同的权重，近处、静态低速等加大 loss weight；gaussion focal loss，smooth l1 loss 等；conv2d
depthnet
encoder resnet
decoder
depth head，conv33，ELU 激活
DDN 深度分布
cognition_head
用 bev 特征 + MLP（或 with layernorm）做车道线检测和属性检测。query 是每个点，预测是否有车道线以及车道线坐标回归；匈牙利匹配、NMS 等后处理
curb_head
和车道线差不多
curb_vision_head
（conv2d+batchnorm）*2+Relu，conv up sample，conv2dbn，conv2d; 可见性检测
god_head
可行驶域：bev 下采样，可以融合 sd map（vector 处理，mask，添加位置编码，特征重塑，作为 kv 做 transformer 融合），上采样，conv 单通道 logits 再预测 sigmoid 概率
occupancy（CE/focal loss），voxel height（masked mse），voxel flow（动态物体权重更高，masked regression，flow loss），semantic（masked ce）；上采，然后预测，
lidar_det
polygon_head
predict_head
roadmark_head
seg_head
sparse4d_head
trajectory_head
loss
ddn、dice、discriminative、focal、guassian focal、hungarian matching、iou、rgs、smooth l1
pnc ai
trick
真值（Label）改造策略
曲率限速、舒适加速度轨迹、回正轨迹（非路口直线多车道巡航场景，30km/h 以上，根据车速和横向距离调整回正时间，并且做碰撞检查）、启停真值（连续 5 帧速度是 0 的清空未来轨迹，补偿底盘延迟 400ms 取第一帧算速度，速度阈值 0.05m/s；起步数据，有效起步是 0.6m，太短的过滤掉）修改
样本配比 / 数据采样策略
Loss 函数 & 动态权重设计
Reward 建模（强化学习 RL 基础）
舒适性（时间衰减的总纵向 acc 以及 jerk 表）、动态碰撞、动态风险、效率、跟车、goal、gt、车道中心线、车道线、导航、静态碰撞、红绿灯
偏好对齐类：DPO / IPO 等（替代传统 RLHF）
负样本构造，比如沿着 s 缩放、碰撞样本、换道样本、横向碰撞样本
策略梯度类：GRPO、PPO 在线优化
数据增强（ego 扰动等）、清洗
vqvae 原理，轨迹 codebook
model
basic
RMSNorm 比 layernorm 计算量笑
位置信息注入 Q 和 K 中，RoPE
GQA 活 MQA 改造 head 数量和 Q 一致，减少 KV 缓存
eager attention
主模型
输入处理，past kv cache position；混合掩码，下三角；位置编码注入，把 RoPE 注入到 hidden states；decoder 循环，attention 和 mlp；再 norm 和 linear 得到 logits，取最后欧 10 个 token 的 hidden；再预测 diffusion flow
DiT；ego 和 time（过 time emb），之后当做条件注入 AdaLN；带噪声的轨迹，经过 pos emb 之后，当做 qkv 过多层 dit
llm
数据设计
模型结构
mot
作为特征注入
智驾 BEV + PNC AI + LLM 技术大纲梳理（整理版，按模块分层，补充简要释义）
整体架构分为三大块：BEV 感知模块、PNC AI（规划控制大模型 / 轨迹生成）、LLM（大模型辅助）
一、BEV 感知
1. BEV Backbone 特征主干
bev_backbone（图像多视角→BEV 投影主干）
CADDN、UVP：多视角图像升维投影至鸟瞰 BEV 空间
FPN / FCN / SSFA：多尺度 BEV 特征融合；SSFA 是空间尺度自适应融合
bev_temporal（BEV 时序模块）
核心：多帧 BEV 时序融合，做坐标补偿 + 外参变换（补偿自车运动）
融合方案：BEV 交叉注意力、LSTM、Transformer+RNN、GRU、直接 Add 相加融合
bevfpn_backbone
多尺度 BEV 特征金字塔；带视频流 + 内部记忆池，理论支持无限长时序
解决训推不一致：teacher-forcing + replay buffer 做分布对齐
bevfpn_nets（多传感器 BEV 提取）
输入：LiDAR + Radar
流程：BEVFPN 提取特征 → 空间位置编码 → 特征降维、聚合、flatten/pooling
feature_fusion_layer（多模态特征融合层，图像 / 激光雷达）
归一化：LiDAR 用 LN/GN；Image 用 LN
融合策略：门控 gate 融合；可对 LiDAR 做随机 drop、局部 mask 数据增强
双流设计：共享层提取共性特征，独立分支保留模态特有特征；门控双流融合
Cross-Attention 双向注意力；局部 3×3 注意力，控制算力爆炸
fv_backbone（前视图像 FV 主干，单视角 2D 特征提取）
RegNet（FAIR）、ResNet、改进 ResNet+FPN；下采样 / 上采样，concat 拼接
fv_nets：fv_backbone + fv_head 整套前视检测网络
fv2bev（前视图像投影 BEV，多模态多帧融合模块）
投影算法：Lift-Splat、CADDN、CV3D
融合维度：多摄像头视角投影 BEV；Cam+LiDAR+Radar 多传感器；多时序帧融合 + 上采样
融合规则：鱼眼相机重叠大，多用 average 平均；普通相机 concat 拼接；远景依赖时序增强
lidar_backbone
Voxel 类：PointPillar、UNet 体素网络
lidar_nets /lidar_det：激光检测整套网络
radar_backbone /radar_nets：雷达特征主干与检测网络
map_backbone：高精地图 BEV 特征提取主干
modal_bridge：模态桥接模块，不同模态特征空间映射对齐
moe_gate：MoE 门控，动态激活专家网络，不同场景路由
pdpn_nets
sdp_llm
streamdetr3d /streampetr：流式 3D DETR 系列，时序 BEV Transformer 检测
temporal_module：通用时序模块
track：多目标跟踪模块
2. BEV Task / Head 任务头
det_head（3D 检测头）
Anchor 范式：预定义不同尺寸、朝向的 grid anchor；anchor 分配、正负样本筛选、IOU
Center 热图范式：BEV 中心点热图预测；同时预测 offset、物体属性、速度、旋转角
Loss 加权：近处目标、静态 / 低速物体加大 loss 权重
Loss：Gaussian Focal Loss、Smooth L1 Loss；基础算子 Conv2d
depthnet（图像深度估计头）
Encoder：ResNet
Decoder
Depth head：3×3 Conv + ELU 激活
DDN：深度分布预测
cognition_head（认知任务头）
BEV 特征 + MLP（可搭配 LN），车道线检测 + 属性识别
Query 为采样点，预测车道线存在性 + 坐标回归；匈牙利匹配、NMS 后处理
curb_head（路沿检测头，逻辑类似车道线）
curb_vision_head（路沿可见性检测）
堆叠 (Conv2d+BN)*2+ReLU → 上采样 → ConvBN，输出路沿可见性
god_head（通用占用 / 可行驶域头）
可行驶域：BEV 下采样，融合 SD 矢量地图；vector 转 mask、位置编码，特征重塑，作为 KV 送入 Transformer 融合；上采样，Conv 输出单通道 logits+sigmoid 概率
Occupancy 占用网络：CE/Focal Loss；voxel 高度（Masked MSE）；voxel flow 体素流（动态物体更高权重，masked 回归 flow loss）；语义分类（Masked CE）
polygon_head
predict_head
roadmark_head（路面标识）
seg_head（BEV 语义分割）
sparse4d_head（4D 时空稀疏预测头）
trajectory_head（轨迹预测头，预测周边物体未来轨迹）
3. BEV Loss 损失集合
DDN、Dice、Discriminative、Focal、Gaussian Focal、Hungarian Matching、IOU、RGS、Smooth L1
二、PNC AI（规划控制 AI / 轨迹生成模型）
核心：基于 BEV 感知输出，生成自车行驶轨迹，包含真值改造、采样、强化学习、偏好对齐、轨迹 Codebook
trick 工程技巧
真值 Label 改造策略
曲率限速、舒适加速度轨迹、回正轨迹：非路口直道多车道巡航，>30km/h，根据车速 + 横向距离调整回正时间，增加碰撞校验
启停真值：连续 5 帧速度置 0 则清空未来轨迹；底盘延迟补偿 400ms，取首帧计算速度，阈值 0.05m/s；起步有效样本过滤，有效起步≥0.6m，过滤短片段
样本配比 / 数据采样策略
Loss 函数 & 动态权重设计
Reward 建模（RL 强化学习）
奖励项：舒适性（时间衰减纵向 acc+jerk）、动态碰撞、动态风险、通行效率、跟车、目标 goal、GT 轨迹、车道中心线、车道线、导航约束、静态碰撞、红绿灯
偏好对齐 DPO / IPO（替代 RLHF）
负样本构造：s 方向缩放轨迹、碰撞轨迹、换道风险轨迹、横向碰撞样本
策略梯度：GRPO、PPO 在线优化；ego 自车扰动数据增强 + 数据清洗
VQVAE + 轨迹 Codebook：轨迹离散编码，把连续轨迹变成离散 token，方便大模型处理
model 模型层
basic 基础组件
RMSNorm：计算量小于 LayerNorm
RoPE：位置信息注入 Q、K
GQA / MQA：减少 KV 缓存，head 数量与 Q 保持一致
eager attention：高效 attention 实现
主模型（轨迹 DiT + Diffusion Flow）
输入处理、past kv cache、位置；混合掩码（下三角）；RoPE 注入 hidden states
Decoder 循环：Attention + MLP → Norm + Linear 输出 logits；取最后 10 个 token hidden state，预测 diffusion flow
DiT 条件机制：ego 状态 + time 做 time embedding，AdaLN 作为条件；带噪声轨迹做 pos embedding，送入多层 DiT 作为 QKV
三、LLM 大模型模块
数据设计
模型结构
MOT 多目标跟踪
LLM 作为特征注入：将 LLM 输出的语义 / 决策特征，融入感知或 PNC 模型
精简总结
整套是端到端智驾 VLA（Vision-Language-Action）架构：
BEV 多模态感知（图像 / 激光 / 雷达 / 地图）→ 感知头输出检测、占用、车道、目标轨迹 → PNC 的 DiT/Diffusion+VQVAE+Codebook 做自车轨迹生成（搭配 RL/DPO 偏好对齐）→ LLM 提供高层语义、导航、场景理解，特征注入下游。
如果你需要，我可以帮你输出：
思维导图 markdown（可导入 XMind/obsidian）
架构框图文字版
把这份大纲压缩成一页 PPT 提纲
按数据流顺序重排（输入→感知→时序融合→预测→规划→控制）
你想要哪个？

bev
backbone model
bev_backbone
caddn、uvp；多视角转 bev，FPN
FCN、FPN、SSFA 多尺度融合
bev_temporal
时序融合，坐标补偿和变换，融合方式：bev 交叉注意力、LSTM 融合、transformer+rnn 融合、GRU、Add
bevfpn_backbone
多尺度 bev；视频流 + 内部记忆，理论来讲无限时序，用 teacher-forcing 和 replay buffer 对齐分布解决训推不一致问题
bevfpn_nets
lidar radar，用 bevfpn 提取特征，空间位置编码，特征降维聚合 flattern pool，
feature_fusion_layer
lidar layernorm、groupnorm，image layernorm，再融合；可以用 gate 来融合，lidar 有时候要 drop 或者局部 mask 一下做数据增强；共享层提取共性，独立层提取特性，image 和 lidar 双流门控融合；cross attention，双向注意力，局部 33 的注意力避免计算爆炸
fv_backbone
regnet：Facebook AI Researcher；down/up sample，cat
resnet
改进版 resnet，结合了 FPN
fv_nets
fv backbone 和 fv head 的结合
fv2bev
liftsplat、caddn、cv3d 等投影方法，多视角多帧融合（多摄像头视角融合投影 bev，多传感器融合 lidar radar cam，多时序融合，上采样）；鱼眼重叠比较多一般用 average，其他一般是 cat，远处的一般保留时序增强
lidar_backbone
voxel（point pillar 等）、unet
lidar_det
lidar_nets
map_backbone
modal_bridge
moe_gate
pdpn_nets
radar_backbone
radar_nets
sdp_llm
streamdetr3d
streampetr
temporal_module
track
bev task/head
det_head
预先设计了一些 anchor（不同尺寸和朝向，grid），而不是直接预测，anchor 目标分配，分析正负样本，iou 交并比
bev heatmap 中心点预测，基于中心点的 offset、属性、速度、rotation 等预测，有不同的权重，近处、静态低速等加大 loss weight；gaussion focal loss，smooth l1 loss 等；conv2d
depthnet
encoder resnet
decoder
depth head，conv33，ELU 激活
DDN 深度分布
cognition_head
用 bev 特征 + MLP（或 with layernorm）做车道线检测和属性检测。query 是每个点，预测是否有车道线以及车道线坐标回归；匈牙利匹配、NMS 等后处理
curb_head
和车道线差不多
curb_vision_head
（conv2d+batchnorm）*2+Relu，conv up sample，conv2dbn，conv2d; 可见性检测
god_head
可行驶域：bev 下采样，可以融合 sd map（vector 处理，mask，添加位置编码，特征重塑，作为 kv 做 transformer 融合），上采样，conv 单通道 logits 再预测 sigmoid 概率
occupancy（CE/focal loss），voxel height（masked mse），voxel flow（动态物体权重更高，masked regression，flow loss），semantic（masked ce）；上采，然后预测，
lidar_det
polygon_head
predict_head
roadmark_head
seg_head
sparse4d_head
trajectory_head
loss
ddn、dice、discriminative、focal、guassian focal、hungarian matching、iou、rgs、smooth l1
pnc ai
trick
真值（Label）改造策略
曲率限速、舒适加速度轨迹、回正轨迹（非路口直线多车道巡航场景，30km/h 以上，根据车速和横向距离调整回正时间，并且做碰撞检查）、启停真值（连续 5 帧速度是 0 的清空未来轨迹，补偿底盘延迟 400ms 取第一帧算速度，速度阈值 0.05m/s；起步数据，有效起步是 0.6m，太短的过滤掉）修改
样本配比 / 数据采样策略
Loss 函数 & 动态权重设计
Reward 建模（强化学习 RL 基础）
舒适性（时间衰减的总纵向 acc 以及 jerk 表）、动态碰撞、动态风险、效率、跟车、goal、gt、车道中心线、车道线、导航、静态碰撞、红绿灯
偏好对齐类：DPO / IPO 等（替代传统 RLHF）
负样本构造，比如沿着 s 缩放、碰撞样本、换道样本、横向碰撞样本
策略梯度类：GRPO、PPO 在线优化
数据增强（ego 扰动等）、清洗
vqvae 原理，轨迹 codebook
model
basic
RMSNorm 比 layernorm 计算量笑
位置信息注入 Q 和 K 中，RoPE
GQA 活 MQA 改造 head 数量和 Q 一致，减少 KV 缓存
eager attention
主模型
输入处理，past kv cache position；混合掩码，下三角；位置编码注入，把 RoPE 注入到 hidden states；decoder 循环，attention 和 mlp；再 norm 和 linear 得到 logits，取最后欧 10 个 token 的 hidden；再预测 diffusion flow
DiT；ego 和 time（过 time emb），之后当做条件注入 AdaLN；带噪声的轨迹，经过 pos emb 之后，当做 qkv 过多层 dit
llm
数据设计
模型结构
mot
作为特征注入

智驾 BEV + PNC AI + LLM 技术大纲梳理（整理版，按模块分层，补充简要释义）
整体架构分为三大块：BEV 感知模块、PNC AI（规划控制大模型 / 轨迹生成）、LLM（大模型辅助）
一、BEV 感知
1. BEV Backbone 特征主干
bev_backbone（图像多视角→BEV 投影主干）
CADDN、UVP：多视角图像升维投影至鸟瞰 BEV 空间
FPN / FCN / SSFA：多尺度 BEV 特征融合；SSFA 是空间尺度自适应融合
bev_temporal（BEV 时序模块）
核心：多帧 BEV 时序融合，做坐标补偿 + 外参变换（补偿自车运动）
融合方案：BEV 交叉注意力、LSTM、Transformer+RNN、GRU、直接 Add 相加融合
bevfpn_backbone
多尺度 BEV 特征金字塔；带视频流 + 内部记忆池，理论支持无限长时序
解决训推不一致：teacher-forcing + replay buffer 做分布对齐
bevfpn_nets（多传感器 BEV 提取）
输入：LiDAR + Radar
流程：BEVFPN 提取特征 → 空间位置编码 → 特征降维、聚合、flatten/pooling
feature_fusion_layer（多模态特征融合层，图像 / 激光雷达）
归一化：LiDAR 用 LN/GN；Image 用 LN
融合策略：门控 gate 融合；可对 LiDAR 做随机 drop、局部 mask 数据增强
双流设计：共享层提取共性特征，独立分支保留模态特有特征；门控双流融合
Cross-Attention 双向注意力；局部 3×3 注意力，控制算力爆炸
fv_backbone（前视图像 FV 主干，单视角 2D 特征提取）
RegNet（FAIR）、ResNet、改进 ResNet+FPN；下采样 / 上采样，concat 拼接
fv_nets：fv_backbone + fv_head 整套前视检测网络
fv2bev（前视图像投影 BEV，多模态多帧融合模块）
投影算法：Lift-Splat、CADDN、CV3D
融合维度：多摄像头视角投影 BEV；Cam+LiDAR+Radar 多传感器；多时序帧融合 + 上采样
融合规则：鱼眼相机重叠大，多用 average 平均；普通相机 concat 拼接；远景依赖时序增强
lidar_backbone
Voxel 类：PointPillar、UNet 体素网络
lidar_nets /lidar_det：激光检测整套网络
radar_backbone /radar_nets：雷达特征主干与检测网络
map_backbone：高精地图 BEV 特征提取主干
modal_bridge：模态桥接模块，不同模态特征空间映射对齐
moe_gate：MoE 门控，动态激活专家网络，不同场景路由
pdpn_nets
sdp_llm
streamdetr3d /streampetr：流式 3D DETR 系列，时序 BEV Transformer 检测
temporal_module：通用时序模块
track：多目标跟踪模块
2. BEV Task / Head 任务头
det_head（3D 检测头）
Anchor 范式：预定义不同尺寸、朝向的 grid anchor；anchor 分配、正负样本筛选、IOU
Center 热图范式：BEV 中心点热图预测；同时预测 offset、物体属性、速度、旋转角
Loss 加权：近处目标、静态 / 低速物体加大 loss 权重
Loss：Gaussian Focal Loss、Smooth L1 Loss；基础算子 Conv2d
depthnet（图像深度估计头）
Encoder：ResNet
Decoder
Depth head：3×3 Conv + ELU 激活
DDN：深度分布预测
cognition_head（认知任务头）
BEV 特征 + MLP（可搭配 LN），车道线检测 + 属性识别
Query 为采样点，预测车道线存在性 + 坐标回归；匈牙利匹配、NMS 后处理
curb_head（路沿检测头，逻辑类似车道线）
curb_vision_head（路沿可见性检测）
堆叠 (Conv2d+BN)*2+ReLU → 上采样 → ConvBN，输出路沿可见性
god_head（通用占用 / 可行驶域头）
可行驶域：BEV 下采样，融合 SD 矢量地图；vector 转 mask、位置编码，特征重塑，作为 KV 送入 Transformer 融合；上采样，Conv 输出单通道 logits+sigmoid 概率
Occupancy 占用网络：CE/Focal Loss；voxel 高度（Masked MSE）；voxel flow 体素流（动态物体更高权重，masked 回归 flow loss）；语义分类（Masked CE）
polygon_head
predict_head
roadmark_head（路面标识）
seg_head（BEV 语义分割）
sparse4d_head（4D 时空稀疏预测头）
trajectory_head（轨迹预测头，预测周边物体未来轨迹）
3. BEV Loss 损失集合
DDN、Dice、Discriminative、Focal、Gaussian Focal、Hungarian Matching、IOU、RGS、Smooth L1
二、PNC AI（规划控制 AI / 轨迹生成模型）
核心：基于 BEV 感知输出，生成自车行驶轨迹，包含真值改造、采样、强化学习、偏好对齐、轨迹 Codebook
trick 工程技巧
真值 Label 改造策略
曲率限速、舒适加速度轨迹、回正轨迹：非路口直道多车道巡航，>30km/h，根据车速 + 横向距离调整回正时间，增加碰撞校验
启停真值：连续 5 帧速度置 0 则清空未来轨迹；底盘延迟补偿 400ms，取首帧计算速度，阈值 0.05m/s；起步有效样本过滤，有效起步≥0.6m，过滤短片段
样本配比 / 数据采样策略
Loss 函数 & 动态权重设计
Reward 建模（RL 强化学习）
奖励项：舒适性（时间衰减纵向 acc+jerk）、动态碰撞、动态风险、通行效率、跟车、目标 goal、GT 轨迹、车道中心线、车道线、导航约束、静态碰撞、红绿灯
偏好对齐 DPO / IPO（替代 RLHF）
负样本构造：s 方向缩放轨迹、碰撞轨迹、换道风险轨迹、横向碰撞样本
策略梯度：GRPO、PPO 在线优化；ego 自车扰动数据增强 + 数据清洗
VQVAE + 轨迹 Codebook：轨迹离散编码，把连续轨迹变成离散 token，方便大模型处理
model 模型层
basic 基础组件
RMSNorm：计算量小于 LayerNorm
RoPE：位置信息注入 Q、K
GQA / MQA：减少 KV 缓存，head 数量与 Q 保持一致
eager attention：高效 attention 实现
主模型（轨迹 DiT + Diffusion Flow）
输入处理、past kv cache、位置；混合掩码（下三角）；RoPE 注入 hidden states
Decoder 循环：Attention + MLP → Norm + Linear 输出 logits；取最后 10 个 token hidden state，预测 diffusion flow
DiT 条件机制：ego 状态 + time 做 time embedding，AdaLN 作为条件；带噪声轨迹做 pos embedding，送入多层 DiT 作为 QKV
三、LLM 大模型模块
数据设计
模型结构
MOT 多目标跟踪
LLM 作为特征注入：将 LLM 输出的语义 / 决策特征，融入感知或 PNC 模型
精简总结
整套是端到端智驾 VLA（Vision-Language-Action）架构：
BEV 多模态感知（图像 / 激光 / 雷达 / 地图）→ 感知头输出检测、占用、车道、目标轨迹 → PNC 的 DiT/Diffusion+VQVAE+Codebook 做自车轨迹生成（搭配 RL/DPO 偏好对齐）→ LLM 提供高层语义、导航、场景理解，特征注入下游。
如果你需要，我可以帮你输出：
思维导图 markdown（可导入 XMind/obsidian）
架构框图文字版
把这份大纲压缩成一页 PPT 提纲
按数据流顺序重排（输入→感知→时序融合→预测→规划→控制）
你想要哪个？