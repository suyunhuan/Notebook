# PV Backbone、Neck 与 View Transform（含 RegNet）

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | RegNet实现原理 |
| Thread ID | `xTGPZXuruwC5E92Xm` |
| 源链接 | https://www.doubao.com/thread/xTGPZXuruwC5E92Xm |
| 所属分类 | 智能驾驶核心技术 / 感知骨干与视图变换 |

## 一句话摘要

环视感知链路是「共享 Backbone → Neck 多尺度融合 → View Transform（LSS / Query / IPM）」；RegNet 是高效 CNN 骨干设计空间，DNet 指 LSS 深度头而非另一套主干，AscendNet 多为昇腾部署口语而非网络结构名。

## 核心关键词

`PV→BEV` · `ResNet` · `RegNet` · `FPN/BiFPN` · `LSS` · `BEVFormer` · `DNet/DepthHead` · `NoStem-RegNet` · `Ascend/CANN`

## 知识脉络

### 1. 完整数据流

```
多相机图像
 → Backbone（共享权重，分相机前向）
 → Neck（FPN 等）
 → View Transform（PV→BEV，核心）
 → BEV Encoder + 检测/分割/地图头
```

### 2. Backbone 选型

| 类型 | 代表 | 用途画像 |
|------|------|----------|
| CNN | ResNet/ResNeXt、VoVNet、MobileNet 系 | 量产基线；VoVNet 常提光照鲁棒；轻量网降精度 |
| 高效 CNN | **RegNetX/Y**、NoStem 变体 | 同 FLOPs 精度/延迟更好；硬件友好分组卷积 |
| ViT | Swin、DINOv2+Adapter | 高算力平台；全局语义强 |

要点：输出多尺度（常 stride 8/16/32）；ImageNet 预训练 + 智驾二次预训练；量化友好。

### 3. Neck

- **FPN**：最通用。  
- **BiFPN**：双向加权，偏轻量。  
- **GeneralizedLSSFPN** 等：对齐 LSS 通道/分辨率。  
工程常以 **stride=8 主特征** 进视图变换以控显存。

### 4. View Transform 三路线

| 路线 | 机制 | 优劣 |
|------|------|------|
| **LSS** | depth bin 概率 Lift → 3D → Splat 到 BEV | 可解释、易部署；远距深度误差放大；需深度监督 |
| **Query Transformer** | BEV Query + 3D 参考点投影 + Deformable SCA（± TSA） | 无显式深度、时空强；算力高、NPU 难、标定敏感 |
| **IPM** | 地面平面单应 | 极快；有高度物体畸变，高速 3D 主感知已淘汰 |

相关：PETR 系可不建稠密 BEV grid，直接 3D object query。

### 5. 名词澄清：ResNet / RegNet / DNet / AscendNet

| 名称 | 实质 |
|------|------|
| ResNet | 残差 CNN 骨干；BEV 论文默认基线 |
| RegNet | 设计空间 NAS 归纳出的线性宽度增长 CNN；X=分组卷积，Y=X+SE |
| **DNet** | 智驾 LSS 语境下几乎总是 **Depth Net（深度头）**，不是主干；勿与 DLA/DenseNet 混淆 |
| AscendNet | **非**标准骨干论文名；口语指昇腾 CANN/ATC 优化部署后的模型实例 |

**LSS 串起来**：Backbone(RegNet/ResNet) → FPN → **DNet** → Lift-Splat → BEV 头 →（可选）昇腾部署。

### 6. RegNet 实现要点（Designing Network Design Spaces）

- 思想：搜的是**设计规则**，不是单个网；block 宽度近似线性 \(u_j=w_0+w_a\cdot j\)，再 \(w_m\) 量化，并切成 4 个 stage，**stage 内通道恒定**（对比 ResNet 指数加倍）。
- 超参：\(w_0,w_a,w_m,d,group\_w\)。
- 结构：Stem + 4 RegStage + Head（BEV 丢弃 Head）。
- Block：1×1 → 3×3 **GroupConv（固定 group_width）** → 1×1；Y 在 3×3 后加 SE。
- 智驾：**NoStem-RegNet**（小卷积堆叠替大 stem）降低 NPU 延迟/量化损失；原文称 FSD 等用量产 RegNetY——**需独立核实**。

### 7. 量产坑点

- 标定漂移、俯仰变化、时戳不同步 → BEV 错位；在线标定 + 外参扰动训练。  
- 多相机重叠区融合冲突。  
- 远距（如 >60m）特征稀疏 → 时序累积。  
- 显存：降 BEV 分辨率、稀疏采样、定制 splat/attention 算子（Fast-BEV 一类优化方向）。  
- LSS 要深度标签；BEVFormer 可不需深度 GT 但更难训。

## 知识点校验

- **Correct**：RegNet「线性宽度 + stage 内等宽 + 分组卷积」相对 ResNet 的结构差异；DNet=DepthHead 的领域用法澄清有价值。
- **Caution**：「特斯拉 FSD HydraNet 主干就是 RegNet / NoStem」等行业传言在公开材料中表述不一，笔记中应标为**未强制证实的二手信息**。
- **Caution**：将「理想/毫末/小鹏/地平线 = LSS」「特斯拉 = BEVFormer」表格化归属过度简化，实际多为混合与自研变体。
- **Expand**：RegNet vs ResNet 在目标 NPU 上的延迟/量化掉点实测；LSS depth CE vs 期望深度回归的消融；deformable attention 的车规算子支持矩阵。

## 延伸阅读建议

- *Designing Network Design Spaces*（RegNet，Radosavovic et al., CVPR 2020）  
- ResNet；SENet（理解 RegNetY）  
- LSS；BEVDet；BEVFormer；BEVFusion；Fast-BEV  
- 昇腾 CANN / ATC 部署文档（算子与量化，而非「AscendNet」论文）

## 与其他主题的关联

- BEV 深度与时序路线对比：[`../notes/13-bev-depth-models.md`](../notes/13-bev-depth-models.md)
- BEV+PNC 总览：[`../notes/02-bev-pnc-stack.md`](../notes/02-bev-pnc-stack.md)
- 训练效率与分布式：[`../notes/04-training-efficiency.md`](../notes/04-training-efficiency.md)
- VLA 感知分支选型：[`../notes/16-qwen3vl-vla-architecture.md`](../notes/16-qwen3vl-vla-architecture.md)
