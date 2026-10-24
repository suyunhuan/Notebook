# BEV 深度与视图变换：路线分类对比

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | 自动驾驶BEV深度模型对比 |
| Thread ID | `xgeM04177ZmZm4AnV` |
| 源链接 | https://www.doubao.com/thread/xgeM04177ZmZm4AnV |
| 所属分类 | 智能驾驶核心技术 / BEV 感知 |

## 一句话摘要

BEV 感知按「显式深度 Lift-Splat（LSS）」与「BEV Query 注意力」两大主线分野；深度头应服务环视绝对尺度投影，不能简单套用单目自监督深度模型。

## 核心关键词

`LSS` · `BEVDet/BEVDepth` · `BEVFormer` · `BEVFusion` · `Depth bin` · `Monodepth2` · `时序 warp` · `Sparse4D/StreamPETR`

## 知识脉络（按路线分类）

### A. 整体 BEV 网络版图

| 路线 | 核心机制 | 代表 | 工程画像 |
|------|----------|------|----------|
| **LSS 显式深度** | 深度分布 → Lift 视锥 → Splat 到 BEV → CNN | LSS、BEVDet、BEVDet4D、BEVDepth、BEVNeXt | 量产友好、可解释、依赖深度质量 |
| **Query Transformer** | 可学习 BEV Query + Deformable Cross-Attn，隐式几何 | BEVFormer(V2)、PETR 系 | 时空建模强、算力高 |
| **稀疏实例 Query** | 无稠密 BEV 网格，实例级时序 | StreamPETR、Sparse4D | 显存友好、利 VLA/长时序；地图任务较弱 |
| **多模态 BEV 融合** | 相机 LSS + 点云 BEV 特征级融合等 | BEVFusion；TransFusion/CMT（实例/跨模态） | 高阶智驾常用 |
| **Mamba/SSM 时序** | 替时序 Attention | MambaBEV 等 | 偏研究前沿 |

**选型速记**

- 纯视觉量产低延迟 → LSS + BEVDet/BEVDepth  
- 精度/端到端大模型 → BEVFormer / Sparse4D / StreamPETR  
- Camera+LiDAR → BEVFusion  
- 显存紧、实例检测 → 稀疏 Query  

---

### B. 「深度模型」在 BEV 中的真正角色

#### B1. LSS DepthHead（工业主路径）

- 输出：**深度 bin 上的 softmax 分布**（非单值回归），服务 Lift-Splat。
- 监督：激光投影像素深度 GT + 与检测/分割联合训练（BEVDepth 范式）。
- 优势：环视统一米级尺度；深度服务于 3D 任务区域。
- 代价：依赖 lidar（或高质量伪深度）标签。

#### B2. 独立单目深度（勿直接当在线 DepthHead）

| 模型 | 监督 | 尺度 | 环视 LSS 适配 |
|------|------|------|----------------|
| Monodepth2 | 光度自监督 | 相对，模糊 | ❌ 多相机尺度不一致会投影错位；可作预训练 init |
| PackNet-SfM | 光度 ± velocity | 部分缓解 | ⚠️ 仍缺环视联合几何；偏学术自监督 BEV |
| DepthAnything(V2) | 大规模伪标签；有 metric 室外版 | 视版本 | ⚠️ 可离线伪标/预训练；非环视标定几何定制，远距不稳 |
| BEVDepth DepthHead | LiDAR GT + 任务联合 | 绝对、多相机对齐 | ✅ LSS 标配 |

**常见误区**

1. Metric DepthAnything ≠ 可直接替换 BEVDepth 头（缺环视外参一致性）。  
2. 光度损失可作辅助，**不能替代** lidar 监督做尺度锚。  
3. velocity loss **不能彻底**消灭单目尺度漂移。

---

### C. 多视角空间融合（单帧内 PV→BEV）

1. **Lift-Splat**：内外参 + 深度分布；误差主因是深度与标定。  
2. **Query Cross-Attn**：3D 参考点投影到图像再可变形采样；不显式深度，但对位姿/标定敏感。  
3. **体素中间表示**（如 CaDDN 类）：先 3D 体素再压到 BEV。  
4. **IPM**：平面假设，仅低速/泊车地面语义，不作高速 3D 主感知。

---

### D. 时序融合（跨帧）

前提：ego-centric BEV → **必须先做 ego-motion 补偿**。

| 范式 | 代表 | 注意 |
|------|------|------|
| 稠密 BEV warp + CNN | BEVDet4D、SOLOFusion | 动态目标拖影；显存随窗长线性增 |
| BEV Query TSA | BEVFormer | 算力随窗长升；位姿误差污染采样点 |
| 稀疏实例时序 | StreamPETR、Sparse4D | 依赖关联；漏检丢记忆 |
| Mamba 时序 | MambaBEV | 工程成熟度低 |

**工程红线**

- 刚性 warp 只补自车运动，不补物体运动 → 需门控/运动建模。  
- 训练用 GT 位姿、推理用噪 IMU → 域差掉点；需位姿扰动增强。  
- 缓存历史 BEV/Query，避免每帧重跑 Backbone+视图变换。  
- 窗长：短窗稳（常 2–3 帧）；长窗利静态/远距但漂移与拖影加重。

---

### E. Backbone 习惯（交叉）

图像侧常见 ResNet / ConvNeXt / Swin；检测头多承 DETR/Deformable DETR。详见 PV Backbone 笔记。

## 知识点校验

- **Correct**：LSS vs BEVFormer 的机制分野；BEVDepth 式 lidar 监督深度分布 vs Monodepth2 自监督相对深度——概念正确。
- **Caution**：原文对「理想/毫末/特斯拉用某方案」等车企归类多为二手归纳，**未经验证不宜当事实引用**。
- **Caution**：MambaBEV/MamBEV 等命名与成熟度随文献快速变化，需核对具体论文，勿当作量产选项。
- **Expand**：深度 bin 数、深度范围与远距 mAP 的消融；无 lidar 域的伪深度/多相机光度联合；LSS splat 算子在 NPU 上的实现差异。

## 延伸阅读建议

- Lift-Splat-Shoot；BEVDet / BEVDet4D / BEVDepth  
- BEVFormer / BEVFormerV2；BEVFusion  
- StreamPETR、Sparse4D；PETR 系列  
- Monodepth2；PackNet-SfM；Depth Anything V2（对照其**非** BEV 原生设定）

## 与其他主题的关联

- PV Backbone 与 View Transform：[`../notes/14-pv-backbone-view-transform.md`](../notes/14-pv-backbone-view-transform.md)
- BEV+PNC 总栈：[`../notes/02-bev-pnc-stack.md`](../notes/02-bev-pnc-stack.md)
- VLA 中 4D 感知接入：[`../notes/16-qwen3vl-vla-architecture.md`](../notes/16-qwen3vl-vla-architecture.md)
- Action 条件中的 BEV 特征：[`../notes/10-action-expert-dit.md`](../notes/10-action-expert-dit.md)
