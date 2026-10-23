# 第 03 页：NaviNN / TrafficNN / Ego-HMF 串级规划

- **原图**：[`images/01a169b3-1892-7f8e-96e7-110135da29ef.jpg`](images/01a169b3-1892-7f8e-96e7-110135da29ef.jpg)
- **主题标签**：规划, 导航场, 红绿灯, 轨迹

## 纠错说明

- Ego-HMF：Hybrid Motion Forecasting / 类似自车运动预测模块，按项目定义
- EHP = Electronic Horizon Provider

## 校正转录

### NaviNN（静态导航场）
- 输入：路径边缘关系；BEV/PV 时序特征
- Encoder → Feat → Decoder(+ query points) → **导航场**

### TrafficNN（红绿灯）
- 输入：TBT/EHP 导航信息；红绿灯模组特征；BEV/PV 时序；并融合 NaviNN Feat
- → **路口通行概率**

### Ego-HMF
- 输入：感知隐层 / BEV·PV 时序 / 结构化；Traffic Feat + NaviNN Feat
- → **轨迹指导**

### 串级逻辑
全局导航 → 局部路径 → 红绿灯到局部 → 轨迹规划效用

### FSD 规划要素
轨迹预测、占据场、速度场、导航场、红绿灯通行概率；并引入 **world model**


## 摘要

导航场、通行概率、自车轨迹三级级联，汇总 FSD 规划场并指向 World Model。
