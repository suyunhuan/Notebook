# 第 10 页：ViT 族谱 + Agent 生命周期

- **原图**：[`images/01a169b3-1d99-7aac-9b77-748baea63a7d.jpg`](images/01a169b3-1d99-7aac-9b77-748baea63a7d.jpg)
- **主题标签**：ViT, Agent

## 纠错说明

- DiNOv2 → DINOv2
- InBand/OutBand → in-band / out-of-band

## 校正转录

### ViT
- DINO：无监督单视角；DA3：相对位置、多视角 3D
- FastViT：CNN+ViT、基于 DINOv2；Qwen ViT：多维旋转 PE
### Agent
- 多端 → MessageBus → Loop/Runner → Tools（文件/shell/Web/MCP/Cron）
- Memory + Dream；生命周期 restore→compact→build→run→save→respond→done


## 摘要

对比视觉骨干差异，并画出完整 Agent 会话生命周期。
