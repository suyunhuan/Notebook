# 逐页转录索引（28 页）

原图目录：[`images/`](images/)

| 页 | 文件 | 主题分类 | 一句话 |
|----|------|----------|--------|
| 01 | [page-01](page-01.md) | VLA / 原生多模态 | mindVLA、OneVL、MoT、统一表征与评测闭环 |
| 02 | [page-02](page-02.md) | 训练路线 | LLM→VL stages 并行 4D Encoder→混训→SFT→RL |
| 03 | [page-03](page-03.md) | 规划串级 | NaviNN / TrafficNN / Ego-HMF 级联 |
| 04 | [page-04](page-04.md) | 感知大图 | Cam+RL→BEV，Lane/GOD/Driveable/Sparse |
| 05 | [page-05](page-05.md) | BEV 检测 | Anchor、heatmap、Conv Head、损失加权 |
| 06 | [page-06](page-06.md) | Ego 推演 | TTC、ARDP 纵向、粒度 Attention |
| 07 | [page-07](page-07.md) | 训练技巧 | Teacher/Self forcing、KV 压缩、FM→DiT |
| 08 | [page-08](page-08.md) | SFT/VQA | 路线问答、traj reward、P1/P2 多样性 |
| 09 | [page-09](page-09.md) | Agent 运行时 | MessageBus → Loop → Runner |
| 10 | [page-10](page-10.md) | ViT + Agent | DINO/DA3/FastViT/Qwen；Agent 生命周期 |
| 11 | [page-11](page-11.md) | Reward | Spatial PE、CoC 质量、轨迹一致 |
| 12 | [page-12](page-12.md) | 4D Scene Graph | VLM 打标、快慢 rollout、SFT→DPO |
| 13 | [page-13](page-13.md) | 流式 4D | 2D/4D adaptor、SSM、KV Cache、多任务头 |
| 14 | [page-14](page-14.md) | 具身 FM | VLM+WM、Memory Bank、dense depth |
| 15 | [page-15](page-15.md) | Video Thinking | 历史 latent→未来帧+action、Wan 2.2 |
| 16 | [page-16](page-16.md) | Latent Reasoning | 4D→MOT→Action；AR LWM rollout |
| 17 | [page-17](page-17.md) | DAgger/4D | 时空 Attn、metric depth、S0–S3 |
| 18 | [page-18](page-18.md) | E2E Token | Fisheye/Cam/Navi/Ego→BEV/PV/Action |
| 19 | [page-19](page-19.md) | 蒸馏技能树 | OPD/MOPD、KL、special token、单机配置 |
| 20 | [page-20](page-20.md) | GRPO/Toggle | 组相对策略优化、token 预算交替 |
| 21 | [page-21](page-21.md) | Drive CoT | ThinkDrive、FSDrive、STL、扩散规划 |
| 22 | [page-22](page-22.md) | 型号规格 | Max/Ultra/Ultimate FM+DiT 参数表 |
| 23 | [page-23](page-23.md) | 损失权重 | llm/FM2LLM/AZ/AR/PP、grad loss |
| 24 | [page-24](page-24.md) | AZ+DiT | GRPO 监控、Cross-Attn、waypoints |
| 25 | [page-25](page-25.md) | 训练栈 | Swift LoRA &lt;7B；Megatron 5D 并行 |
| 26 | [page-26](page-26.md) | 训练流水线 | swift/megatron/vllm、ModuleWrapper |
| 27 | [page-27](page-27.md) | 多传感器 FM | 环鱼/前视/Lidar/Navi→FM→Diffusion |
| 28 | [page-28](page-28.md) | 策略总览 | ViT/4D→LLM、Hz 目标、数据与训练策略 |

## 主题 → 页码速查

- **感知 / 4D / 流式**：04, 05, 13, 16, 17, 18, 27, 28  
- **规划 / Action / Diffusion**：03, 06, 16, 18, 21, 23, 24, 27  
- **基模 / MLLM / WM / VLA**：01, 07, 12, 14, 15, 22, 28  
- **训练 / RL / 蒸馏**：02, 08, 11, 12, 17, 19, 20, 21  
- **工程 / Agent**：09, 10, 25, 26  
