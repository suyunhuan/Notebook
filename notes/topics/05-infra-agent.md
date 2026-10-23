# 主题：训练工程与 Agent 运行时

关联页：09, 10, 25, 26

---

## 1. 训练栈分界

| 规模 | 框架 | 关键选项 |
|------|------|----------|
| **&lt;7B** | ModelScope **SWIFT** + Transformers | LoRA rank=8, α=32；约 1% 参数；DDP/FSDP；fp32/bf16/fp16 |
| **7B–100B** | SWIFT + **Megatron** | TP/PP/DP/EP/CP；sequence parallel；packing；FlashAttention；recompute；通信优化；lazy tokenize |

多模态上限备忘：`max pixel 16M`；`max img tokens 16K`；`video 500`（页 25）。  
另一组实验：`image token 1150`；`video 128`；`max frames 16`；`max length 8192`；`lr=2e-6`；bf16；gradient checkpoint（页 26）。

---

## 2. LoRA Target 选择

常见递进：

1. `q+v`  
2. `q,k,v,o`  
3. `q,k,v,o + gate,up,down`（全线性近似）  

另含 aligner / merger（多模态投影）。辅助：**channel loss**、辅助 loss。

---

## 3. Megatron 5D 并行直觉

| 维 | 作用 | 何时加大 |
|----|------|----------|
| TP | 切矩阵 | 单层太大 |
| PP | 切层 | 层数深 |
| DP | 切数据 | 吞吐 |
| EP | 切专家 | MoE |
| CP | 切序列上下文 | 超长上下文 |

配合：micro vs global batch、activation recompute、packing 降 padding、lazy tokenize 加快启动。

---

## 4. 训练环封装（ModuleWrapper）

```
env 初始化 → 加载模型 → dataset/preprocess/dataloader
→ 训练(forward → loss → backward → param update)
→ 保存与评估
```

损失侧可含 **DiT loss**。单机蒸馏配置例：bf16、`num_workers=8`、数据不解压、按公式拼有效 batch、10 epochs、每 3k 存 ckpt；维护 special tokens（unused/ignore/unk/answer、long/short cat）。

---

## 5. RL 训练监控（页 24）

必看：`KL`、温度、`lr`、reward hacking、collapse。  
算法选项：GRPO / PPO / DPO。

---

## 6. Agent 运行时（页 09–10）

```
CLI / WebUI / Telegram / Discord / 飞书 / 微信
        → MessageBus (in-band / out-of-band)
        → AgentLoop（session、skills、memory、hooks、进度）
        → Context Builder（system prompt + 上下文）
        → AgentRunner（LLM Provider、工具循环、流式、终止）
        → Tools：文件 / shell / Web / MCP / Cron
        → Memory + Dream（长期记忆）
```

生命周期：`restore → compact → build → run → save → respond → done`。

与自动驾驶主线的关系：同一套「会话状态 + 工具循环」可服务于数据标注 Agent、评测 Agent、仿真调度；笔记中与 ViT 选型写在同一页，说明团队同时建设模型与工具链。

---

## 7. 检查清单

- [ ] &lt;7B 是否误开全量 Megatron（过重）  
- [ ] LoRA target 是否覆盖 MLP（gate/up/down）  
- [ ] 长视频 token 是否触发 OOM（先降 max frames）  
- [ ] Agent Loop 是否持久化 session 与 knowledge pack  
- [ ] RL 监控面板是否含 KL 与 length 分布  
