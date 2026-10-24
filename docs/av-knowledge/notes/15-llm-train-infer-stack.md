# 大模型训练—推理全栈：Swift、Megatron 与推理后端

## 元信息

| 字段 | 内容 |
|------|------|
| 原对话标题 | 大模型推理后端 |
| Thread ID | `xw7zUWEkYkxGOoTJG` |
| 源链接 | https://www.doubao.com/thread/xw7zUWEkYkxGOoTJG |
| 所属分类 | 大模型技术 / 训练与推理工程 |

## 一句话摘要

主流工程流水线是「Megatron 类做大预训练 → Swift/TRL 做 SFT+DPO → vLLM/TRT-LLM 等引擎上线」；推理监控盯 TTFT/TPOT/KV，训练监控盯 MFU/通信/grad。

## 核心关键词

`ms-swift` · `Megatron` · `SFT` · `DPO` · `vLLM` · `PagedAttention` · `TensorRT-LLM` · `AWQ/GPTQ` · `TTFT/TPOT` · `MFU`

## 知识脉络

### 1. 框架定位

| 组件 | 定位 | 适合 |
|------|------|------|
| **Swift（ModelScope）** | HF/PEFT 封装的微调+推理一站式 | 7B–70B 级 SFT/DPO、快速验证、多模态 Qwen 等 |
| **Swift-Transformers** | 对 HF transformers 的补丁增强库 | 与 Swift 配套；**勿与原版 transformers 混环境** |
| **Megatron-LM / Megatron-DeepSpeed** | TP+PP+DP 三维并行预训练 | 70B+ / 万亿；非快速微调 |

选型：预训练 Megatron → 权重转 HF → Swift LoRA SFT → Swift DPO；PPO-RLHF 工业上多被 DPO 替代。

### 2. 训练链路

1. **预训练**：自回归 LM；超大用 Megatron。  
2. **SFT**：指令数据；全参或 LoRA/QLoRA；**只对 answer token 计 loss**。  
3. **RLHF（PPO）**：RM + Actor/Critic + KL 到 reference；不稳、贵。  
4. **DPO**：chosen/rejected 偏好对，无独立 RM、无 PPO；Swift 原生路径。

### 3. 推理引擎地图

**Prefill**（一次性吃 prompt）vs **Decode**（逐 token + KV cache）。

| 类别 | 引擎 | 要点 |
|------|------|------|
| 调试 | HF transformers | 兼容强、吞吐差 |
| 高吞吐服务 | **vLLM** | PagedAttention + continuous batching；Swift 主推 |
| 前缀共享 | SGLang | RadixAttention；Agent/RAG；Swift 常需独立起服务 |
| 极致 NVIDIA | TensorRT-LLM | 编译优化、FP8/FP4；部署重 |
| 国产/特定 | LMDeploy 等 | Qwen 等适配好 |
| 本地轻量 | llama.cpp / Ollama | GGUF；CPU/边缘 |
| 其他 | TGI、DeepSpeed-MII、ExLlamaV2、MLC-LLM… | 存量或专用场景 |

Swift `infer_backend` 常见：`hf` / `vllm` / `tensorrt-llm` / `llama.cpp`。

量化：训练侧 QLoRA ≠ 推理侧 AWQ/GPTQ/GGUF；格式与引擎绑定。

### 4. 推理监控（生产看板）

- **业务**：P50/P90/P99 **TTFT**、**TPOT**、E2E、请求/token 吞吐、排队、成功率/超时。  
- **引擎**：KV 占用率/驱逐、batch、pending、TP 通信。  
- **硬件**：GPU util（LLM 解码常非 100%）、显存（权重+KV）、带宽瓶颈。  
- **质量**：空回复、重复、截断、结构化失败率。  

诊断：TTFT 高→prefill；TPOT 高→decode/KV/带宽；排队涨且 util 低→显存/KV 不够。

### 5. 训练监控（对照）

- 收敛：train/eval loss、lr、grad_norm、NaN。  
- 吞吐：tokens/s、step time、**MFU**（≠ SM util）。  
- 并行：AllReduce、TP 通信、**PP bubble**、ZeRO allgather。  
- 数据：dataloader 饥饿、磁盘/CPU。  
- 显存：权重+梯度+**Adam 状态**+激活（checkpointing）。

注意：训练 tokens/s 与推理生成 tokens/s **定义不同**。

### 6. 常见坑

- Megatron↔HF 权重合并/分片转换。  
- Swift 与原生 transformers 版本冲突。  
- AWQ 权重不能当 FP16 普通加载。  
- 长上下文 KV 显存预估不足。  
- PPO 不稳 → 优先 DPO（通用建议，非绝对）。

## 知识点校验

- **Correct**：Prefill/Decode、PagedAttention 价值、训练 MFU 与推理 TTFT 指标体系的主干正确。
- **Caution**：原文「Swift 新版已弱化原生 RLHF、优先 DPO」依赖具体版本 changelog，集成前核对当前 ms-swift 文档。
- **Caution**：引擎「业界最流行/全部迁移 vLLM」是动态格局（SGLang、TRT-LLM、厂商引擎并行），不宜写成永恒结论。
- **Expand**：智驾车载侧（Orin/Thor 等）与云端 vLLM 栈的差异；多 LoRA / 多模态 VLA 推理的 KV 与视觉编码器调度；投机解码（EAGLE 等）在规控低延迟场景的适用性。

## 延伸阅读建议

- vLLM / PagedAttention 论文与文档  
- Megatron-LM；DeepSpeed ZeRO  
- DPO 论文；ms-swift 官方文档  
- TensorRT-LLM；SGLang RadixAttention  
- 训练 MFU 计算方法（PaLM/Megatron 技术报告中的约定）

## 与其他主题的关联

- 训练效率优化：[`../notes/04-training-efficiency.md`](../notes/04-training-efficiency.md)
- AdamW 等优化器：[`../notes/06-adamw-optimizer.md`](../notes/06-adamw-optimizer.md)
- 智驾规控 DPO：[`../notes/12-planning-dpo-rl.md`](../notes/12-planning-dpo-rl.md)
- Qwen3-VL VLA + Swift：[`../notes/16-qwen3vl-vla-architecture.md`](../notes/16-qwen3vl-vla-architecture.md)
- VLM 训练阶段：[`../notes/01-vlm-training-stages.md`](../notes/01-vlm-training-stages.md)
