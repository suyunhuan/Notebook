# 大模型推理后端

Thread ID: xw7zUWEkYkxGOoTJG
URL: https://www.doubao.com/thread/xw7zUWEkYkxGOoTJG

---

大模型推理后端
2026 年 10 月 23 日•AI 生成可能有误 注意核实
介绍 swift 框架，介绍 swift transformers，megatron 等详细的使用策略；介绍大模型训练推理流程，训练包含 sft 和 rlhf 等，推理包含各种配置、推理引擎、推理性能、推理精度等各个方面
大模型训练推理全栈：Swift、Swift-Transformers、Megatron 详解
Swift（ModelScope Swift）是阿里魔搭推出的大模型训练 / 微调 / 推理框架；Swift-Transformers 是其配套的 transformers 增强库；Megatron（Megatron-LM）是英伟达开源的大规模分布式训练框架，主打万亿参数大模型预训练。
下文分为三块：框架介绍、大模型完整训练链路（预训练→SFT→RLHF）、推理全链路（引擎、精度、性能、配置），以及三者的使用策略对比。
一、框架介绍 & 使用策略
1. Swift（ModelScope Swift）
定位：面向开发者的一站式大模型微调 + 推理框架，基于 HuggingFace Transformers / PEFT / Accelerate 封装，主打低代码、快速上手、支持全参数 / PEFT 微调、多模态、一键部署。
核心能力
支持 LLM、多模态模型：Qwen、Llama、GLM、InternLM、LLaVA 等；
丰富微调方案：全参数微调、LoRA/QLoRA、AdaLoRA、IA3、DPO、KTO、ORPO（新版 Swift 已经弱化原生 RLHF，优先 DPO）；
分布式：DDP、DeepSpeed、ZeRO、模型并行；
推理：原生支持 vLLM、TensorRT-LLM、llama.cpp，一键切换推理后端；
数据集封装、metrics 评估、模型导出、上传魔搭 Hub。
Swift-Transformers
Swift 配套增强库，对 HF transformers 做补丁扩展：
增加更多模型适配、修复推理 bug；
增加滑动窗口、注意力优化、量化加速；
兼容 Swift 的训练参数、支持模型分片加载；
注意：不是独立新框架，是 Swift 依赖的增强版本 transformers，和原版 hf-transformers 不能混用环境。
Swift 使用策略（落地选型）
✅ 适合场景：
中小模型（7B~70B）SFT、DPO；
快速验证 idea、业务场景微调；
单人 / 小团队，不想手写大量分布式代码；
需要快速做模型评估、Demo 推理、上线测试。
❌ 不适合：
万亿级超大模型预训练（Megatron 更合适）；
极致底层算子深度定制。
典型使用流程
bash
# 安装
pip install ms-swift[all]
# LoRA微调 SFT
swift sft \
--model qwen/Qwen2-7B-Instruct \
--dataset my-custom.jsonl \
--lora_target_modules q_proj,v_proj \
--quantization_bit 4 \
--deepspeed zero2
# 推理
swift infer --model qwen/Qwen2-7B-Instruct --adapter ./output/lora
# DPO偏好对齐（替代RLHF首选）
swift dpo --model xxx --dataset preference.jsonl

2. Megatron-LM（Megatron）
定位：NVIDIA 面向超大模型预训练的原生分布式框架，从零实现 Transformer，不依赖 HF transformers。核心创新是张量并行、流水线并行，专门解决千亿 / 万亿参数模型单卡放不下的问题。
衍生分支：Megatron-LM、Megatron-DeepSpeed（微软 + 英伟达合并，DeepSpeed ZeRO + Megatron 并行）。
核心并行策略（Megatron 最关键）
张量并行 TP（Tensor Parallelism）：把单个 Transformer 层矩阵切分到多张 GPU，同层计算并行。TP 组内 GPU 通信量大，一般同一台机器内。
流水线并行 PP（Pipeline Parallelism）：把 Transformer 层切分，不同 GPU 负责不同层，batch 分成 micro batch 流水执行，降低显存占用。跨机通信压力更小。
数据并行 DP：传统 DDP，不同 GPU 持有完整模型副本，数据分片。
Megatron = TP + PP + DP 三维并行，是千亿大模型预训练标准方案。
✅ 适合场景：
从零预训练 70B+，数百 B、万亿大模型；
大规模集群（多机多卡 InfiniBand）；
需要极致优化 Transformer 底层算子。
❌ 缺点：
上手门槛极高，模型需要按照 Megatron 格式重新定义；
原生 SFT/RLHF 支持弱，数据集、tokenizer 需要大量适配；
不适合快速微调实验，开发周期长。
Megatron 使用策略
硬件前提：IB 高速网络，多机集群；
一般只用来做预训练阶段；预训练完成后导出权重，转 HF 格式，再用 Swift/trl 做 SFT、DPO；
权重转换坑很多：Megatron 的 TP 分片权重需要合并才能导出 HF；
现在工业界主流：预训练 Megatron-DeepSpeed；微调 SFT/DPO 用 Swift / TRL。
补充：Megatron 原生 RLHF 实现很少，早期 Megatron RLHF 代码维护差，现在几乎没人用 Megatron 做 RLHF，优先 DPO。
框架对比
表格
框架	定位	并行	适合阶段	模型规模	上手难度
Swift	一站式微调推理	DDP/ZeRO，TP 由 vLLM 接管	SFT、DPO、推理、评估	7B~70B 为主	低
Swift-Transformers	HF 增强补丁	无独立并行，配合 Swift	训练推理模型适配	所有 Swift 支持模型	中等
Megatron-LM	超大模型预训练	TP+PP+DP	预训练	70B~ 万亿	极高
二、大模型完整训练流程：预训练 → SFT → RLHF/DPO
1. 预训练（Pretrain）
目标：学习通用语言知识，原始 base 模型，没有对话能力。
数据：海量无标注文本；
损失：标准自回归 LM loss；
框架：超大模型用 Megatron-DeepSpeed；中小模型也可以用 Swift。
预训练完成得到 Base Model。
2. SFT 监督微调（Supervised Fine-tuning）
目标：让 base 模型学会遵循指令、对话格式，是对齐第一步。
数据：高质量指令样本（输入 + 标准答案）；
训练方式：
全参数微调：更新全部权重，显存开销巨大；
PEFT 微调（LoRA/QLoRA）：冻结主干，只训练少量低秩矩阵，显存大幅降低；Swift 最常用；
Loss：只计算回答部分 token 的交叉熵损失，prompt 部分 mask 不参与 loss；
产物：SFT 模型。
Swift 对 SFT 封装完善，支持 jsonl、sharegpt 格式数据集，自动 mask prompt，支持 deepspeed、4bit 量化微调。
3. RLHF 基于人类反馈的强化学习（传统三阶段）
⚠️ 当前工业界越来越多用 DPO 替代 RLHF，因为 RLHF 复杂、不稳定、算力开销大。Swift 新版优先 DPO。
RLHF 三个阶段：
SFT 阶段：上面已经完成，得到 SFT 模型；
RM 奖励模型训练：
同一个 prompt，收集多条模型回答，人工排序（好回答 vs 坏回答）；训练奖励模型，输入 prompt + 回答，输出标量奖励分数；
PPO 强化学习优化：
Actor：SFT 模型；
Critic：奖励模型；
Reference Model：SFT 模型，用来限制 KL 散度，防止模型崩坏；
目标：最大化奖励模型分数，同时约束和 SFT 模型输出差异（KL penalty）。
RLHF 痛点
PPO 训练不稳定，调参困难；
需要单独训练奖励模型，数据成本高；
训练循环要多次推理，算力开销大。
DPO（Direct Preference Optimization，RLHF 替代方案）
Swift 原生支持 DPO，不需要单独训练奖励模型，不需要 PPO。
原理：直接用偏好数据（同一个 prompt，chosen 回答，rejected 回答），优化损失，让 chosen 的概率高于 rejected。
✅ 优点：代码简单、训练稳定、算力开销接近 SFT，现在主流对齐方案。
总结训练链路：
【预训练 Megatron】→ 导出权重 →【SFT Swift LoRA】→【偏好对齐 DPO Swift】，这是现在最主流工程流水线。
三、大模型推理全链路：引擎、配置、精度、性能
推理目标：给定输入 token，自回归逐 token 生成输出。推理分为Prefill 阶段（输入 prompt 一次性编码） 和 Decode 阶段（逐 token 生成，KV 缓存是核心）。
1. 主流推理引擎
原生 HF transformers 推理
最简单，但是性能差；KV 缓存实现效率低，不支持连续批处理；仅用于调试。Swift infer 默认后端之一。
vLLM
当前最流行开源推理引擎，核心：PagedAttention 分页 KV 缓存，把 KV 缓存当成虚拟内存管理，极大提升连续批处理吞吐量；支持 TP 张量并行；支持 AWQ/GPTQ 量化；Swift 可以一键切换--infer_backend vllm。
TensorRT-LLM
NVIDIA 官方推理引擎，底层算子极致优化，FP16/INT4 量化，支持 TP/PP；延迟更低，适合 NVIDIA 显卡大规模部署；Swift 支持 TRT-LLM 后端。
llama.cpp
CPU / 低显存 GPU 推理，GGUF 量化格式，适合本地小模型；Swift 支持。
Text Generation Inference(TGI)：HF 官方服务化推理框架。
Swift 推理策略：
开发调试：hf；上线高吞吐：vLLM；低延迟 NVIDIA 集群：TRT-LLM；本地小模型：llama.cpp。
2. 推理精度方案（量化）
推理量化本质：降低权重 / 激活 bit 位，减少显存占用、提升速度，换取少量精度损失。
FP16 / BF16：原始精度，效果最好，显存占用大；A100/H100 推荐 BF16。
INT8：权重 INT8，激活 FP16；损失很小，显存减半。
AWQ / GPTQ（权重 INT4）：当前最主流 GPU 量化，4bit 权重，激活 FP16；7B 模型仅需 4~6G 显存，精度下降可控；vLLM 原生支持 AWQ/GPTQ。
GGUF Q4_K_M、Q5_K_M：llama.cpp 量化格式，CPU 推理。
注意区分：训练时量化（QLoRA 4bit 训练） vs 推理量化（AWQ/GPTQ）
QLoRA 是训练阶段量化主干模型；AWQ/GPTQ 是训练完成后离线量化权重，用于推理加速。
3. 关键推理配置项
plaintext
max_new_tokens：最大生成长度
temperature：随机性，越小越确定（0=greedy）
top_p / top_k：采样截断
repetition_penalty：抑制重复输出
kv_cache：开启/关闭（关闭速度极慢）
tensor_parallel_size：vLLM TP切分，模型跨多卡
gpu_memory_utilization：vLLM显存占用上限（0~1）

4. 推理性能指标
首 token 延迟（Prefill 延迟）：输入 prompt 之后，第一个 token 出来的时间；长 prompt 瓶颈；
Token 吞吐（tokens/s）：每秒生成 token 数量；Decode 阶段指标；
并发能力：同时处理多少用户请求；vLLM PagedAttention 核心优势就是高并发；
显存占用：权重显存 + KV 缓存显存。
长文本场景：KV 缓存会随序列长度线性增长，是显存最大开销。
5. 推理部署模式
单机单卡：7B 模型 AWQ；
单机多卡 TP：70B 模型，TP=4；
多机 TP+PP：超大模型；
API 服务：vLLM/TGI 起 OpenAI 兼容 API，业务调用。
四、工程落地完整流水线总结
预训练：千亿级大模型使用 Megatron-DeepSpeed，多机 IB 集群；训练完成，权重合并转 HF 格式；
SFT：使用 Swift，LoRA/QLoRA 做指令微调，验证对话效果；
偏好对齐：Swift DPO（优先），替代传统 RLHF；
模型评估：Swift 内置评估脚本，评测 MMLU、GSM8K 等；
推理准备：模型 AWQ/GPTQ 量化；
上线推理：Swift 调用 vLLM 后端，启动 OpenAI 兼容 API 服务。
五、常见坑点
Megatron 权重转 HF 格式容易出错，TP 分片合并是大坑；
Swift 和原生 transformers 版本冲突，不要混用环境；
RLHF PPO 不稳定，优先 DPO；
vLLM 推理必须匹配模型版本，AWQ 量化模型不能直接用普通 FP16 权重加载；
推理 KV 缓存占用，长文本场景显存预估一定要预留 KV 空间。
如果你需要，我可以进一步：
给一份可直接跑的 Swift SFT + DPO 完整脚本；
Megatron 预训练启动脚本，并行参数配置样例；
vLLM 推理性能压测脚本；
详细对比 RLHF PPO 和 DPO 数学公式。
推理后端都有哪些
搜索 3 个关键词，参考 18 篇资料
大模型推理后端（推理引擎）全列表 + Swift 支持情况
Swift 内置可切换的 infer_backend：hf、vllm、tensorrt-llm、llama.cpp；下面把业界主流全部归类说明。
一、PyTorch 原生调试类（原型、调试，不适合高并发上线）
1. HuggingFace Transformers（hf）
原理：原生 PyTorch 实现，基础 KV Cache，无连续批
优点：兼容性最强，任何 HF 模型直接跑，调试首选
缺点：显存碎片化严重、吞吐很低，长文本并发差
Swift：--infer_backend hf，默认调试后端
2. DeepSpeed-MII（Microsoft）
基于 DeepSpeed，复用训练的并行能力，支持 TP/PP
适合：已经用 DeepSpeed 做训练的模型，统一训练推理栈
缺点：社区活跃度不如 vLLM，新模型适配慢
二、GPU 高吞吐在线服务（工业生产最常用）
3. vLLM（伯克利）⭐Swift 主推上线后端
核心：PagedAttention 分页 KV 缓存 + Continuous Batching 连续批
支持：TP/PP、AWQ/GPTQ/FP8、多 LoRA、MoE、Speculative Decoding (EAGLE)，AMD / 国产卡生态好
优势：开箱即用、模型支持极广、社区最活跃，OpenAI 兼容 API
Swift：--infer_backend vllm，生产首选
4. SGLang（LMSYS）
核心：RadixAttention（前缀树 KV 缓存共享），大量共享 prompt 场景（RAG、Agent、多轮对话）优势巨大
强项：结构化输出 (JSON 约束)、前缀缓存、speculative decoding
适合：Agent、RAG、大量共用 system prompt 业务
Swift 目前没有原生 SGLang 后端，需要独立起 SGLang 服务
5. TensorRT-LLM（NVIDIA TRT-LLM）⭐英伟达极致性能
核心：编译式引擎，CUDA 底层算子融合、FP4/FP8 硬件原生加速
优点：单卡延迟最低、原始算力利用率最高；TP/PP/EP 专家并行
缺点：仅 NVIDIA GPU，需要提前编译模型，部署成本高，新增模型适配工作量大
Swift：--infer_backend tensorrt-llm
6. TGI Text Generation Inference（HuggingFace 官方）
HF 官方服务引擎，早期工业标准
特性：流式输出、张量并行、AWQ/GPTQ；但没有 PagedAttention
现状：现在大部分新项目迁移 vLLM，仅存量业务在用
7. LMDeploy（商汤）
国产推理引擎，对 Qwen、InternLM 等国产模型适配好，轻量化，支持 AWQ、FP8
适合国产模型快速部署，国内社区友好
三、本地 / CPU / 边缘轻量推理
8. llama.cpp
C/C++ 纯原生实现，支持 CPU、Mac Metal、低显存 GPU，GGUF 量化（Q4_K_M 等）
特点：不依赖 PyTorch，适合本地单机小模型、笔记本 / 边缘
Swift：--infer_backend llama.cpp
9. Ollama
上层封装，底层引擎就是 llama.cpp，简化 GGUF 模型管理，一键起 API；不属于底层引擎，是应用层工具。Swift 不直接对接 ollama 后端。
10. CTranslate2
C++ 推理库，CPU/GPU 轻量化，适合离线批量、低资源场景，早期翻译模型常用。
11. ExLlamaV2
专门针对 GPTQ INT4 量化模型的高速推理引擎，7B/13B GPTQ 模型速度很强，适合单卡量化本地推理。
12. MLC-LLM
TVM 编译栈，面向移动端 / 手机端（安卓 / IOS），把模型编译到手机 NPU，端侧推理。
四、其他企业 / 硬件专用推理后端
Triton Inference Server（NVIDIA）：通用推理服务框架，可以挂载 TRT-LLM、vLLM 作为后端，做负载均衡、多模型管理
MindSpore Lite、CANN Ascend：昇腾 NPU 专用推理引擎
ONNX Runtime / ORT：微软通用推理，支持 CPU/GPU，适合轻量化模型，LLM 大模型性能一般
TensorRT：基础 TensorRT，非 TRT-LLM，原生不支持 KV 缓存，很少直接跑 LLM
快速选型对照表
表格
引擎	核心技术	最佳场景	Swift 原生支持
HF transformers	原生 KV Cache	调试、验证	✅
vLLM	PagedAttention + 连续批	通用线上高吞吐，首选	✅
SGLang	RadixAttention 前缀缓存	RAG、Agent、大量共享 prompt	❌
TRT-LLM	编译算子融合 FP8/FP4	NVIDIA 集群极致低延迟	✅
llama.cpp	GGUF 量化	CPU、Mac 本地小模型	✅
TGI	HF 原生批处理	存量 HF 业务	❌
DeepSpeed-MII	DeepSpeed 并行	DeepSpeed 训练推理一体化	❌
ExLlamaV2	GPTQ 专用内核	单卡 GPTQ 量化本地推理	❌
Swift 推理后端使用策略（承接上一轮 Swift 介绍）
开发、调参、SFT 后效果验证：hf
业务上线，通用高并发、7B~70B：vLLM（推荐）
H100/A100 集群，追求极致延迟、超大模型：TRT-LLM
本地笔记本、无高端 GPU 跑小模型：llama.cpp
Agent/RAG 大量共用 system prompt：单独部署 SGLang，Swift 调用 HTTP API
补充关键概念区分
推理后端（engine）：负责模型计算、KV 缓存、算子；
服务网关（Triton、RayServe）：负载均衡、多实例管理，不是推理后端；
量化格式和引擎绑定：GGUF → llama.cpp；AWQ/GPTQ → vLLM/TRT-LLM；FP8 → vLLM/TRT-LLM。
要不要我整理一份 Swift 切换不同推理后端的完整命令样例（vLLM / TRT-LLM /llama.cpp）？
性能监控都需要关注哪些
大模型推理性能监控指标（分 4 大类：业务指标、推理性能指标、资源指标、模型质量指标）
适用于 vLLM / TRT-LLM / SGLang / LMDeploy 等所有推理引擎，同时适配 Swift 启动的推理服务。分「面向业务」、「引擎内部」、「硬件资源」、「异常 & 模型效果」，同时标注线上核心重点指标。
一、业务侧指标（最优先，用户视角）
反映真实用户体验，是 SLA 核心
首 token 延迟 (TTFT, Time To First Token)
用户输入 prompt → 返回第一个 token 耗时。
瓶颈：prefill 阶段，长 prompt、大 batch、KV 内存压力会显著拉高 TTFT
重点区分：P50/P90/P99，不要只看均值；P99 是线上告警基准
token 生成延迟 / 每 token 耗时 (Time Per Output Token, TPOT)
decode 阶段，相邻两个输出 token 之间间隔；决定流式输出顺滑度。
总请求耗时 (E2E Latency)
从请求进入到最后一个 token 返回的总耗时。
请求吞吐（Request Throughput）
每秒完成多少个完整请求。
Token 吞吐（Token Throughput）【引擎核心指标】
每秒处理的 token 总数，分为：
prefill token/s（输入 token 处理速度）
decode token/s（输出生成 token 速度）
并发数 / 在线活跃请求数 (Active Requests)
当前同时在跑的请求数量。vLLM/SGLang 靠连续批，并发上限由显存决定。
请求成功率 & 错误码
4xx（参数错误、prompt 超长）、5xx（OOM、引擎崩溃、超时）；超时率是重点。
排队等待时长（Queue Latency）
请求到达后，排队等待调度的时间。排队变长 = 服务过载，需要扩容。
告警重点：P99 TTFT 突增、排队延迟飙升、5xx 错误率上升。
二、推理引擎内部指标（引擎层面，定位瓶颈）
vLLM / SGLang / TRT-LLM 都暴露这类 metrics（prometheus）
KV 缓存指标（重中之重，LLM 推理最大瓶颈）
KV 缓存占用率：已使用 KV 缓存 / 总 KV 显存
KV 缓存命中率（SGLang RadixAttention 独有，RAG 场景非常关键）
KV 缓存分配失败次数（vLLM 分页缺页、evict 驱逐次数；驱逐变多会严重拉高延迟）
序列等待驱逐数量：为腾出显存，把老请求 KV 踢掉，会导致用户请求中断 / 重跑
批处理指标
当前 batch 大小、平均 batch size
连续批：调度迭代次数、每轮调度的新 prefill 数量 + decode 请求数量
等待队列长度（pending requests）
采样 & 生成统计
平均输入 token 长度（prompt 长度分布）
平均输出生成长度
采样参数分布（temperature/top_p 等，用于定位随机输出异常）
模型并行相关指标（TP/PP 多卡）
卡间通信耗时（NCCL 通信时间占比）
通信等待时间：GPU 空等数据，说明并行负载不均衡
量化相关
量化 kernel 耗时、权重加载耗时（冷启动加载模型耗时）
三、硬件资源监控（底层资源，判断瓶颈在 GPU / 内存 / 网络）
GPU（最核心）
GPU 利用率（nvidia-smi）
注意：LLM 推理不追求 100% 利用率！
过高（>95% 持续）：计算饱和，延迟会上涨
过低：显存瓶颈，GPU 闲着，被 KV 缓存限制并发
GPU 显存占用
权重显存（固定）
KV 缓存显存（动态变化，随并发和序列长度增长）
显存碎片、剩余空闲显存；OOM 前一般是显存持续打满
GPU 温度、功耗、SM 时钟、显存带宽
显存带宽：大模型推理很多时候瓶颈不是计算，是显存带宽，尤其是 decode 阶段
NCCL 通信指标（多机 / 多卡 TP/PP）
IB 带宽、通信丢包、NCCL 错误，多机集群重点监控
主机侧
CPU 利用率：prefill 预处理、tokenizer、调度线程、http 网关
CPU 内存（系统内存）：模型加载、swap（swap 一旦使用，性能雪崩，必须告警）
网卡带宽：多机分布式、网关转发、日志指标上报
磁盘 IO：模型权重加载、离线缓存；冷启动关注，线上推理一般不碰盘
四、模型质量 & 业务效果监控（容易被忽略）
性能好≠模型输出没问题
输出异常统计：
空回复、截断、重复循环（repetition）、乱码、特殊 token 比例
模型输出评估指标（抽样）
拒绝率：安全拒绝、幻觉比例
业务侧人工 / 自动评测（如匹配业务答案、JSON 格式通过率，结构化输出场景）
向量 / 采样分布：logprobs 分布，异常突变代表模型 / 量化出问题
五、资源限制 & 阈值类指标
请求超时数量
最大序列长度超限
单请求 token 上限超限
模型热加载 / 动态 LoRA 加载耗时、加载失败次数（Swift + vLLM 动态 LoRA 场景重点）
六、瓶颈定位判断思路（快速诊断）
TTFT 高，TPOT 正常 → Prefill 瓶颈：prompt 太长、batch 过大、GPU 算力 / 带宽不足
TTFT 正常，TPOT 很高 → Decode 瓶颈，大概率：KV 缓存紧张、显存带宽瓶颈
排队队列持续上涨，GPU 利用率不高 → 显存瓶颈（KV 缓存不够）
排队上涨，GPU 利用率接近 100% → 计算瓶颈，算力不足
延迟突然抖动、毛刺多 → KV 频繁 evict 驱逐，或者 NCCL 通信不稳定
七、监控系统选型
指标采集：vLLM prometheus exporter、TRT-LLM metrics、dcgm（英伟达 GPU 指标）
存储可视化：Prometheus + Grafana
日志：Loki / ELK，记录失败请求、输入输出样例
链路追踪：OpenTelemetry，记录单请求 TTFT、prefill/decode 分段耗时
八、极简核心指标清单（生产看板必放）
做 dashboard 优先放这一组
P50/P90/P99 TTFT
TPOT
总 token 吞吐 (prefill/decode 拆分)
活跃并发请求数 + pending 排队数
KV 缓存使用率、驱逐次数
GPU 利用率、GPU 显存占用
请求成功率、超时率、5xx 错误率
如果你想，我可以：
给一份 Grafana 看板指标清单 + PromQL 查询语句（vLLM）；
或者给一套线上告警阈值推荐（比如 KV 缓存 > 85% 告警，P99 TTFT>2s 告警等）。
训练性能监控指标呢
大模型训练性能监控指标（预训练 / SFT / DPO / PPO 通用，Megatron / Swift / DeepSpeed）
训练监控和推理差异极大：训练是前向 + 反向 + 梯度更新，算力密集，瓶颈常出在算力、NCCL 通信、梯度 / 权重显存、数据加载；分为业务 / 训练进度指标、训练吞吐指标、分布式通信指标、GPU 硬件指标、损失 & 收敛指标、异常指标。
适用：Megatron-LM、Megatron-DeepSpeed、Swift（Deepspeed ZeRO / DDP）、TRL。
一、训练进度 & 收敛指标（算法视角，看模型有没有学好）
训练 Loss
训练集 loss（每 step），区分 SFT 的 label-only loss（prompt 部分 mask 不计入 loss）
验证集 eval loss（定期 eval，不能只看 train loss，防止过拟合）
loss 抖动、loss 爆炸、loss 不下降是核心告警项
RLHF/DPO：DPO 看 DPO loss；PPO 要单独看 actor loss、critic loss、KL 散度、奖励值 reward。
学习率 lr：当前 step 学习率，监控 warmup、cosine 衰减是否正常，lr 跳变是 bug。
全局 step /epoch、已处理 token 总量（总训练 token 数，预训练最重要指标）
验证集评测指标：MMLU、GSM8K 等，每若干 step 跑一次；看模型能力提升。
样本统计：平均序列长度、被 mask 的 prompt token 占比（SFT 场景）；
梯度相关
梯度范数 grad_norm：梯度爆炸时会飙升；梯度消失时趋近 0
梯度裁剪触发次数：如果频繁触发 clip，代表训练不稳定，需要调参
权重 / 梯度溢出：NaN/Inf loss、NaN 梯度（一旦出现直接告警，通常是精度、lr、数值溢出）
二、训练吞吐指标（性能核心，衡量训练有多快）
训练吞吐标准单位：tokens /sec（整个集群每秒处理 token）
全局 token 吞吐：tokens_per_second
集群总吞吐；可以拆：每个 GPU 单卡 token 吞吐 tokens/sec/gpu，用来判断负载均衡
预训练场景，单卡 token 吞吐是集群选型、算力评估核心。
step 耗时（iteration time）
一次完整迭代（前向 + 反向 + 梯度更新）的耗时。
拆分子阶段耗时（很多框架可埋点）：
数据加载耗时（dataloader）
前向传播 forward time
反向传播 backward time
梯度 allreduce / 参数更新 step time
样本吞吐：samples/sec（较少用，序列长度变化大时不如 token 吞吐）
MFU（Model FLOPs Utilization，模型 FLOPs 利用率，⭐训练最核心性能指标）
区别于 GPU SM 利用率！GPU 利用率看硬件，MFU 看Transformer 模型理论浮点算力有没有被打满
MFU = 模型理论 FLOPs / 硬件实际 FLOPs
预训练理想：70%~90%；低于 60% 一般存在瓶颈（通信慢、数据拖后腿、kernel 差）
SFT/LoRA 微调 MFU 天然低于预训练，LoRA 一般 30%~60% 属于正常区间
样本利用率：重复样本、坏样本过滤比例，数据侧指标
小提示：
推理看 token/s 是生成 token；训练 token/s 是输入训练样本的 token，定义完全不一样，不要混为一谈。
三、分布式并行指标（Megatron TP+PP+DP / DeepSpeed ZeRO 重点）
1. 数据并行 DDP
AllReduce 通信耗时、allreduce 占 step 总耗时比例
梯度同步等待时间：GPU 计算完梯度，等其他卡梯度的等待时间；等待高 = 负载不均衡
2. Megatron 张量并行 TP
TP 组内 NCCL 通信耗时；TP 越大，单步通信开销越高
TP 组内负载不均衡：不同 rank 计算耗时差异
3. Megatron 流水线并行 PP
流水线气泡时间（Pipeline Bubble）⭐PP 最重要指标
流水线切层带来的 GPU 空闲时间；PP 层数切分不合理气泡率飙升，MFU 暴跌
micro-batch 调度等待时间
4. DeepSpeed ZeRO（Swift 常用 ZeRO1/2/3）
ZeRO 通信开销：梯度 / 参数分片的 allgather、reduce-scatter 耗时
参数预取（prefetch）耗时、参数释放延迟
ZeRO-3：参数分片通信会显著增加通信量，IB 网络质量直接决定性能上限
通用分布式指标
NCCL 通信带宽、NCCL 错误、超时、重试次数
rank 之间 step 耗时方差：各 GPU 迭代时间差异大 = 负载不均衡（数据不均匀、层负载差异）
四、GPU 硬件资源指标
GPU SM 利用率（nvidia-smi / DCGM）
训练和推理不一样，训练是计算密集，理想可以跑到 85~100%；
利用率高但 MFU 低：算子低效、大量空闲计算周期
利用率低：被通信 / 数据加载卡住，GPU 在等数据
GPU 显存占用（训练显存压力远大于推理）
显存构成：权重 + 梯度 + 优化器状态（Adam 占大头！Adam 优化器会占用权重 2 倍显存）+ 激活值 + 临时 buffer
激活重计算（activation checkpointing）开启前后显存变化
显存碎片、内存分配失败、OOM（致命告警）
显存带宽、GPU 功耗、温度、时钟频率、ECC 错误（H100/A100 重点）
GPU 内存拷贝（H2D / D2H）：主机↔显存拷贝耗时，数据加载瓶颈时会很高
五、CPU & 主机侧指标（容易忽略的瓶颈：数据加载）
CPU 利用率：dataloader、tokenizer、数据预处理、日志、指标上报线程
CPU 内存：数据集加载、缓存；一旦用到 swap，训练直接断崖降速，swap 必须告警
磁盘 IO：读取训练数据集（jsonl/parquet/webdataset）；磁盘慢会造成 GPU 空等，叫data starvation（数据饥饿）
网卡带宽：多机集群 NCCL 通信，InfiniBand 带宽监控，网卡丢包
典型场景：GPU 利用率很低，但 CPU / 磁盘打满 → 数据加载瓶颈
六、数据加载（Dataloader）专项指标
队列等待时间：GPU 等待下一个 batch 的耗时（data starvation）
预取队列长度、样本预处理耗时
数据加载错误、坏样本数量、tokenizer 失败样本
大预训练 Megatron 一般用 webdataset；Swift SFT 多用 HF datasets，多进程 dataloader 容易出瓶颈。
七、训练特有的异常指标 & 告警项
NaN / Inf loss、NaN 梯度
梯度爆炸（grad_norm 突增）
OOM、CUDA out of memory
NCCL 通信报错、rank 掉线、进程崩溃
step 耗时突然大幅上涨 / 抖动（可能显存碎片、网络波动）
验证 loss 突然飙升（过拟合、坏数据）
检查点保存 / 加载耗时、ckpt 保存失败（Megatron 分片权重保存尤其容易出问题）
激活重计算开关带来的耗时变化
八、不同训练任务的指标侧重点
预训练（Megatron）
重点：MFU、总 token 吞吐、气泡率、NCCL 通信、总训练 token、train/eval loss；显存、IB 网络。
SFT（Swift LoRA/QLoRA）
LoRA 显存占用低，一般不会 PP；重点：token 吞吐、grad_norm、eval loss、dataloader 速度。QLoRA 还要监控 4bit 权重加载、量化算子耗时。
DPO（Swift）
监控 DPO loss、chosen/rejected 的 log 概率差、eval loss；不需要 critic 模型，指标比 PPO 简单很多。
RLHF PPO
额外监控：reward 均值、KL 散度、actor loss、critic loss、ppo clip 损失；KL 散度过大代表模型偏离 SFT 基座。
九、快速瓶颈诊断逻辑
MFU 低 + GPU 利用率低 → 通信瓶颈（NCCL）或者数据饥饿（dataloader 慢）
MFU 低 + GPU 利用率很高 → 算子低效、大量小计算，FLOPs 没利用起来
step 耗时逐步变长 → 显存碎片、序列变长、或者 ZeRO-3 通信累积开销
多机 rank 之间 step 耗时差距很大 → 负载不均衡
loss 震荡大，grad_norm 很大 → 训练不稳定，需要降 lr、加梯度裁剪
保存 ckpt 耗时越来越长 → 磁盘 IO 瓶颈
十、生产训练看板核心指标（最简必选）
集群总 token/s，单卡 token/s
MFU
step 迭代耗时 & 各阶段拆分（dataloader/forward/backward/update）
train loss / eval loss，grad_norm
GPU 利用率、GPU 显存占用
NCCL 通信耗时占比
数据加载等待时间
NaN/Inf 计数、OOM、rank 崩溃错误数
补充：采集工具
Megatron、Megatron-DeepSpeed：原生输出训练日志，可接入 wandb/tensorboard；可导出 prometheus metrics
Swift：支持 wandb、tensorboard，每 step 输出 loss、token 吞吐
GPU 硬件指标：DCGM、nvidia-smi、nvdcgm-exporter
分布式通信：NCCL profiler、nsys 系统级性能剖析
性能剖析：Nsight Systems、PyTorch Profiler，定位 forward/backward/ 通信耗时
如果你需要，我可以：
给 Wandb 监控面板推荐指标分组；
或者一套训练告警阈值参考；
或者区分：预训练 vs SFT vs DPO 各自推荐的监控侧重点。
垂域大模型训练的时候是否只训练新增的 special token 词表呢，是的话怎么实现
垂域大模型：新增 special token，不是只训练新增 token 的词表，下面分原理、边界、工程实现（Swift+HuggingFace）、坑点完整讲清楚。
先给核心结论：
词表本身是静态数组，不能 “训练词表”；训练优化的是新增 token 对应的 embedding 向量（输入 embedding + lm_head 输出权重）；词表只是字符串→token id 映射，属于配置，不参与梯度更新。
默认方案：冻结基座主干，只初始化 + 训练新增 special token 对应的 embedding（LoRA / 全参数微调都可以这么做）；
❌ 误区：不是只训新增 token，如果开全参数 SFT，主干模型权重也会更新；只有手动冻结主干，才只更新新增 token 的 embedding。
special token 和普通新增领域 token 逻辑一致，只是用途不同（分隔符、特殊标记如<|doc|>、<|cite|>等）。
一、底层原理：Tokenizer、Embedding、LM Head 三者关系
LLM 结构：
Tokenizer + vocab 词表：字符串 ↔ token id 映射，纯查表，无参数，不可训练。新增 special token 就是往 vocab 字典里插入新 token 字符串，分配新 id。
Input Embedding（输入嵌入层）：[vocab_size, hidden_dim]，把 token id 映射成 hidden 向量，可训练参数。
LM Head（输出头）：一般和输入 embedding 权重共享（tie_weight），[hidden_dim, vocab_size]，最后一步预测下一个 token logits，可训练参数。
新增 special token 操作：
第一步：tokenizer.add_special_tokens([...])，词表长度变大（vocab_size += N，N 是新增 token 数量）
第二步：resize 模型 embedding 层和 lm_head，扩充矩阵，给新增的 N 个 id 分配向量。
新增向量初始化常用：用基座现有 embedding 均值初始化（效果最稳）；也可以随机初始化（容易不稳定）。
问题来了：扩充后的 emb 矩阵，哪些参数参与训练？
方案 A（常用垂域 SFT）：冻结基座全部主干 Transformer，只训练新增的 N 个 embedding 向量 + 新增 lm_head 输出向量。✅ 这就是你问的场景：主干不动，仅学习新 token 表征。
方案 B：全参数 SFT：emb 全部参数 + transformer 全部参数一起更新。新增 token 的 emb 会学，旧 token embedding 也会被改动，基座通用能力容易遗忘。
方案 C：LoRA 微调：主干冻结，LoRA 低秩矩阵训练；新增 token embedding 默认还是要单独放开训练，LoRA 不覆盖 embedding 层。
⚠️ 关键点：LoRA 不会自动训练新增 token embedding。LoRA 只作用在 q/k/v/o 等 attention/MLP 矩阵；embedding 层不在 LoRA target_modules 里。所以加 special token + LoRA 微调时，必须手动放开新增 embedding 参数参与训练。
二、两种策略对比（垂域场景选型）
策略 1：只训练新增 token 对应的 embedding（主干全部冻结）
✅ 适用：基座很强，垂域只需要引入特殊分隔标记，不想破坏基座原有知识；比如 RAG 引用标记<cite> </cite>、文档块标记<doc>。
Transformer 主干全部requires_grad=False
只有新增 N 个 token 对应的 embedding 权重行、lm_head 对应的列打开 requires_grad=True
优点：算力极低，几乎不会遗忘基座能力；只学习新标记语义
缺点：新增 token 表征能力有限；如果垂域需要大量新词汇，效果弱于 SFT+LoRA
策略 2：LoRA 微调主干 + 同时训练新增 token embedding（垂域最常用！）
✅ 适用：垂域不仅要 special 分隔符，还要让模型学习领域知识（医疗 / 法律 / 工业）。
Transformer 主干冻结，只训练 LoRA 矩阵（q_proj,v_proj 等）
同时，新增 special token 对应的 embedding 向量打开梯度，一起训练
这是工业垂域大模型标准做法：LoRA 学领域知识；新增 emb 学特殊标记。
极少场景：全参数 SFT，所有参数一起更新（算力大，容易灾难性遗忘，7B 以上垂域很少用）
三、代码实现（HF 原生 + Swift 两种方式）
3.1 HuggingFace 原生实现逻辑
python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 1. 添加special token
new_tokens = ["<|doc|>", "<|cite|>", "<|end_doc|>"]
num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
print(f"新增token数量：{num_new_tokens}, 新词表大小：{len(tokenizer)}")

# 2. resize embedding & lm_head，扩充权重矩阵
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# 3. 冻结主干，只放开新增token的embedding（重点！）
# 先全部冻结
for param in model.parameters():
    param.requires_grad = False

# embedding层
emb = model.get_input_embeddings()
# 只打开新增token对应的行的梯度
emb.weight.requires_grad = True
# 构造mask：只让新增id参与梯度
old_vocab_size = model.config.vocab_size - num_new_tokens
# 对前 old_vocab_size 的权重关闭梯度，新增部分打开
# 实现方式：hook（推荐），或者分开优化器分组

# lm_head同理，如果tie_word_embeddings=False，也要处理lm_head
lm_head = model.get_output_embeddings()
if lm_head is not None:
    lm_head.weight.requires_grad = True

# 4. 优化器分组：只对新增embedding参数更新
# 用torch hook实现：旧vocab部分embedding梯度置零，只有新增token保留梯度

原生 HF 直接写参数 mask 比较麻烦，工程上更推荐用 Optimizer 分组：把新增 embedding 参数单独抽出来交给优化器，其余参数不进优化器。
3.2 MS-Swift 中添加 special token 并只训练新增 emb（垂域常用）
Swift 封装了--additional_special_tokens，但默认不会自动只冻结主干只训新 emb，需要手动设置：
bash
swift sft \
--model qwen/Qwen2-7B-Instruct \
--additional_special_tokens '<|doc|>,<|cite|>' \
--dataset domain_data.jsonl \
# 开启LoRA微调主干
--lora_target_modules q_proj,v_proj \
# 关键：冻结基座主干，仅LoRA参数 + 新增embedding参与训练
--freeze_vocab False \
--trainable_parameters 'embedding,lm_head' \
--deepspeed zero2

Swift 参数说明：
additional_special_tokens：自动加到 tokenizer，自动 resize embedding
trainable_parameters 用来指定可训练参数，设置embedding,lm_head，配合冻结主干，实现只训练新增 token embedding
如果同时想要 LoRA + 新增 emb 一起训练：LoRA 参数 + embedding/lm_head 一起加入可训练参数列表
⚠️ Swift 坑：当使用 LoRA 时，默认 embedding 层是冻结的！新增 special token 必须显式开启 embedding 训练。
四、初始化策略（新增 emb 向量怎么初始化，非常影响收敛）
3 种方案，推荐均值初始化
均值初始化（推荐）：取原有全部词表 embedding 向量求均值，赋值给新增 token 向量。新 token 一开始有基础语义，训练收敛快，稳定。HF resize_token_embeddings 默认就是这个策略。
随机初始化：高斯随机向量，新 token 无任何先验，训练慢，容易出现输出异常、token 不生效。
复制某个现有 token 向量：复制一个已有 token，适合标记和原有 token 语义接近场景，很少用。
五、训练、推理阶段的配套注意事项
训练阶段
数据集里必须正确插入新增 special token，tokenizer 要能识别<|doc|>，不能被拆成子 token；
损失计算：SFT loss mask 规则不变，special token 照常参与 loss 计算；
优化器：新增 emb 可以单独设置学习率，一般高于 LoRA 学习率（比如 LoRA lr=1e-4，emb lr=5e-4）；新 token 表征需要更快学习。
验证：训练前后要验证：tokenizer("<|doc|>") 输出单个 token id，不能被切分。
推理阶段（重点，很多人踩坑）
推理时 tokenizer 必须用【训练时保存的 tokenizer】，词表包含新增 special token；
模型权重：
如果是只训 embedding：保存完整模型（emb 层更新了）；
如果 LoRA+emb 一起训练：LoRA 适配器 + 更新后的 embedding 层，推理加载时，需要合并 emb 变更；vLLM/TRT-LLM 都支持扩展词表模型；
不要用原始基座 tokenizer 推理，原始词表不认识新增 token，会自动拆分成子 token，完全失效。
六、常见误区澄清
❌ “训练词表”：词表是字符串映射，不可训练，只能扩充，训练的是 embedding 向量
❌ add_special_tokens 之后自动会训练新 token：不会！主干默认冻结，emb 层默认冻结，必须手动打开梯度
❌ LoRA 会自动学习新增 special token：不会，LoRA 只作用 transformer 矩阵，emb 不在 LoRA 目标模块，必须单独开启 embedding 训练
❌ 新增 token 越多越好：每增加一个 special token，emb 矩阵多一行，显存增加；大量新增 token 容易稀释表征，垂域一般控制在几个～20 个以内
七、什么时候只用新增 emb，什么时候上 LoRA？
仅需要分隔标记、文档标记、引用标记，不需要学习领域知识：主干冻结，只训练新增 token embedding，低成本，不丢基座能力。
不仅要 special 标记，还要模型学习垂域知识（医疗术语、合同语法、工业参数）：LoRA 微调 + 同步训练新增 special token embedding，垂域标准方案。
八、可选进阶方案：冻结旧 token emb，仅更新新增 token 的梯度实现
PyTorch 实现方式：register_hook，对原有 vocab 的 embedding 梯度置零，只保留新增 token 梯度，这样优化器只会更新新增行。适合不想改动原有任何旧 token embedding。
python
# hook函数，把旧vocab部分梯度置0
def zero_old_emb_grad(grad):
    grad[:old_vocab_size, :] = 0
    return grad

emb.weight.register_hook(zero_old_emb_grad)

如果你需要，我可以提供：
一份完整可直接运行的 HF 脚本：新增 special token，冻结主干，仅训练新增 token embedding；
或者 Swift 完整启动命令 + 数据集样例（带<|doc|>标记的 sharegpt 数据）。
为什么经常有训推不一致的问题，实际的坑到底在哪
训推不一致（Train-Inference Gap）根源与实战坑点
一句话概括：训练阶段和推理阶段的模型、tokenizer、数据预处理、采样逻辑、算子、并行策略、量化、特殊 token 处理，只要任意一处配置不统一，就会出现训推不一致。
典型现象：训练集上 loss 很低、eval 效果很好；上线推理输出乱码、特殊标记失效、长文本截断、重复循环、格式不遵守、幻觉暴增、PPL 飙升。垂域新增 special token、LoRA/QLoRA、vLLM/TRT-LLM 场景是重灾区。
下面分类拆解底层原因 + 真实工程坑（结合前面聊的 special token、Swift、vLLM、Megatron）
一、Tokenizer & 词表 / 特殊 Token 不一致（垂域新增 special token 最高发）
这是垂域模型 top1 坑，就是咱们上一节聊的新增<|doc|>、<|cite|>这类标记。
训练时 tokenizer 新增了 additional_special_tokens，推理用原始基座 tokenizer
训练：<|doc|>是单个独立 token id；
推理：原始词表不认，把<|doc|>拆成多个子 token <'<','|','d','o','c','|','>'>；
现象：模型不识别分隔符，忽略文档边界，引用标记失效，输出乱格式。
坑点：保存模型时只存 LoRA 权重，忘记同步保存 / 分发更新后的 tokenizer。LoRA 本身不携带 tokenizer 和词表信息！
special token id 不匹配
训练时eos_token_id/pad_token_id是 A，推理配置里写死成 B；训练和推理 eos/pad/sep id 不一致，会提前停止生成或者不停止。
tokenizer 参数不一致
truncation、padding、model_max_length两边配置不同；训练固定 max_seq_len=4096，推理设置成 2048，直接截断 prompt。
空格 / 换行、空白符处理差异：HF tokenizer、vLLM 内置 tokenizer 版本不一致，对空白字符编码结果不一样，输入 token 序列都变了。
校验手段：训练和推理侧对同一段字符串，打印tokenizer.encode()的 token id 列表，必须完全一致。
二、模型结构 & 权重加载不一致（LoRA、新增 Embedding 重灾区）
LoRA + 新增 embedding：推理只加载 LoRA，没有加载更新后的 embedding
前面讲过：新增 special token 时，我们更新了基础模型的 embedding 矩阵。
训练：基座 emb 已 resize 并更新了新 token 向量；
推理：只加载原始基座权重 + LoRA 适配器，emb 还是旧的、没有扩容；
后果：新 token id 越界，直接报错；或者加载了旧 emb，新 token 向量是随机值，标记完全失效。
很多人误以为 LoRA 包包含 emb 变更：LoRA 不存储基础 embedding 层的权重！
两种解决：
方案 1：训练后合并 emb 变更到基座（全量保存），推理加载完整模型；
方案 2：vLLM 支持embedding_layer补丁，推理时加载额外 embedding 权重。
训练用的并行切分格式 ≠ 推理引擎权重格式
Megatron 预训练输出 TP 分片权重，训练直接用分片；推理直接喂分片权重给 vLLM，没有合并、转换；权重错位。
DeepSpeed ZeRO 保存的分片 checkpoint，推理直接加载分片，没有合并成 HF 格式。
模型配置 config 不一致
训练 config 里面vocab_size、tie_word_embeddings、max_position_embeddings、rope scaling，推理用旧 config。
最常见：RoPE 缩放（长上下文版本），训练开启 rope scaling，推理没开，长文本直接崩。
三、数据预处理、Prompt 模板不一致（业务最容易踩的坑）
哪怕模型权重完全一样，prompt 模板不一样，输出就完全不一样。
训练 SFT 数据集用了固定对话模板，推理没套相同模板
Swift SFT 会自动套用chat_template；如果推理自己手写 prompt 拼接，少了一个换行、缺少系统标记、eos 标记缺失，就是训推 gap。
例子：训练："<|system|>xxx<|user|>xxx<|assistant|>"；推理直接输入用户问题，没有拼接完整模板。模型不知道进入对话模式。
loss mask 逻辑≠推理输入逻辑
SFT 训练只对回答部分计算 loss，prompt 部分 mask；推理没有 mask，只是完整输入 prompt。
衍生坑：训练时会在数据预处理阶段过滤某些 token、截断策略；推理截断策略不一样。
空格、换行、特殊转义：数据预处理阶段对\n、\t做清洗，推理侧没有清洗，输入文本不一样。
✅ 标准校验：拿 SFT 样本里的一条输入，分别走训练侧预处理代码 和 推理侧 prompt 组装代码，输出字符串完全一致，再 encode 对比 token id。
四、训练 vs 推理的计算逻辑差异（数学层面的 gap）
激活重计算（gradient checkpointing）
训练开启activation_checkpointing节省显存；推理没有这个机制。一般不影响输出，但部分旧版本模型会有数值微小漂移；多轮累积会放大。
精度不一致：训练 BF16，推理 FP16 / INT4 AWQ/GPTQ 量化
训练：BF16 混合精度；
推理：直接 AWQ INT4 量化；
量化会带来权重数值误差，模型行为偏移，长文本、领域特殊 token 尤为明显。
坑：只在 FP16 做效果验证，直接上 4bit 上线，没有量化后的评测。
Dropout
训练开启 dropout；推理时 dropout 自动关闭。一般基座 LLM dropout 很小，影响不大；但自定义垂域头、额外层时，会出现分布偏移。
五、采样 / 生成参数不一致（很多人忽略）
训练阶段是 teacher forcing，没有采样，是输入前 N 个 token 预测下一个；
推理是自回归采样逐 token 生成，temperature、top_p、repetition_penalty、max_new_tokens、stop_words/eos任何不一致都会导致结果差异。
训练 SFT 用的是确定的标准答案；推理开 temperature>0 带来随机性；
训练数据集里的停止符，推理 stop token 没有配置；模型不会停止生成，无限输出。
重要区分：这不是模型能力变弱，是生成策略不一样，但业务上感知就是训推不一致。
六、推理引擎算子 / KV 缓存 / Attn 实现差异（HF vs vLLM/TRT-LLM）
同一套权重，HF transformers、vLLM、TRT-LLM 之间本身就存在微小数值差异。
Attention 实现差异：原生 HF 的 SDPA、vLLM PagedAttention、TRT-LLM fused attention，底层 CUDA 算子不同，浮点微小误差，长序列下会逐步累积。
KV 缓存实现
训练是无 KV 缓存的完整前向；推理使用 KV 缓存增量解码。理论数学等价，但不同算子带来浮点漂移；极端长文本、MoE 模型更明显。
Speculative decoding、前缀缓存（SGLang RadixAttention）
开启推测解码，推理结果会和原生 greedy 不一样，属于引擎能力带来的 gap。
RoPE、ALiBi 算子实现版本差异，不同引擎对旋转位置编码计算略有差别。
工程经验：短文本差异很小；4k 以上长文本，浮点误差会累积放大，很容易肉眼看到输出差异。
七、并行带来的数值漂移（多卡 TP/PP）
训练使用 TP+PP+DP；推理用不同 TP 切分数量。
例如：训练 TP=4，推理 TP=2。NCCL 跨卡通信的浮点规约顺序不同，产生微小数值误差。
浮点数加法不满足结合律，不同卡规约顺序，logits 会有微小差别，采样后输出 token 变化。
八、环境、版本依赖不一致（隐性大坑）
transformers 版本、torch 版本、cuda 版本，训练和推理环境不一致；
Swift 依赖 swift-transformers 补丁，推理环境没有安装同版本 swift-transformers，模型代码行为不同；
FlashAttention 版本不一致。
现象：本地复现没问题，上线容器效果变差，环境漂移。
快速排查流程（出现训推不一致时，按顺序定位）
固定同一条测试样本，分别打印训练预处理后完整 prompt 字符串，推理组装后的 prompt 字符串 → 比对文本是否完全一致
把两个字符串送入 tokenizer，输出 token id 列表，逐位对比，看哪里不一样（大概率 special token）
使用同一个推理后端（优先 HF transformers，排除 vLLM 算子差异），加载完全相同权重，关闭所有随机（temperature=0 greedy）对比输出
这一步是黄金对照：HF 作为基准。
如果 HF 基准和训练 eval 结果一致，但 vLLM 结果不一样 → 引擎算子 / 量化问题；
如果 HF 基准就和训练 eval 不一致 → tokenizer/prompt/ 权重 / 配置问题。
关闭量化，FP16/BF16 原生精度验证，排除量化引入的 gap
检查 eos、pad、新增 special token id、rope、vocab_size 等 config 参数
检查 embedding 是否完整加载（新增 token 场景必查）
工程最佳实践，规避训推不一致
tokenizer 随模型一起保存，推理必须加载训练时保存的 tokenizer，不要复用基座原始 tokenizer；
新增 special token+LoRA 场景：训练结束，做一次全量权重导出（含更新后的 embedding），作为推理基准版本；LoRA 只做增量实验；
统一 chat_template：训练和推理使用完全相同tokenizer.chat_template，禁止手写 prompt 拼接；
训推使用相同版本 transformers、swift-transformers、torch、cuda；容器镜像统一；
上线前必须做基准对照实验：greedy 采样，HF 原生 transformers 作为基准，对比 vLLM/TRT-LLM 输出；
长文本场景：单独做长序列一致性验证；
量化模型：量化后单独做评测，不要直接假设量化后效果不变。
高频踩坑总结（一句话记）
新增 special token 场景：最坑的是只导出 LoRA，不带更新后的 embedding 和 tokenizer
prompt 模板：训练自动套 chat_template，推理忘记套，是业务最常见
算子 / 引擎：vLLM 和 HF 有浮点漂移，长文本会放大
精度：训练 BF16，推理直接上 INT4，跳过量化评测
并行：训练 TP 数和推理 TP 不一致，规约顺序带来数值漂移
如果你想，我可以写一段训推一致性校验脚本：输入同一个文本，分别返回 token id、HF greedy 输出，用来快速定位是 tokenizer 问题还是引擎问题。
和豆包继续聊