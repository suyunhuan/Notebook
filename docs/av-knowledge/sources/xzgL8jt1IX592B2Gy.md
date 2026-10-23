# 在线蒸馏（OPD）体系与智驾应用

Thread ID: xzgL8jt1IX592B2Gy
URL: https://www.doubao.com/thread/xzgL8jt1IX592B2Gy

---

GENERAL BACKGROUND
Motivation
 mitigate exposure bias and distribution mismatch（imitation learning）, e.g., student not expressive enough to mimic the teacher's distribution
 self-evolving
 dense token-level supervision（rewards）
 learn from external teachers or previous better selves.
usage
 especially useful for smaller models
 MOPD(Xiaomi MIMO V2): distilling multiple expert models into the same one, Xiaomi MIMO V2
 Strong to Weak Distillation (Qwen3 ): first offline distillation for knowledge acquisition, then online distillation. qwen3 VL
 GLM-5：swiftly recover the skills acquired in earlier SFT and RL multiple learning stages (Reasoning RL and General RL)
 training prompts: sampled from the corresponding teachers' RL training sets and mixed in appropriate proportions.
 Deepseek V4
 full vocab OPD instead of token-level KL estimate, to reduce grad var
 caching only the last-layer teacher hidden states, and compute the full-vocab logits on the fly.
 more than ten teacher models（experts obtained from sft and RL） covering various domains are employed to distill a single student model.
 Black-box on-policy distillation（e.g., from GPT5）: frames the student LLM as a generator and trains a discriminator to distinguish its responses from the teacher LLMs, creating a minimax game
a minimax game.
Different Post-training Objectives
Methods divergence Supervision Advantage disadvantage samples
SFT NLL GT sequences train-inference distribution mismatch
sequence-level KD NLL teacher sequences
supervised KD forward KL teacher logits overestimate void regions of teacher model
Rejection Sampling NLL student sequences with correct labels  on-policy
 reinforce positive samples，improve top@1? positive
on-policy imitation learning NLL student sequences, with expert labels (iteratively)  on-policy
 popular in robotics and deep RL positive(GT, expert)
on-policy distillation reverse KL divergence teacher logits（teacher-forcing） of student sequences
（do not backpropagate through the student’s sampling distribution）  evolve during training
 minimize train-inference distribution mismatch
 prioritize the tokens where the teacher assigns high probability, which can avoid low-quality generations
at the expense of less diverse generations positive and negative
Generalized KD NLL
forward KL (on/off)
reverse KL  teacher-generated or GT
 student sequences  evolve during training
 minimize train-inference distribution mismatch best divergence can be task-dependent positive and negative
On-policy Objectives
 Rejection Sampling: using only student sequences with positive labels. only positive data（GT）
 on-policy imitation learning：on-policy imitation approaches (e.g. Ross et al., 2011) iteratively collect sequences using the student policy, obtain expert labels for those sequences, and then retrain the student on this dataset.
 on-poilicy distillation：token-level feedback from the teacher logits
 evolve during training
 minimize train-inference distribution mismatch
图片 1 combination of forward and backward KL
图片 2 combination of off-policy sft/distillation and on-policy distillaiton
图片 3 unified view of RL（global reward） and on-policy distillation（token-wise reward）
图片 4 Application of MOPD in MIMO-V2
Choices of KL（derivation）:
 forward KL: mean-seeking, mode-cover，学习到老师模型平滑化的分布。
 reverse KL: mode-seeking, 聚类到老师模型的 modes（老师模型集中化的分布）。
D_KL（Q||P）
 P: teacher； Q: student,
 foward KL: D_KL (P||Q)
 reverse KL: D_KL (Q||P)
 The optimal mixture of for forward and reverse KD can vary across tasks.
Application to Autonomous Driving
☐ 自己更好的就用 GT samples 来 SFT 吧：例如智驾数据
☐ 别人更好的 (e.g., Mimo V2) 或者自己历史阶段更好的（e.g., DS v3, GLM5）就用 OPD 吧：
☐ 通识和指令遵从数据：蒸馏通用模型例如 Qwen3 235B
☐ 智驾不如端到端的 case：蒸馏端到端模型 VA
☐ 智驾原子能力、通用和难例决策能力：也可以先训练一个更大的模型当老师
☐ 此法可以配合大蒸小循环嵌套，举一反三
☐ TBD: 再仔细思考一下 forward KD: mean-seeking 和 backward KD: mode-seeking, less similar 应该怎么配合使用。
Extentions
“如果早知道这个报错 / XX 信息，你觉得刚才那个 token 选得怎么样？”
5.1 On-policy self-distillation
做法： prompt 中加入 hint 之后的学生初始化成了一个新老师：
 q+hint+on-policy a -> (self-）teacher logits。Intuition: a sufficiently capable LLM can rationalize external privileged reasoning traces and teach its weaker self (i.e., the version without access to privileged information)
 原始的 hint 主要是 error feed，包括
 hint 类型：
 verified cot: OPSD
 智驾领域可以考虑白盒 / 指令 COT（如果效果更好的话）
 error feedback: SDPO
 环境输出：例如代码解释器返回的运行时错误）、未通过的测试用例信息等。
 参考答案：如果在同一个采样批次中，有其他采样路径成功解决了问题，那么这个成功的答案也可以作为失败路径的 “反馈” 提供给教师。
 partial answer。有点类似指令 cot->reasoning cot，从给定指令到自己 think 出指令
 whatever hint
 注意：这也可以用来构造 DEBUG 数据。会更加模型亲和。
 注意：hint 也可以用于 RL 的种子数据
 注意：这个和指令 cot->reasoning cot（VLR->A 变成 VL->RA），指令控车 ->meta action 控车可以联合使用（VLA->T, VL->AT） @薛文元 @孙雅静 @刘洁锐
 如果 VLA->T 很好，那很可能我们只需要保证模型可以出很准的 A， T 自然也就好了。A<->T 质检的 consistency 可能天然就很高。
 这个可以用 sft 课程学习来实现，也可以结合 hint-based opsd 来实习哈～
 VLA->T 这个阶段在哪里？
5.2 单轮 -> 多轮， 多轮 -> 单轮
 单轮扩展成多轮推理，把中间轮当做 “助教”。
 心智模型：引导相当于把模型 “roll-in” 到一片更容易拿到奖励的中间区域 。先在这片区域学会 “怎么继续就能拿到奖励”，再慢慢学会 “从题目开头怎么走到这片区域”。这样难题就从 “既要发现好区域、又要学会怎么解” 变成 “先学会怎么解，再学会怎么走到那里”。
 迁移的关键不是让模型记住那段前缀，而是让推理过程产生重叠。因为 long-cot 的模型经常在生成中出现自检、回头改写、反思、回溯。POPE 的观点是：这些行为会让 “带引导的 rollout” 访问到很多更接近原题起点的内部状态，从而扩大与 “无引导 rollout” 可达状态的交集，迁移就更容易发生。
 依赖的能力项：instuction-following、Self-verification、backtracking behaviors
 智驾应用：多层级 COC reasoning、多层级导航指令。先学会主因 -> 决策，再学会主因的主因 -> 主因 -> 决策。慢慢扩展到 long horizon reasoning
多轮蒸馏回单轮推理: RLTF
 self-distillation
 feedback modeling
5.3 如何让模型在那些目前 “完全不会做” 的难题上获得有效的探索信号？（TBD）
 限制 entropy
 优化 clip 参数（减少 clip）: incease the high clip ratio, \epsilon_high
 直接优化 pass@k（衡量 exploration）
 简单 + 难问题混合训练：优化过程会过度专注于那些已经能解决的简单问题（以此快速通过刷分来降低 Loss），从而抑制了对困难问题的探索和改进。类似 curriculum learning
 专家数据 sft/off policy distill：高质量推理轨迹很贵，规模化困难；而且经验上蒸馏常常会 “封顶” 后续 RL 的收益，off-policy 训练还可能让 RL 不稳定。
5.4 怎么用 on-policy distillation 做准确率之外的其他 reward。例如安全性、舒适性等。
就是 on-policy 得到的结果上面可以再加一个 reward 的 loss。类似端到端的 loss-weight。
Reference
 ON-POLICY DISTILLATION OF LANGUAGE MODELS: LEARNING FROM SELF-GENERATED MISTAKES
 MiniLLM: Knowledge Distillation of Large Language Models
 KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning
 Unlocking On-Policy Distillation for Any Model Family
 Qwen3 Technical Report
 Qwen3-VL Technical Report
 MiMo-V2-Flash Technical Report
 Black-Box On-Policy Distillation of Large Language Models
 GKD — swift 3.12.0.dev0 documentation
 Recipe: Async On-Policy Knowledge Distillation Trainer — verl documentation
 GLM-5:
 https://zhuanlan.zhihu.com/p/2004506304065065334?share_code=1mybKRL35Wr9e&utm_psn=2009413232092274701
 SPDO: Reinforcement Learning via Self-Distillation
 OPSD: On-Policy Self-Distillation for Large Language Models
 POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration
笔记整理：On-Policy Distillation（OPD）体系 + 智驾 VLA 落地思路
核心主线：On-Policy Distillation（在线蒸馏），解决模仿学习里经典的曝光偏差 (exposure bias)、分布不匹配 (distribution mismatch)；区别于离线 SFT / 离线 KD，OPD 用 student 自己采样出来的序列，拿 teacher logits 做 token 级监督，实现模型自演化。
1. 通用背景 General Background
动机 Motivation
缓解模仿学习的曝光偏差、训练推理分布偏移：小模型（student）表达能力不足，很难拟合 teacher 的数据分布
支持自演化 self-evolving
稠密的 token 粒度监督信号（逐 token reward）
学习来源：外部专家 teacher，或是模型自身历史上更强的旧版本（过去更好的自己）
典型落地案例
MOPD（小米 MiMo V2）：多专家蒸馏，把多个专家模型压缩进同一个 student
Qwen3 / Qwen3-VL 强弱蒸馏 Strong to Weak：先离线蒸馏获取基础知识，再做在线蒸馏迭代
GLM-5：恢复多阶段 SFT+RL 学到的能力（推理 RL + 通用 RL）；训练 prompt 采样自 teacher 的 RL 数据集，按合适比例混合
DeepSeek V4
采用全词表 OPD，替代 token 级 KL 估计，降低梯度方差
只缓存 teacher 最后一层隐状态，实时计算全词表 logits
十余个来自 SFT/RL 阶段、覆盖多领域的专家 teacher，蒸馏单个 student
黑盒在线蒸馏（例如 GPT5 作为 teacher）：student 作为生成器，训练判别器区分 student 与 teacher 输出，构造极小极大博弈（minimax game）
适用场景：对小模型增益尤其明显
2. 后训练目标对比 Different Post-training Objectives
表格
方法	散度	监督来源	优点	缺点	样本
SFT	NLL	真实 GT 序列		训练 - 推理分布不匹配	
sequence-level KD	NLL	teacher 输出序列			
supervised KD	forward KL	teacher logits		高估 teacher 分布空白区域	
Rejection Sampling	NLL	student 采样序列，保留带正确标签样本	✅ 在线策略 on-policy
✅ 强化正样本，提升 top@1		正样本
on-policy imitation learning	NLL	student 迭代采样序列 + 专家标注	✅ on-policy
✅ 机器人 / 深度 RL 领域常用		正样本（GT / 专家）
on-policy distillation	reverse KL	student 采样序列，teacher logits（teacher-forcing，不回传 student 采样分布梯度）	✅ 训练过程持续演化
✅ 缓解 train-inference 分布偏移
✅ 优先学习 teacher 高概率 token，规避低质量生成	生成多样性下降	正 + 负样本
Generalized KD	NLL
forward KL（在线 / 离线）
reverse KL	teacher 生成样本 / GT
student 采样样本	✅ 训练演化
✅ 缓解分布偏移	最优散度和任务强相关	正 + 负样本
On-policy 目标三类细分
Rejection Sampling：仅保留 student 采样出来的正标签样本，只用 GT 正数据
on-policy imitation learning：迭代用当前 student 策略采样轨迹，请专家打标签，再训练 student（经典 Ross 2011）
on-policy distillation：teacher 提供 token 级 logits 反馈；训练迭代演化、减少训练推理分布偏移
配套图示概念
前向 KL + 反向 KL 混合
离线 SFT/KD + on-policy 蒸馏混合范式
RL（全局 reward）与 OPD（token-wise 逐 token reward）统一视图
MiMo-V2 中 MOPD 应用架构
3. KL 散度选择与推导 Choices of KL
约定：P=teacher 分布；Q=student 分布
Forward KL：
D
KL
	​

(P∣∣Q)
 前向 KL，均值匹配 mean-seeking：覆盖 teacher 全部模态 mode-cover；学习 teacher 平滑分布。会去拟合 teacher 分布所有区域，包括概率很低的空白区域。
Reverse KL：
D
KL
	​

(Q∣∣P)
 反向 KL，模态匹配 mode-seeking：让 student 集中拟合 teacher 高概率模态；聚类到 teacher 的核心 mode。不会强行去学习 teacher 低概率区域。
✅ 前向 / 反向 KD 最优混合比例，随任务变化。TBD：思考两者如何搭配使用，尤其分布相似度低的场景。
4. 自动驾驶领域应用 Application to Autonomous Driving
简单决策规则：
如果 GT 本身质量足够好：优先用 GT 样本做 SFT（智驾原生数据）
如果有更强外部模型（MiMo V2）/ 自身历史更强版本（DS v3、GLM5）：使用 OPD
通识、指令遵从：蒸馏通用大模型（如 Qwen3 235B）
智驾端到端能力弱于专家端到端 VA 模型：蒸馏 VA 端到端模型
智驾原子能力、通用决策、难例：训练更大模型作为 teacher
可嵌套「大模型蒸馏小模型」循环，反复迭代
待思考：forward KD（mean-seeking）与 reverse KD（mode-seeking）如何组合
5. 扩展方向 Extensions
核心思想：Hint 引导，问句附加提示信息，构造特权信息，生成更强 self-teacher。
一句话范式：q + hint + on-policy action → self-teacher logits
直觉：能力足够强的大模型，可以基于特权推理轨迹做解释，反过来教没有特权信息的弱版本。hint 本质是特权信息。
5.1 On-policy self-distillation（OPSD）在线自蒸馏
Hint 类型：
Verified CoT：OPSD；智驾可尝试白盒 / 指令式 CoT
Error feedback：SDPO；错误反馈
环境输出：代码解释器报错、失败测试用例等
同批次成功样本答案：同一次采样里成功路径，作为失败路径的反馈
Partial answer 部分答案：类似指令 CoT → reasoning CoT；VLR→A 转为 VL→RA
任意 hint
落地价值：
构建更适配模型的 DEBUG 数据集
hint 可作为 RL 种子数据
和智驾 VLA 链路联合：VLA→T、VL→AT（@薛文元 @孙雅静 @刘洁锐）
如果 VLA→T 效果好，只需要保证动作 A 足够准确，T 自然成立；A 与 T 一致性质检天然高
实现方式：SFT 课程学习 /hint-based OPSD
待确认：VLA→T 阶段放在整个链路哪个位置
5.2 单轮 ↔ 多轮蒸馏转换
单轮 → 多轮推理
把中间轮次当作助教。
心智模型：roll-in 引导，把模型带到更容易拿到奖励的中间状态；先学会 “从中间状态怎么拿到奖励”，再学习 “从初始状态走到中间状态”，拆分难题。
POPE 观点：长 CoT 模型自带自检、回溯改写；带 hint rollout 能访问更多靠近原始输入的内部状态，扩大与无 hint rollout 状态交集，提升迁移能力。
依赖能力：指令跟随、自校验、回溯能力
智驾落地：多层级 COC 推理、多层导航指令；课程式学习：主因→决策，再学习主因的主因→主因→决策，扩展长时序推理。
多轮 → 单轮推理（RLTF）
自蒸馏 + feedback modeling，把多轮推理能力压缩回单轮模型。
5.3 难题探索信号：如何让模型攻克完全不会的样例（TBD）
现有候选方案
限制熵 entropy
调整 clip 参数：降低 clip 强度，调高
ϵ
high
	​

，允许更大更新幅度
直接优化 pass@k 指标，度量探索能力
难易样本混合训练 / 课程学习：单纯训练容易样本会快速压低 loss，但抑制对难例探索
专家数据 SFT /off-policy 蒸馏：高质量推理轨迹成本高；蒸馏容易封顶后续 RL 收益，离线训练会造成 RL 不稳定
5.4 OPD 拓展非准确率类 Reward：安全性、舒适性
在 on-policy 采样结果上叠加额外 reward loss，类似端到端多任务 loss 加权。
Reference
ON-POLICY DISTILLATION OF LANGUAGE MODELS: LEARNING FROM SELF-GENERATED MISTAKES
MiniLLM: Knowledge Distillation of Large Language Models
KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning
Unlocking On-Policy Distillation for Any Model Family
Qwen3 Technical Report
Qwen3-VL Technical Report
MiMo-V2-Flash Technical Report
Black-Box On-Policy Distillation of Large Language Models
GKD — swift 3.12.0.dev0 documentation
Recipe: Async On-Policy Knowledge Distillation Trainer — verl documentation
GLM-5
知乎专栏 SPDO: Reinforcement Learning via Self-Distillation
OPSD: On-Policy Self-Distillation for Large Language Models
POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration
提炼核心结论（可直接放进 PPT）
OPD 核心：student 采样，teacher 打分（token logits），on-policy 训练，根治曝光偏差，支持自演化；反向 KL 为主流选择，前向 KL 覆盖分布，两者可混合。
OPD 非常适合小模型；teacher 来源不限于外部模型，模型自身更强版本也可以做 teacher（自蒸馏 OPSD）。Hint 是 OPSD 关键：错误反馈、CoT、环境反馈、同批次成功样例都可以作为特权 hint。
智驾 VLA 场景落地路径：
GT 高质量数据 → SFT；存在更强专家模型 / 历史强版本 → OPD/MOPD
hint-based OPSD 可以和 VLA→T 链路、多层级 COC 长时序推理联合训练，课程学习降低训练难度
OPD 框架不限于优化准确率，可以叠加安全性、舒适性等自定义 reward loss；但难例探索、KL 混合配比仍有待进一步验证。
如果你需要，我可以继续输出：
版本 A：精简成一页 PPT 提纲
版本 B：整理成论文读书笔记（摘要 + 关键公式 + 研究 gap）
版本 C：智驾 VLA 链路 OPD 训练技术方案（训练流水线、数据构造、loss 公式）

GENERAL BACKGROUND
Motivation
 mitigate exposure bias and distribution mismatch（imitation learning）, e.g., student not expressive enough to mimic the teacher's distribution
 self-evolving
 dense token-level supervision（rewards）
 learn from external teachers or previous better selves.
usage
 especially useful for smaller models
 MOPD(Xiaomi MIMO V2): distilling multiple expert models into the same one, Xiaomi MIMO V2
 Strong to Weak Distillation (Qwen3 ): first offline distillation for knowledge acquisition, then online distillation. qwen3 VL
 GLM-5：swiftly recover the skills acquired in earlier SFT and RL multiple learning stages (Reasoning RL and General RL)
 training prompts: sampled from the corresponding teachers' RL training sets and mixed in appropriate proportions.
 Deepseek V4
 full vocab OPD instead of token-level KL estimate, to reduce grad var
 caching only the last-layer teacher hidden states, and compute the full-vocab logits on the fly.
 more than ten teacher models（experts obtained from sft and RL） covering various domains are employed to distill a single student model.
 Black-box on-policy distillation（e.g., from GPT5）: frames the student LLM as a generator and trains a discriminator to distinguish its responses from the teacher LLMs, creating a minimax game
a minimax game.
Different Post-training Objectives
Methods divergence Supervision Advantage disadvantage samples
SFT NLL GT sequences train-inference distribution mismatch
sequence-level KD NLL teacher sequences
supervised KD forward KL teacher logits overestimate void regions of teacher model
Rejection Sampling NLL student sequences with correct labels  on-policy
 reinforce positive samples，improve top@1? positive
on-policy imitation learning NLL student sequences, with expert labels (iteratively)  on-policy
 popular in robotics and deep RL positive(GT, expert)
on-policy distillation reverse KL divergence teacher logits（teacher-forcing） of student sequences
（do not backpropagate through the student’s sampling distribution）  evolve during training
 minimize train-inference distribution mismatch
 prioritize the tokens where the teacher assigns high probability, which can avoid low-quality generations
at the expense of less diverse generations positive and negative
Generalized KD NLL
forward KL (on/off)
reverse KL  teacher-generated or GT
 student sequences  evolve during training
 minimize train-inference distribution mismatch best divergence can be task-dependent positive and negative
On-policy Objectives
 Rejection Sampling: using only student sequences with positive labels. only positive data（GT）
 on-policy imitation learning：on-policy imitation approaches (e.g. Ross et al., 2011) iteratively collect sequences using the student policy, obtain expert labels for those sequences, and then retrain the student on this dataset.
 on-poilicy distillation：token-level feedback from the teacher logits
 evolve during training
 minimize train-inference distribution mismatch
图片 1 combination of forward and backward KL
图片 2 combination of off-policy sft/distillation and on-policy distillaiton
图片 3 unified view of RL（global reward） and on-policy distillation（token-wise reward）
图片 4 Application of MOPD in MIMO-V2
Choices of KL（derivation）:
 forward KL: mean-seeking, mode-cover，学习到老师模型平滑化的分布。
 reverse KL: mode-seeking, 聚类到老师模型的 modes（老师模型集中化的分布）。
D_KL（Q||P）
 P: teacher； Q: student,
 foward KL: D_KL (P||Q)
 reverse KL: D_KL (Q||P)
 The optimal mixture of for forward and reverse KD can vary across tasks.
Application to Autonomous Driving
☐ 自己更好的就用 GT samples 来 SFT 吧：例如智驾数据
☐ 别人更好的 (e.g., Mimo V2) 或者自己历史阶段更好的（e.g., DS v3, GLM5）就用 OPD 吧：
☐ 通识和指令遵从数据：蒸馏通用模型例如 Qwen3 235B
☐ 智驾不如端到端的 case：蒸馏端到端模型 VA
☐ 智驾原子能力、通用和难例决策能力：也可以先训练一个更大的模型当老师
☐ 此法可以配合大蒸小循环嵌套，举一反三
☐ TBD: 再仔细思考一下 forward KD: mean-seeking 和 backward KD: mode-seeking, less similar 应该怎么配合使用。
Extentions
“如果早知道这个报错 / XX 信息，你觉得刚才那个 token 选得怎么样？”
5.1 On-policy self-distillation
做法： prompt 中加入 hint 之后的学生初始化成了一个新老师：
 q+hint+on-policy a -> (self-）teacher logits。Intuition: a sufficiently capable LLM can rationalize external privileged reasoning traces and teach its weaker self (i.e., the version without access to privileged information)
 原始的 hint 主要是 error feed，包括
 hint 类型：
 verified cot: OPSD
 智驾领域可以考虑白盒 / 指令 COT（如果效果更好的话）
 error feedback: SDPO
 环境输出：例如代码解释器返回的运行时错误）、未通过的测试用例信息等。
 参考答案：如果在同一个采样批次中，有其他采样路径成功解决了问题，那么这个成功的答案也可以作为失败路径的 “反馈” 提供给教师。
 partial answer。有点类似指令 cot->reasoning cot，从给定指令到自己 think 出指令
 whatever hint
 注意：这也可以用来构造 DEBUG 数据。会更加模型亲和。
 注意：hint 也可以用于 RL 的种子数据
 注意：这个和指令 cot->reasoning cot（VLR->A 变成 VL->RA），指令控车 ->meta action 控车可以联合使用（VLA->T, VL->AT） @薛文元 @孙雅静 @刘洁锐
 如果 VLA->T 很好，那很可能我们只需要保证模型可以出很准的 A， T 自然也就好了。A<->T 质检的 consistency 可能天然就很高。
 这个可以用 sft 课程学习来实现，也可以结合 hint-based opsd 来实习哈～
 VLA->T 这个阶段在哪里？
5.2 单轮 -> 多轮， 多轮 -> 单轮
 单轮扩展成多轮推理，把中间轮当做 “助教”。
 心智模型：引导相当于把模型 “roll-in” 到一片更容易拿到奖励的中间区域 。先在这片区域学会 “怎么继续就能拿到奖励”，再慢慢学会 “从题目开头怎么走到这片区域”。这样难题就从 “既要发现好区域、又要学会怎么解” 变成 “先学会怎么解，再学会怎么走到那里”。
 迁移的关键不是让模型记住那段前缀，而是让推理过程产生重叠。因为 long-cot 的模型经常在生成中出现自检、回头改写、反思、回溯。POPE 的观点是：这些行为会让 “带引导的 rollout” 访问到很多更接近原题起点的内部状态，从而扩大与 “无引导 rollout” 可达状态的交集，迁移就更容易发生。
 依赖的能力项：instuction-following、Self-verification、backtracking behaviors
 智驾应用：多层级 COC reasoning、多层级导航指令。先学会主因 -> 决策，再学会主因的主因 -> 主因 -> 决策。慢慢扩展到 long horizon reasoning
多轮蒸馏回单轮推理: RLTF
 self-distillation
 feedback modeling
5.3 如何让模型在那些目前 “完全不会做” 的难题上获得有效的探索信号？（TBD）
 限制 entropy
 优化 clip 参数（减少 clip）: incease the high clip ratio, \epsilon_high
 直接优化 pass@k（衡量 exploration）
 简单 + 难问题混合训练：优化过程会过度专注于那些已经能解决的简单问题（以此快速通过刷分来降低 Loss），从而抑制了对困难问题的探索和改进。类似 curriculum learning
 专家数据 sft/off policy distill：高质量推理轨迹很贵，规模化困难；而且经验上蒸馏常常会 “封顶” 后续 RL 的收益，off-policy 训练还可能让 RL 不稳定。
5.4 怎么用 on-policy distillation 做准确率之外的其他 reward。例如安全性、舒适性等。
就是 on-policy 得到的结果上面可以再加一个 reward 的 loss。类似端到端的 loss-weight。
Reference
 ON-POLICY DISTILLATION OF LANGUAGE MODELS: LEARNING FROM SELF-GENERATED MISTAKES
 MiniLLM: Knowledge Distillation of Large Language Models
 KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning
 Unlocking On-Policy Distillation for Any Model Family
 Qwen3 Technical Report
 Qwen3-VL Technical Report
 MiMo-V2-Flash Technical Report
 Black-Box On-Policy Distillation of Large Language Models
 GKD — swift 3.12.0.dev0 documentation
 Recipe: Async On-Policy Knowledge Distillation Trainer — verl documentation
 GLM-5:
 https://zhuanlan.zhihu.com/p/2004506304065065334?share_code=1mybKRL35Wr9e&utm_psn=2009413232092274701
 SPDO: Reinforcement Learning via Self-Distillation
 OPSD: On-Policy Self-Distillation for Large Language Models
 POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration

笔记整理：On-Policy Distillation（OPD）体系 + 智驾 VLA 落地思路
核心主线：On-Policy Distillation（在线蒸馏），解决模仿学习里经典的曝光偏差 (exposure bias)、分布不匹配 (distribution mismatch)；区别于离线 SFT / 离线 KD，OPD 用 student 自己采样出来的序列，拿 teacher logits 做 token 级监督，实现模型自演化。
1. 通用背景 General Background
动机 Motivation
缓解模仿学习的曝光偏差、训练推理分布偏移：小模型（student）表达能力不足，很难拟合 teacher 的数据分布
支持自演化 self-evolving
稠密的 token 粒度监督信号（逐 token reward）
学习来源：外部专家 teacher，或是模型自身历史上更强的旧版本（过去更好的自己）
典型落地案例
MOPD（小米 MiMo V2）：多专家蒸馏，把多个专家模型压缩进同一个 student
Qwen3 / Qwen3-VL 强弱蒸馏 Strong to Weak：先离线蒸馏获取基础知识，再做在线蒸馏迭代
GLM-5：恢复多阶段 SFT+RL 学到的能力（推理 RL + 通用 RL）；训练 prompt 采样自 teacher 的 RL 数据集，按合适比例混合
DeepSeek V4
采用全词表 OPD，替代 token 级 KL 估计，降低梯度方差
只缓存 teacher 最后一层隐状态，实时计算全词表 logits
十余个来自 SFT/RL 阶段、覆盖多领域的专家 teacher，蒸馏单个 student
黑盒在线蒸馏（例如 GPT5 作为 teacher）：student 作为生成器，训练判别器区分 student 与 teacher 输出，构造极小极大博弈（minimax game）
适用场景：对小模型增益尤其明显
2. 后训练目标对比 Different Post-training Objectives
表格
方法	散度	监督来源	优点	缺点	样本
SFT	NLL	真实 GT 序列		训练 - 推理分布不匹配	
sequence-level KD	NLL	teacher 输出序列			
supervised KD	forward KL	teacher logits		高估 teacher 分布空白区域	
Rejection Sampling	NLL	student 采样序列，保留带正确标签样本	✅ 在线策略 on-policy
✅ 强化正样本，提升 top@1		正样本
on-policy imitation learning	NLL	student 迭代采样序列 + 专家标注	✅ on-policy
✅ 机器人 / 深度 RL 领域常用		正样本（GT / 专家）
on-policy distillation	reverse KL	student 采样序列，teacher logits（teacher-forcing，不回传 student 采样分布梯度）	✅ 训练过程持续演化
✅ 缓解 train-inference 分布偏移
✅ 优先学习 teacher 高概率 token，规避低质量生成	生成多样性下降	正 + 负样本
Generalized KD	NLL
forward KL（在线 / 离线）
reverse KL	teacher 生成样本 / GT
student 采样样本	✅ 训练演化
✅ 缓解分布偏移	最优散度和任务强相关	正 + 负样本
On-policy 目标三类细分
Rejection Sampling：仅保留 student 采样出来的正标签样本，只用 GT 正数据
on-policy imitation learning：迭代用当前 student 策略采样轨迹，请专家打标签，再训练 student（经典 Ross 2011）
on-policy distillation：teacher 提供 token 级 logits 反馈；训练迭代演化、减少训练推理分布偏移
配套图示概念
前向 KL + 反向 KL 混合
离线 SFT/KD + on-policy 蒸馏混合范式
RL（全局 reward）与 OPD（token-wise 逐 token reward）统一视图
MiMo-V2 中 MOPD 应用架构
3. KL 散度选择与推导 Choices of KL
约定：P=teacher 分布；Q=student 分布
Forward KL：
D
KL
	​

(P∣∣Q)
 前向 KL，均值匹配 mean-seeking：覆盖 teacher 全部模态 mode-cover；学习 teacher 平滑分布。会去拟合 teacher 分布所有区域，包括概率很低的空白区域。
Reverse KL：
D
KL
	​

(Q∣∣P)
 反向 KL，模态匹配 mode-seeking：让 student 集中拟合 teacher 高概率模态；聚类到 teacher 的核心 mode。不会强行去学习 teacher 低概率区域。
✅ 前向 / 反向 KD 最优混合比例，随任务变化。TBD：思考两者如何搭配使用，尤其分布相似度低的场景。
4. 自动驾驶领域应用 Application to Autonomous Driving
简单决策规则：
如果 GT 本身质量足够好：优先用 GT 样本做 SFT（智驾原生数据）
如果有更强外部模型（MiMo V2）/ 自身历史更强版本（DS v3、GLM5）：使用 OPD
通识、指令遵从：蒸馏通用大模型（如 Qwen3 235B）
智驾端到端能力弱于专家端到端 VA 模型：蒸馏 VA 端到端模型
智驾原子能力、通用决策、难例：训练更大模型作为 teacher
可嵌套「大模型蒸馏小模型」循环，反复迭代
待思考：forward KD（mean-seeking）与 reverse KD（mode-seeking）如何组合
5. 扩展方向 Extensions
核心思想：Hint 引导，问句附加提示信息，构造特权信息，生成更强 self-teacher。
一句话范式：q + hint + on-policy action → self-teacher logits
直觉：能力足够强的大模型，可以基于特权推理轨迹做解释，反过来教没有特权信息的弱版本。hint 本质是特权信息。
5.1 On-policy self-distillation（OPSD）在线自蒸馏
Hint 类型：
Verified CoT：OPSD；智驾可尝试白盒 / 指令式 CoT
Error feedback：SDPO；错误反馈
环境输出：代码解释器报错、失败测试用例等
同批次成功样本答案：同一次采样里成功路径，作为失败路径的反馈
Partial answer 部分答案：类似指令 CoT → reasoning CoT；VLR→A 转为 VL→RA
任意 hint
落地价值：
构建更适配模型的 DEBUG 数据集
hint 可作为 RL 种子数据
和智驾 VLA 链路联合：VLA→T、VL→AT（@薛文元 @孙雅静 @刘洁锐）
如果 VLA→T 效果好，只需要保证动作 A 足够准确，T 自然成立；A 与 T 一致性质检天然高
实现方式：SFT 课程学习 /hint-based OPSD
待确认：VLA→T 阶段放在整个链路哪个位置
5.2 单轮 ↔ 多轮蒸馏转换
单轮 → 多轮推理
把中间轮次当作助教。
心智模型：roll-in 引导，把模型带到更容易拿到奖励的中间状态；先学会 “从中间状态怎么拿到奖励”，再学习 “从初始状态走到中间状态”，拆分难题。
POPE 观点：长 CoT 模型自带自检、回溯改写；带 hint rollout 能访问更多靠近原始输入的内部状态，扩大与无 hint rollout 状态交集，提升迁移能力。
依赖能力：指令跟随、自校验、回溯能力
智驾落地：多层级 COC 推理、多层导航指令；课程式学习：主因→决策，再学习主因的主因→主因→决策，扩展长时序推理。
多轮 → 单轮推理（RLTF）
自蒸馏 + feedback modeling，把多轮推理能力压缩回单轮模型。
5.3 难题探索信号：如何让模型攻克完全不会的样例（TBD）
现有候选方案
限制熵 entropy
调整 clip 参数：降低 clip 强度，调高
ϵ
high
	​

，允许更大更新幅度
直接优化 pass@k 指标，度量探索能力
难易样本混合训练 / 课程学习：单纯训练容易样本会快速压低 loss，但抑制对难例探索
专家数据 SFT /off-policy 蒸馏：高质量推理轨迹成本高；蒸馏容易封顶后续 RL 收益，离线训练会造成 RL 不稳定
5.4 OPD 拓展非准确率类 Reward：安全性、舒适性
在 on-policy 采样结果上叠加额外 reward loss，类似端到端多任务 loss 加权。
Reference
ON-POLICY DISTILLATION OF LANGUAGE MODELS: LEARNING FROM SELF-GENERATED MISTAKES
MiniLLM: Knowledge Distillation of Large Language Models
KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning
Unlocking On-Policy Distillation for Any Model Family
Qwen3 Technical Report
Qwen3-VL Technical Report
MiMo-V2-Flash Technical Report
Black-Box On-Policy Distillation of Large Language Models
GKD — swift 3.12.0.dev0 documentation
Recipe: Async On-Policy Knowledge Distillation Trainer — verl documentation
GLM-5
知乎专栏 SPDO: Reinforcement Learning via Self-Distillation
OPSD: On-Policy Self-Distillation for Large Language Models
POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration
提炼核心结论（可直接放进 PPT）
OPD 核心：student 采样，teacher 打分（token logits），on-policy 训练，根治曝光偏差，支持自演化；反向 KL 为主流选择，前向 KL 覆盖分布，两者可混合。
OPD 非常适合小模型；teacher 来源不限于外部模型，模型自身更强版本也可以做 teacher（自蒸馏 OPSD）。Hint 是 OPSD 关键：错误反馈、CoT、环境反馈、同批次成功样例都可以作为特权 hint。
智驾 VLA 场景落地路径：
GT 高质量数据 → SFT；存在更强专家模型 / 历史强版本 → OPD/MOPD
hint-based OPSD 可以和 VLA→T 链路、多层级 COC 长时序推理联合训练，课程学习降低训练难度
OPD 框架不限于优化准确率，可以叠加安全性、舒适性等自定义 reward loss；但难例探索、KL 混合配比仍有待进一步验证。
如果你需要，我可以继续输出：
版本 A：精简成一页 PPT 提纲
版本 B：整理成论文读书笔记（摘要 + 关键公式 + 研究 gap）
版本 C：智驾 VLA 链路 OPD 训练技术方案（训练流水线、数据构造、loss 公式）