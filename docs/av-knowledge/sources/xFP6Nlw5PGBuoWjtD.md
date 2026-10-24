# VLM智驾模型训练阶段整理

Thread ID: xFP6Nlw5PGBuoWjtD
URL: https://www.doubao.com/thread/xFP6Nlw5PGBuoWjtD

---

整理一下这个格式，训练阶段
目的
可训练参数
数据类型
数据配比
备注
1 VLM（视觉理解）
图文高层语义对齐，不破坏 LLM 和 VIT encoder
连接器
简单自然图像、简单文档 (大字识别等)
2012：VLM 多模态模型训练
2 VLM 预训练和 SFT（视觉理解）
图文、视频文本细粒度对齐，编码器提升
视觉编码器 + 连接器，语言模型
复杂图像、复杂文档、xxx
3 4D 编码器对齐 VLM
4D 编码器 + 连接器，视觉编码器 + 连接器，语言模型
环视图 - 文本
2012： 带 4D 编码器的 VLM 训练
阶段 4.1 FM/4D 编码器对齐 LLM（可选）
补充 4D 信息
(通过 FM 或者 VLM 直接建模环视视频流）
4D 编码器连接器
VFLM, 环视图 - 文本
VFLA: 环视图 - 决策
ADS:
外接 FM/4D 增训空间理解
阶段 4.2 智驾垂域增训（可选 4D 编码器）
保持 VLM 能力，新增智驾垂域能力
4D 编码器连接器，视觉编码器 + 连接器，语言模型
6s2Hz &/ 2s10Hz (dynamic frame rate)
SFT:
VLM： 550K DriveCore 单帧 & 多帧 VLA: 100w 9*4 v0.3 nothinking 6s2hz 视频
OPD（+sft）: 223K 中文分析，31K 中文分析 + 决策；44K CODA-LM 物体理解
VLM: not yet
VA：
ADS:
VLM 增训智驾能力，通用决策
model size scaling： DriveBench V4.5 2B 61 分，4B 68 分，8B 72 分
阶段 5 VLM（+AE） 后训练（多模态理解）
replay 通用决策能力，VQA
新增 coc + 难例决策 + 轨迹监督
语义决策 ->special token
（考虑是否和上一阶段 merge）
VLM（时空编码器 + 连接器，视觉编码器 + 连接器，语言模型 lora）
2s10Hz
VIL->AP 【FV30.3，630w 帧，高帧率】
VIL->RAP【FV30.3 包含自动和数据部人工产线 + 去年 5 大难例， 30w 帧】
ADS:
replay 通用决策 + 难例决策
阶段 6.1 VLA
对齐 VLM (4&5) 和 AE (5)
planner 连接器， planner
VLAT: 144w 帧
ADS:
对齐 VLM (4&5) 和 AE (5)
阶段 6.2.1 VLA（optional）
联训
VLM+planner 链接器 + planner
单指令 L3：VLA->T, VLRA->T, VL*->T
复合指令：L2 指令体系
训练：
warm_up stage:
VLA->T 要提前到这个阶段
指令导航控车，指令泛化
联训
阶段 6.2.2 VLA
联训
VIT proj，FM proj, LLM, planner 链接器 + planner
（除了 VIT, FM 的其他所有参数）
VFSL->AT：800w 帧 ->3000+w 帧 -> 全量
VFSL->RAT: 30w 帧
VLM: DriveCore 50w
VLM 的其他数据：xxx
训练：
stage1:
VF_mAT-F_s 3600w
stage2：
replay VF_mAT-F_s 100w 以内
VF_mRAT-F_s ：30w
推理：
VLAT->VLRAT
ADS:
联训
阶段 6.2.3 VLA
联训
VLM+planner 链接器 + planner
replay: VFL->AT, VFSL->RAT
单指令 L3：VLA->T, VLRA->T
复合指令：L2 指令体系
训练：
warm_up stage:
VLA->T 要提前到这个阶段
指令导航控车，指令泛化
ADS:
联训
我将按照你指定的目的、可训练参数、数据类型、数据配比、备注统一字段，梳理所有训练阶段内容，规整为清晰结构化表格，修正层级、统一格式、补齐缺失字段，内容完全保留原始信息。
VLM智驾模型全阶段训练流程梳理表
创建时间：09-23 09:01
需要我帮你精简表格冗余表述，让整体更清爽直观吗？
直接在对话框回答
VLM 智驾模型训练阶段整理表
表格
训练阶段	目的	可训练参数	数据类型	数据配比	备注
阶段 1：VLM（视觉理解）	图文高层语义对齐，不破坏 LLM 和 VIT encoder	连接器	简单自然图像、简单文档 (大字识别等)	无	2012：VLM 多模态模型训练
阶段 2：VLM 预训练和 SFT（视觉理解）	图文、视频文本细粒度对齐，编码器提升	视觉编码器 + 连接器，语言模型	复杂图像、复杂文档、xxx	无	无
阶段 3：4D 编码器对齐 VLM	4D 编码器与 VLM 对齐	4D 编码器 + 连接器，视觉编码器 + 连接器，语言模型	环视图 - 文本	无	2012：带 4D 编码器的 VLM 训练
阶段 4.1 FM/4D 编码器对齐 LLM（可选）	补充 4D 信息，通过 FM 或者 VLM 直接建模环视视频流	4D 编码器连接器	VFLM：环视图 - 文本
VFLA: 环视图 - 决策	无	ADS：外接 FM/4D 增训空间理解
阶段 4.2 智驾垂域增训（可选 4D 编码器）	保持 VLM 能力，新增智驾垂域能力	4D 编码器连接器，视觉编码器 + 连接器，语言模型	智驾视频，动态帧率	帧率：6s2Hz &/ 2s10Hz
SFT:
VLM：550K DriveCore 单帧 & 多帧
VLA: 100w 9*4 v0.3 nothinking 6s2hz 视频
OPD（+sft）: 223K 中文分析，31K 中文分析 + 决策；44K CODA-LM 物体理解
VA：暂无
VLM：not yet	ADS: VLM 增训智驾能力，通用决策
model size scaling：DriveBench V4.5 2B 61 分，4B 68 分，8B 72 分
阶段 5 VLM（+AE） 后训练（多模态理解）	replay 通用决策能力，VQA；新增 coc + 难例决策 + 轨迹监督，语义决策 ->special token；考虑是否和上一阶段 merge	VLM（时空编码器 + 连接器，视觉编码器 + 连接器，语言模型 lora）	高帧率智驾视频	帧率：2s10Hz
VIL->AP 【FV30.3，630w 帧】
VIL->RAP【FV30.3，30w 帧，含自动 + 人工产线 + 去年 5 大难例】	ADS: replay 通用决策 + 难例决策
阶段 6.1 VLA	对齐 VLM (4&5) 和 AE (5)	planner 连接器， planner	VLA 视频帧	VLAT: 144w 帧	ADS: 对齐 VLM (4&5) 和 AE (5)
阶段 6.2.1 VLA（optional）	联训；warm_up 阶段提前 VLA->T；指令导航控车，指令泛化	VLM+planner 链接器 + planner	指令控车数据	单指令 L3：VLA->T, VLRA->T, VL*->T
复合指令：L2 指令体系	ADS：联训
阶段 6.2.2 VLA	联训，迭代扩充数据	VIT proj，FM proj, LLM, planner 链接器 + planner（除 VIT, FM 的其他所有参数）	VFSL 智驾视频，VLM 数据	VFSL->AT：800w 帧→3000+w 帧→全量
VFSL->RAT: 30w 帧
VLM: DriveCore 50w
stage1: VF_mAT-F_s 3600w
stage2：replay VF_mAT-F_s 100w 以内，VF_mRAT-F_s：30w
推理：VLAT->VLRAT	ADS：联训
阶段 6.2.3 VLA	联训，replay 固化决策能力，指令泛化	VLM+planner 链接器 + planner	VFL、VFSL 复盘数据	replay: VFL->AT, VFSL->RAT
单指令 L3：VLA->T, VLRA->T
复合指令：L2 指令体系
warm_up stage: VLA->T 前置	ADS：联训
如果你想要纯文本版本（不带 markdown 表格），我可以直接输出。

整理一下这个格式，训练阶段
目的
可训练参数
数据类型
数据配比
备注
1 VLM（视觉理解）
图文高层语义对齐，不破坏 LLM 和 VIT encoder
连接器
简单自然图像、简单文档 (大字识别等)
2012：VLM 多模态模型训练
2 VLM 预训练和 SFT（视觉理解）
图文、视频文本细粒度对齐，编码器提升
视觉编码器 + 连接器，语言模型
复杂图像、复杂文档、xxx
3 4D 编码器对齐 VLM
4D 编码器 + 连接器，视觉编码器 + 连接器，语言模型
环视图 - 文本
2012： 带 4D 编码器的 VLM 训练
阶段 4.1 FM/4D 编码器对齐 LLM（可选）
补充 4D 信息
(通过 FM 或者 VLM 直接建模环视视频流）
4D 编码器连接器
VFLM, 环视图 - 文本
VFLA: 环视图 - 决策
ADS:
外接 FM/4D 增训空间理解
阶段 4.2 智驾垂域增训（可选 4D 编码器）
保持 VLM 能力，新增智驾垂域能力
4D 编码器连接器，视觉编码器 + 连接器，语言模型
6s2Hz &/ 2s10Hz (dynamic frame rate)
SFT:
VLM： 550K DriveCore 单帧 & 多帧 VLA: 100w 9*4 v0.3 nothinking 6s2hz 视频
OPD（+sft）: 223K 中文分析，31K 中文分析 + 决策；44K CODA-LM 物体理解
VLM: not yet
VA：
ADS:
VLM 增训智驾能力，通用决策
model size scaling： DriveBench V4.5 2B 61 分，4B 68 分，8B 72 分
阶段 5 VLM（+AE） 后训练（多模态理解）
replay 通用决策能力，VQA
新增 coc + 难例决策 + 轨迹监督
语义决策 ->special token
（考虑是否和上一阶段 merge）
VLM（时空编码器 + 连接器，视觉编码器 + 连接器，语言模型 lora）
2s10Hz
VIL->AP 【FV30.3，630w 帧，高帧率】
VIL->RAP【FV30.3 包含自动和数据部人工产线 + 去年 5 大难例， 30w 帧】
ADS:
replay 通用决策 + 难例决策
阶段 6.1 VLA
对齐 VLM (4&5) 和 AE (5)
planner 连接器， planner
VLAT: 144w 帧
ADS:
对齐 VLM (4&5) 和 AE (5)
阶段 6.2.1 VLA（optional）
联训
VLM+planner 链接器 + planner
单指令 L3：VLA->T, VLRA->T, VL*->T
复合指令：L2 指令体系
训练：
warm_up stage:
VLA->T 要提前到这个阶段
指令导航控车，指令泛化
联训
阶段 6.2.2 VLA
联训
VIT proj，FM proj, LLM, planner 链接器 + planner
（除了 VIT, FM 的其他所有参数）
VFSL->AT：800w 帧 ->3000+w 帧 -> 全量
VFSL->RAT: 30w 帧
VLM: DriveCore 50w
VLM 的其他数据：xxx
训练：
stage1:
VF_mAT-F_s 3600w
stage2：
replay VF_mAT-F_s 100w 以内
VF_mRAT-F_s ：30w
推理：
VLAT->VLRAT
ADS:
联训
阶段 6.2.3 VLA
联训
VLM+planner 链接器 + planner
replay: VFL->AT, VFSL->RAT
单指令 L3：VLA->T, VLRA->T
复合指令：L2 指令体系
训练：
warm_up stage:
VLA->T 要提前到这个阶段
指令导航控车，指令泛化
ADS:
联训

我将按照你指定的目的、可训练参数、数据类型、数据配比、备注统一字段，梳理所有训练阶段内容，规整为清晰结构化表格，修正层级、统一格式、补齐缺失字段，内容完全保留原始信息。
VLM智驾模型全阶段训练流程梳理表
创建时间：09-23 09:01
需要我帮你精简表格冗余表述，让整体更清爽直观吗？

VLM 智驾模型训练阶段整理表
表格
训练阶段	目的	可训练参数	数据类型	数据配比	备注
阶段 1：VLM（视觉理解）	图文高层语义对齐，不破坏 LLM 和 VIT encoder	连接器	简单自然图像、简单文档 (大字识别等)	无	2012：VLM 多模态模型训练
阶段 2：VLM 预训练和 SFT（视觉理解）	图文、视频文本细粒度对齐，编码器提升	视觉编码器 + 连接器，语言模型	复杂图像、复杂文档、xxx	无	无
阶段 3：4D 编码器对齐 VLM	4D 编码器与 VLM 对齐	4D 编码器 + 连接器，视觉编码器 + 连接器，语言模型	环视图 - 文本	无	2012：带 4D 编码器的 VLM 训练
阶段 4.1 FM/4D 编码器对齐 LLM（可选）	补充 4D 信息，通过 FM 或者 VLM 直接建模环视视频流	4D 编码器连接器	VFLM：环视图 - 文本
VFLA: 环视图 - 决策	无	ADS：外接 FM/4D 增训空间理解
阶段 4.2 智驾垂域增训（可选 4D 编码器）	保持 VLM 能力，新增智驾垂域能力	4D 编码器连接器，视觉编码器 + 连接器，语言模型	智驾视频，动态帧率	帧率：6s2Hz &/ 2s10Hz
SFT:
VLM：550K DriveCore 单帧 & 多帧
VLA: 100w 9*4 v0.3 nothinking 6s2hz 视频
OPD（+sft）: 223K 中文分析，31K 中文分析 + 决策；44K CODA-LM 物体理解
VA：暂无
VLM：not yet	ADS: VLM 增训智驾能力，通用决策
model size scaling：DriveBench V4.5 2B 61 分，4B 68 分，8B 72 分
阶段 5 VLM（+AE） 后训练（多模态理解）	replay 通用决策能力，VQA；新增 coc + 难例决策 + 轨迹监督，语义决策 ->special token；考虑是否和上一阶段 merge	VLM（时空编码器 + 连接器，视觉编码器 + 连接器，语言模型 lora）	高帧率智驾视频	帧率：2s10Hz
VIL->AP 【FV30.3，630w 帧】
VIL->RAP【FV30.3，30w 帧，含自动 + 人工产线 + 去年 5 大难例】	ADS: replay 通用决策 + 难例决策
阶段 6.1 VLA	对齐 VLM (4&5) 和 AE (5)	planner 连接器， planner	VLA 视频帧	VLAT: 144w 帧	ADS: 对齐 VLM (4&5) 和 AE (5)
阶段 6.2.1 VLA（optional）	联训；warm_up 阶段提前 VLA->T；指令导航控车，指令泛化	VLM+planner 链接器 + planner	指令控车数据	单指令 L3：VLA->T, VLRA->T, VL*->T
复合指令：L2 指令体系	ADS：联训
阶段 6.2.2 VLA	联训，迭代扩充数据	VIT proj，FM proj, LLM, planner 链接器 + planner（除 VIT, FM 的其他所有参数）	VFSL 智驾视频，VLM 数据	VFSL->AT：800w 帧→3000+w 帧→全量
VFSL->RAT: 30w 帧
VLM: DriveCore 50w
stage1: VF_mAT-F_s 3600w
stage2：replay VF_mAT-F_s 100w 以内，VF_mRAT-F_s：30w
推理：VLAT->VLRAT	ADS：联训
阶段 6.2.3 VLA	联训，replay 固化决策能力，指令泛化	VLM+planner 链接器 + planner	VFL、VFSL 复盘数据	replay: VFL->AT, VFSL->RAT
单指令 L3：VLA->T, VLRA->T
复合指令：L2 指令体系
warm_up stage: VLA->T 前置	ADS：联训
如果你想要纯文本版本（不带 markdown 表格），我可以直接输出。