# Handwriting Raw Batch B

> 来源：7 张手写笔记照片（附件顺序 = Page 08–14）。转录已做笔误/缩写还原；模糊处标 `[?]`。

## Page 08 — a954-756b

**主题**: VQA/轨迹 SFT 的数据多样性、P1→P2 固化问题，以及 Polaris2 high-overlap 与收敛诊断。

**转录（纠错后）**:

- 补 VQA；跨模态检索
- Data-driven · condition · text · Data ViT · FM → MA · Traj
- loss ↓ · 放更多 data · 加强 grounding SFT 正样本
- `pv` / `bev` 组合表：
  - ✓ ✓
  - ✓ ✗
  - ✗ ✓
  - ✗ ✗
- skill ↓
- `P1 + Q → A`
- `P2 + Q → MA-Traj`
- P1 多样性；P2 很固化；继承不到 P1；推理格式
- 泛化 · C-EVAL 提点 1.4%
- 通用…
- attention；dropout 和 Augment
- replay · 多样性
- 人设 · prompt · segment
- π0.7 泛化很好
- 改写等 · dropout
- **1. traj**
  - 注入
  - RL reward 放 SFT 里 · Quality：像 loss weight
  - 训练时区分好坏 · 推好的
- 扰动：`{ bev, pv, ViT 图像, crop, random }`
- Polaris2 · HKU NLP
  - `A B C [D] [Av]`
  - `A B C [E] [Ax]`
  - high overlap · low difference
- 最大的 smooth 看看是否降
- 可能 lr 太大 · 任务难度（过早收敛）
- 数字对建模难

**不确定项**:

- `FM → MA`：按上下文还原为 Flow Matching / Foundation Model → Meta-Action；原笔记缩写未写全。
- C-EVAL「1.4%」：也可能是「14%」笔迹压缩，以百分点提升理解更稳妥但未确认。
- `[Av]` / `[Ax]`：动作/轨迹分支标记，字母含义未写明。
- 「推好的」：也可能是「推广好的」。

---

## Page 09 — aa0d-72b9

**主题**: 消息通道 → Agent Loop / Context Builder / Agent Runner 的 LLM Agent 调用闭环。

**转录（纠错后）**:

```
Telegram / Webhook / CLI 发消息
  → inbound Message（bus）· workspace
  → [Agent Loop] 选 session
      加载 skills · memory · 历史
  → Context Builder
      组装 system prompt + 上下文
  → [Agent Runner]
      调用 LLM Provider
  → model 请求 → 协作/执行 → 回传 · 迭代
  → response 写入 session · 发布 OUT
```

- **Agent Loop**（偏 channel）
  - 会话 · 工作流 · 上下文
  - Hook · 进度推送
- **Agent Runner**（面向 LLM）
  - 模型调用 · 工具调用（循环）
  - 流式输出 · 迭代终止

**不确定项**:

- 「协作」：也可能是「执行 / tool 调用」草写。
- 「发布 OUT」：outbound 发布，末字略糊。

---

## Page 10 — aad4-705f

**主题**: ViT 族（DINO / DA3 / FastViT / Qwen-ViT）对比，叠加 MessageBus Agent 系统骨架。

**转录（纠错后）**:

**上半：视觉骨干**

- (global attention · 高分辨率计算 ↑↑)
- 可学习位置编码（固定分辨率 · 绝对坐标）
- **DINO** → ViT · 无监督 · 单视角 · 先单视角再整合视角
- **DA3** → 相对位置 · ViT + token 重排 · 多视角 3D
- **FastViT**：BN · DW 卷积 · 基于 DINOv2
- **Qwen-ViT**：
  - Conv · 多模 · 旋转位置编码 2D/3D/4D
  - (window attention) · 修 ViT 层 · DeepStack
  - `hwt → adapter`

**下半：Agent 架构**

```
发消息 → MessageBus
┌ CLI / WebUI / Telegram / Discord / 飞书 / 微信 ┐
└──────────→ MessageBus（Inbound / Outbound）──→ LLM Provider
session key · path/workspace · skills · memory · task

┌ AgentLoop（会话管理 / 上下文）
│ AgentRunner（LLM + 工具循环）
└─ Tools: 文件 / shell / Web / MCP / Cron
   Memory + Dream · 长期记忆

restore → compact → build → run → save → respond → done
```

**不确定项**:

- **DA3**：推断为 Depth Anything 3（相对位置 + 多视角 3D 相符），原笔记未展开。
- `hwt`：按 H×W×T（高/宽/时）理解。
- DW：Depthwise（卷积）。
- 流程中 `save`：也可能写作 `serve`，按 session 持久化取 save。

---

## Page 11 — abac-72e0

**主题**: 多模态 Dropout / Spatial PE，以及 CoC 质量与一致性的 Reward Model 打分项。

**转录（纠错后）**:

- 多模态 Dropout · 2D 随机丢弃
- patch drop
- Spatial PE：`[x, y, dist, sin(yaw), cos(yaw), rel_ang]`

**Reward model** `{`

1. **CoC 质量**
   - 感知 0–5
   - 推理 0–5
   - 决策 0–5 → 得分部分
   - 用 CoC 让 LLM 生成问题 Yes/No…
2. **CoC + TBT 一致**
   - Query · 3.5–39.7B `[?]`
3. **决策–轨迹一致**（画图上）· 车道线 · 障碍
4. **CoC–决策**
5. **CoC 因果倒置**

`}`

**不确定项**:

- CoC：按智驾笔记体系还原为 Chain of Causation（因果链），非普通 CoT。
- TBT：Turn-by-Turn（与后页「TBT 导航」一致）。
- `rel_ang`：相对角特征，笔迹略简。
- 「3.5–39.7B」：疑为模型规模/版本标注（如 Qwen 类），原文不完整。
- 感知/推理/决策量表：亦见「>0.5」类阈值读法，此处按 0–5 分制转录。

---

## Page 12 — ac7a-7c16

**主题**: 4D Scene Graph + VLM/RAG 动态打标，以及慢/快 Rollout → SFT → DPO 的激活率提升。

**转录（纠错后）**:

**4D Scene Graph**

```
[检索 RAG] ──→ VLM ←── [图]
                │
         盲区 / 内轮预测[?] / 场内[?] / 数据 / 物图
                ↓
         4D 时序 · 动态打标
                ↓
         VLA Caption Loss[?]
```

- 物体属性
- 全局 / 局部
- 模糊 / 细粒度全导航
- TBT 导航

**训练对齐**

```
{慢, 快} → Rollouts（标快慢）→ SFT → pair 对 → DPO
（通过手动 · 关键词[?] · 置信度）
```

- **慢激活率**
  - 通用：28.4% → 74.3%
  - 难例：25.9% → 70.6%

**不确定项**:

- 「内轮预测」：原字似「内轻」，按弯道内轮差速/轨迹常见术语推断。
- 「场内 / 网内 / 商区…」：VLM 下列项笔迹多义，场景列表与能力列表两种读法并存。
- 「VLA Caption Loss」：亦可能读作 NHA Caption / LoC；取与 Caption 损失更贴的读法并标疑。
- 「关键词」：也可能是「关键词过滤 / 关键词器」简写。

---

## Page 13 — ad35-7f5e

**主题**: 流式多模态 4D 架构：LiT/多相机 → adaptor/Attn → KV+Router+SSM，多任务头输出。

**转录（纠错后）**:

```
流式 · 文本 ──→
LiT（双视角）──→ 2D adaptor ──→ add & cat
                 语义 & 空间/时间 · Cross Attention

m-cam → [Image Patch] → 中间 Attention → Attention
                              ↓
                         4D adaptor
                              ↓
              ┌─────────────────────────┐
              │ KV Cache · Attn · Router │
              │ SSM ←→ State             │
              └─────────────────────────┘
              流式 · 6s · 2s @ 2Hz

              ↓ 多模态 4D Streaming embedding
              ├ Ray
              ├ Depth
              ├ Scene Flow
              ├ Camera
              ├ Scale
              ├ P / BCG[?]
              └ planning
```

**不确定项**:

- LiT：Language-Image Tuning / Locked-image Text Tuning。
- 「6s · 2s @ 2Hz」：与动态帧率「6s@2Hz / 2s@…」笔记口径对齐；2s 支路帧率原文未写全（常见对照为 2s@10Hz）。
- `P / BCG`：输出头缩写不明（疑 Pose/BEV/PnC 等），保持原形。
- 「中间 Attention」：中间层/中间融合注意力，具体模块名未写。

---

## Page 14 — adf6-7dae

**主题**: 具身 Foundation Model（VLM+WM+Critic）与 Omni Stream 4D 记忆库 / 多级 depth。

**转录（纠错后）**:

```
human data → 物理交互 → 复杂任务 → Foundation Model → 真机
```

- 2 / 2.5 指 · RL
- 视 · 触 · 关节 · 多模态

**①** 力觉柔顺 · 精细操作

```
        Mem …
   ┌ VLM · 长程 · 显式记忆 ──→ 语义接口 · 具身推理链
   │ WM  · 物理 · 动作
   └──────────────→ Critic Model
```

**② Memory Bank Event**

- Omni Stream · 4D · 兼容 2D ViT 与图像 align
- 需要训 · 资源
- n 百帧 · streaming · memory 衰减但不会消失
- depth 一致
- 1280 输入
- stage 1 低分辨率
- stage 2 高分辨率 → 1280
- 几何 · 语义 · 自监督（dense depth）
- depth 是不是足够？

**不确定项**:

- 「2/2.5 指」：二指 / 二点五指夹爪配置。
- Omni Stream：专名大小写按专有名词写作 Omni Stream；是否正式产品名未确认。
- 「图像 align」：跨模态对齐对象（ViT↔图像/LiT）笔迹略简。
- Critic Model 与 Mem 的反馈箭头方向：图示有回指，细节以框图为准。
