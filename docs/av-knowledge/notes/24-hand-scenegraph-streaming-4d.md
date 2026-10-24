# 手写笔记：Scene Graph、流式 4D 与具身 Memory

## 元信息

| 字段 | 内容 |
|------|------|
| 来源页码 | Handwriting Page 12–14（`_raw_batch_b.md` · ac7a / ad35 / adf6） |
| 所属分类 | B. VLA / 决策 · 4D 表征与 Memory；C2 偏好对齐 |
| 笔记类型 | 手写精炼（非豆包对话） |
| 相关原始图 | `../assets/handwriting/01a169b1-ac7a-*` … `01a169b1-adf6-*` |

## 一句话摘要

用 **4D Scene Graph + VLM/RAG** 做动态打标与慢/快 Rollout→SFT→DPO；在线侧是 **LiT/多相机→adaptor→KV+Router+SSM** 的流式 4D；具身 FM 以 **VLM+WM+Critic** 与 Omni Stream 记忆库做长程一致。

## 核心关键词

`Scene Graph` · `RAG` · `VLM` · `Rollout` · `DPO` · `LiT` · `SSM` · `KV Cache` · `Router` · `Omni Stream` · `World Model` · `Critic` · `Memory Bank`

## 缩写与术语详解

| 缩写/词 | 全称 / 含义 | 在本页语境 |
|---------|-------------|------------|
| **Scene Graph** | 场景图 | 物体–关系–属性的图结构；4D 表示带时序 |
| **RAG** | Retrieval-Augmented Generation | 检索先验喂给 VLM 打标/描述 |
| **VLM** | Vision-Language Model | 动态打标、caption、具身推理链 |
| **VLA** | Vision-Language-Action | Caption Loss[?] 等对齐项 |
| **TBT** | Turn-by-Turn | 模糊/细粒度全导航条件 |
| **SFT** | Supervised Fine-Tuning | Rollout 标注后的监督 |
| **DPO** | Direct Preference Optimization | 快/慢 pair 偏好对齐 |
| **LiT** | Locked-image Text Tuning（或 Language-Image Tuning） | 双视角图文锁定/调谐进 2D adaptor |
| **SSM** | State Space Model（如 Mamba 族） | 与 Attn/Router 并列，维护流式 **State** |
| **KV Cache** | Key-Value Cache | 流式注意力状态复用 |
| **Router** | 路由模块 | 在 KV/Attn/SSM 间分配计算或专家 |
| **WM** | World Model | 物理与动作预测 |
| **Critic** | 价值/评判模型 | 对 VLM+WM 轨迹/动作打分 |
| **Omni Stream** | 手稿专名：统一流式 4D 记忆 | 兼容 2D ViT 与图像 align；长记忆衰减但不清零 |
| **BCG / P** | 输出头缩写不明 | 与 Ray/Depth/Scene Flow/planning 并列，保持原形 |

## 知识脉络

### 1. 4D Scene Graph 打标与偏好（Page 12）

```
[检索 RAG] ──→ VLM ←── [图]
                ↓
     盲区 / 内轮差[?] / 场景列表 / 物图
                ↓
         4D 时序 · 动态打标
                ↓
         VLA Caption Loss[?]
```

关注：物体属性；全局/局部；模糊 vs 细粒度导航；TBT。

**训练对齐**

```
{慢, 快} → Rollouts（标快慢）→ SFT → pair → DPO
（手动 · 关键词[?] · 置信度过滤）
```

手稿 **慢激活率** 提升信号（内部）：

| 集合 | 前 → 后 |
|------|---------|
| 通用 | 28.4% → 74.3% |
| 难例 | 25.9% → 70.6% |

### 2. 流式多模态 4D（Page 13）

```
流式文本 ──────────────────────────────┐
LiT（双视角）→ 2D adaptor → add & cat   │  语义 & 空间/时间 Cross-Attn
m-cam → Image Patch → Attn → 4D adaptor┘
                    ↓
        ┌─────────────────────────┐
        │ KV Cache · Attn · Router │
        │ SSM ←→ State             │
        └─────────────────────────┘
        流式窗口：约 6s · 2s @ 2Hz（2s 支路帧率手稿未写全；对照 [01] 常见 2s@10Hz）
                    ↓
        Streaming embedding → Ray / Depth / Scene Flow / Camera /
                              Scale / P·BCG[?] / planning
```

要点：2D adaptor 与 4D adaptor 分工；SSM 扛长时状态，Attn 扛短时交互；Router 控算力。

### 3. 具身 Foundation Model 与 Memory（Page 14）

```
human data → 物理交互 → 复杂任务 → Foundation Model → 真机
```

- 模态：视 · 触 · 关节；夹爪配置「2 / 2.5 指」+ RL。  
- 能力①：力觉柔顺、精细操作。

```
Mem …
  VLM（长程显式记忆）→ 语义接口 · 具身推理链
  WM （物理 · 动作）
       └──────────→ Critic Model
```

**Memory Bank Event / Omni Stream**

- 兼容 2D ViT 与图像 align；需训、耗资源。  
- n 百帧 streaming；**记忆衰减但不会消失**。  
- depth 一致；输入可达 1280：stage1 低分辨率 → stage2 高分辨率到 1280。  
- 几何 · 语义 · 自监督 dense depth；追问「depth 是否足够」。

## 知识点校验

1. **Correct**：慢/快行为 pair → SFT → DPO 与 [12](12-planning-dpo-rl.md) 同构；流式 4D 用 KV+状态模型符合长视频/驾驶时序约束；VLM+WM+Critic 是具身常见三件套。
2. **Caution**：慢激活率百分比、Caption Loss 读法、LiT 确切论文指称、`P/BCG` 头、Omni Stream 是否产品名——均为手稿级，**不可当论文结论引用**。
3. **Caution**：「6s·2s@2Hz」与知识库动态帧率口径（6s@2Hz / 2s@10Hz）需对齐，避免混用帧率。
4. **Expand**：Scene Graph 节点/边 schema 与占用网络接口；SSM State 与 [16](16-qwen3vl-vla-architecture.md) State Management 是否同一套；Memory 衰减函数与关键事件写入策略。

## 与其他主题的关联

- [12 Planning DPO/RL](12-planning-dpo-rl.md) — 慢快 pair 与 DPO
- [01 VLM 训练阶段](01-vlm-training-stages.md) — 动态帧率 6s/2s
- [16 Qwen3-VL VLA](16-qwen3vl-vla-architecture.md) — 4D 主干与 State Management
- [13 BEV 深度](13-bev-depth-models.md) — Depth / Scene Flow 头
- [09 Alpamayo](09-alpamayo-reasoning-vqa.md) — 打标与推理条件
- [08 CoC](08-vla-ambiguity-coc.md) — 因果链与导航一致
- [19 VL–4D 训练管线](19-hand-vl-4d-training-pipeline.md) — 离线 4D 预训练如何接到流式推理
