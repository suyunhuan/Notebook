# 主题：规划、动作与扩散策略

关联页：03, 06, 16, 18, 21, 23, 24, 27

---

## 1. 规划问题分层

笔记中的串级逻辑（页 03）：

```
全局导航 → 局部路径 → 红绿灯约束到局部 → 轨迹规划效用
```

对应模块：

| 模块 | 输入要点 | 输出 |
|------|----------|------|
| **NaviNN** | 路径边缘关系、BEV/PV 时序 | **导航场**（静态） |
| **TrafficNN** | TBT/EHP、灯态特征、NaviFeat | **路口通行概率** |
| **Ego-HMF** | 感知隐层 + Traffic/Navi Feat | **轨迹指导** |

FSD 规划要素汇总：轨迹预测、占据场、速度场、导航场、通行概率；并明确要 **引入 World Model**。

---

## 2. 从 Token 到 Action 的两条解码路径

### 2.1 确定性 / 自回归头

- **Action Decoder**：接共享 Encoder 或 AR Large World Model。  
- **ARDP 纵向**（页 06）：  
  - 交互 · 感知推演  
  - 语义 · 风险降速  
- 风险量：TTC、速度相关项、collision Δ。

### 2.2 扩散 / DiT 头

```
Condition: ego / time / point / latent+token
        → DiT × N
        → waypoints 例如 (5, 2) 或 EgoTraj/ObjTraj 25×2
```

损失权重示例（页 23）：AZ=1, AR=1, PP=0.05, llm/FM2LLM=0.1；监控 `grad loss`。

结构细节（页 24）：FM 特征经 MLP+RMSNorm → **AZ×3 Cross-Attn**（Q 来自 lm token，K/V 来自 FM）→ lm-head；并行 DiT 出航点（Diffusion Policy）。

---

## 3. Latent Reasoning（世界状态 rollout）

页 16：

1. Large Transformer 做 **World state Reasoning**。  
2. 以 0.5s 步长 rollout：`t+0.5 … t+2.5`（约 2s、4 步）。  
3. 汇入 **Action Decoder**，可接 **AR LWM**。  
4. 训练可用 **God state**（特权未来/完美状态）稳定监督。

直觉：先在隐空间「想象未来」，再解码动作，比单帧回归更抗噪声。

---

## 4. 约束与引导（页 21）

| 方法 | 作用 |
|------|------|
| Classifier Guidance | 用分类/规则信号引导生成 |
| **STL**（Signal Temporal Logic） | 量化违例程度，回传梯度 |
| Gen-Drive / DIFFAD 等 | 逻辑约束注入规划 |
| Action-Weighted Reward | 时间加权提升 Planning Acc |
| Planning Diversity | 避免模式坍缩 |

扩散三条线：① 生成后 RL 微调 ② 联合感知–规划 ③ STL 约束引导。

---

## 5. 与感知接口

- BEV 头：BZL / GOD / COG / Navi；PV：Seg / Depth。  
- Occ Flow + SDF：给规划提供占据与距离场。  
- MetaAction：辅助离散意图，降低连续控制搜索空间。  
- navi encode：road class / TSD / dist-to-crossing。

---

## 6. 评测与失败模式

- **Acc vs Diversity**：过度模仿导致坍缩；过度多样导致不安全。  
- **TTC / 碰撞**：纵向风险是否及时降速。  
- **Reward hacking**：轨迹「看起来像」但规则违例——用 STL 与一致性 reward 对冲。  
- **Hz**：规划头必须满足 3B@5–8Hz / 7B@3Hz 等产品目标。

---

## 7. 检查清单

- [ ] 导航场与灯态是否级联进自车轨迹头（避免信息绕过）  
- [ ] DiT 条件是否含 ego/time，噪声是否区分 ego/obj  
- [ ] rollout 时域是否与控制周期对齐（如 0.5s×4）  
- [ ] 是否同时监控 Acc、Diversity、规则违例率  
- [ ] 损失权重是否导致 PP/辅助头被淹没  
