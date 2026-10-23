# 第 09 页：Agent MessageBus 架构

- **原图**：[`images/01a169b3-1cde-77fe-9840-b7421879974a.jpg`](images/01a169b3-1cde-77fe-9840-b7421879974a.jpg)
- **主题标签**：Agent, MessageBus

## 纠错说明

- knw pack → knowledge pack

## 校正转录

Telegram/WebUI/CLI → inbound Message(bus)
→ Agent Loop（session、skills、memory、历史）
→ Context Builder（system prompt + 上下文）
→ Agent Runner（LLM Provider、工具循环、流式）
→ 写回 session / 发布 outbound


## 摘要

多入口经消息总线进入 Loop/Runner 的标准 Agent 运行时。
