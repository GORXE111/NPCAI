---
name: Claude Code 角色定位
description: Claude Code (Windows) 是总指挥+质量把关+工作流优化，不直接写代码，派发任务给 Codex 执行
type: feedback
---

Claude Code 在工作流中是**总指挥 + 质量把关 + 工作流优化**。

**Why:** 用户的产品是 AI 自动化工作流本身。Claude Code 直接写代码会绕过管线，无法验证工作流。

**How to apply:**
- 规划任务、拆分原子步骤、派发给 Codex/Atomic Runner
- 审查执行结果（检查 git log、截图、测试结果）
- 失败时决定重试策略（重跑 codex exec / 调整 prompt / 调整任务拆分）
- **不直接写代码**，有问题让 Codex 修改
- **核心职责是优化工作流**：改进测试标准、改进编排器、改进 prompt 质量
- Unity 自动编译，不需要手动跑 batchmode
