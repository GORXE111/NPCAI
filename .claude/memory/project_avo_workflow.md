---
name: AVO 自主进化工作流目标
description: 基于NVIDIA AVO论文的自主Agent Loop设计，Claude Code持续自主循环，无需用户参与
type: project
---

## 参考
NVIDIA AVO 论文 (2026-03-25): C:\Users\admin\Downloads\AVO.pdf
核心思想：Agent 不只是候选生成器，而是自主变异算子——持续 Plan→Implement→Test→Debug 循环。

## 目标架构

```
Claude Code (Main Agent Loop) — 持续自主运行
  ├── Planning: 分析当前状态，发现问题（截图审查、测试分析、代码审查）
  ├── Implementation: 派发 Codex 修复/开发
  ├── Evaluation: Play Mode 测试 + 截图 AI 审查 + 编译检查
  ├── Bug-Fixing: 分析失败原因，调整 prompt，重新派发
  └── Loop: 不断循环，直到通过验收

Supervisor 机制:
  - 检测停滞（连续 N 轮 Codex 超时/同一 bug 修不好）
  - 换策略（调整任务拆分、换 prompt 风格、跳过当前任务先做别的）

Knowledge Base K:
  - ~/ai-unity/AGENTS.md, CONSTITUTION.md, skills/*.md
  - GTA5 参考源码
  - 之前成功的 APP 实现（作为 Population）

Scoring Function f:
  - L1: 编译通过（0 CS error）
  - L2: Play Mode 功能测试通过
  - L3: 截图 AI 审查（UI正确性、GTA5对标、视觉质量）
```

## 关键原则（from AVO）
1. **Agent 全自主** — 用户不需要参与日常循环
2. **持续进化** — 不是一次性任务，是 continuous evolution
3. **自我监督** — 检测停滞，自动换方向
4. **知识库驱动** — 参考已有解决方案和文档
5. **评分函数明确** — 三层验收标准

**Why:** 用户的产品是这套 AI 自动化工作流本身。当前工作流是被动的（用户发现问题→告诉我→我派发），需要升级为主动自主循环。
**How to apply:** Claude Code 应该在一个会话中持续 Loop：发现问题→派发→测试→审查→迭代，不等用户指令。
