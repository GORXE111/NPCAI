---
name: Prompt Engineering Guide
description: 用户提供的提示词写作规范 - 面向行为/协议/失败路径，适用于所有MD文档和工作流设计
type: feedback
---

用户有一份 Prompt Engineering Guide（D:\CLAUDECODE PROJECT\prompt-analysis-bilingual.md），要求后续写 MD 文档和工作流时遵循。

核心原则：
- 先定义边界再定义目标
- 把任务改写成协议化流程（不是愿望）
- 显式覆盖失败路径
- 压制低价值行为
- 明确输出契约
- 区分探索、审批、执行、汇报

**Why:** 用户认为好的提示词/文档是操作规程，不是口号。追求稳定交付而非最大自由度。
**How to apply:** 写任何 MD 文档（论文大纲、实验协议、工作流说明）时，按"协议式"结构组织：定义边界→步骤序列→失败处理→输出格式→验证方式。
