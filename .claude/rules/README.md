# NPCAI 项目规则与知识库

这里存放项目的过程规则（怎么做研究）和知识库（做了什么、学到什么）。

## 文件索引

| 文件 | 类型 | 内容 |
|------|------|------|
| [retrospective_process.md](retrospective_process.md) | 过程规则 | 每次研究后如何复盘、如何更新知识库 |
| [knowledge_deliverables.md](knowledge_deliverables.md) | **产出清单** | 论文 claim → artifact 映射、19 项交付物、4 阶段 roadmap、风险登记表 |
| [knowledge_methodology.md](knowledge_methodology.md) | **方法论** | SFT / 5 种记忆 / DPO / Stage 3 偏好对生成深度 |
| [knowledge_npc_research.md](knowledge_npc_research.md) | 知识库 | 2025-2026 benchmark 景观、并行工作、Gap Analysis |
| [knowledge_paper_evolution.md](knowledge_paper_evolution.md) | 知识库 | 论文方向 v0 → v1 → v2 → v3 的演化记录和教训 |
| [knowledge_training_gotchas.md](knowledge_training_gotchas.md) | 知识库 | MPS dtype / LoRA 过拟合 / 数据质量 / 内存限制 等技术坑 |

## 新人/新会话使用方式

1. **第一步先读 `knowledge_paper_evolution.md`** —— 了解论文当前方向和为什么是这个方向
2. **查 `knowledge_deliverables.md`** —— 看现在该做什么、整体进度、最低交付门槛
3. **看方法论查 `knowledge_methodology.md`** —— SFT / 记忆 / DPO 概念深度
4. **看 benchmark 查 `knowledge_npc_research.md`** —— CPDC 2025、BFCL V4、并行工作
5. **遇到技术问题查 `knowledge_training_gotchas.md`** —— 90% 的 MPS/LoRA 问题都有记录
6. **完成一轮工作后按 `retrospective_process.md` 更新知识库**

## 当前论文方向（2026-04-24 起）

**v2: "Tool-Using Small Language Models for Interactive Game NPCs"**

核心：让 Qwen3.5-0.8B 在 M4 Mac 上通过 Unity 游戏引擎 API 工具（play_animation, show_item 等）"行动"，对标 CPDC 2025 获奖的 Qwen3-14B + L40S 方案。

Pareto 位置：17× 参数更少、消费硬件、真实游戏引擎绑定。
