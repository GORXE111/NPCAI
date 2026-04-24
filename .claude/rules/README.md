# NPCAI 项目规则与知识库

这里存放项目的过程规则（怎么做研究）和知识库（做了什么、学到什么）。

## 文件索引

| 文件 | 类型 | 内容 |
|------|------|------|
| [retrospective_process.md](retrospective_process.md) | 过程规则 | 每次研究后如何复盘、如何更新知识库 |
| [knowledge_npc_research.md](knowledge_npc_research.md) | 知识库 | 2025-2026 benchmark 景观、并行工作、Gap Analysis、Roadmap |
| [knowledge_paper_evolution.md](knowledge_paper_evolution.md) | 知识库 | 论文方向 v0 → v1 → v2 的演化记录和教训 |
| [knowledge_training_gotchas.md](knowledge_training_gotchas.md) | 知识库 | MPS dtype / LoRA 过拟合 / 数据质量 / 内存限制 等技术坑 |

## 新人/新会话使用方式

1. **先读 `knowledge_paper_evolution.md`** —— 了解论文当前方向和为什么是这个方向
2. **查 `knowledge_npc_research.md`** —— 看 benchmark 选择、concurrent work、可复用资产
3. **遇到技术问题先查 `knowledge_training_gotchas.md`** —— 90% 的 MPS/LoRA/数据问题都有记录
4. **完成一轮工作后按 `retrospective_process.md` 更新知识库**

## 当前论文方向（2026-04-24 起）

**v2: "Tool-Using Small Language Models for Interactive Game NPCs"**

核心：让 Qwen3.5-0.8B 在 M4 Mac 上通过 Unity 游戏引擎 API 工具（play_animation, show_item 等）"行动"，对标 CPDC 2025 获奖的 Qwen3-14B + L40S 方案。

Pareto 位置：17× 参数更少、消费硬件、真实游戏引擎绑定。
