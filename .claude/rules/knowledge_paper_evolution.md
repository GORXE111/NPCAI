# 论文方向演化记录

记录每次方向变更的起因、依据、得失。避免绕回老路。

---

## 2026-04-24 / v0 → v2 大复盘

### 背景
从项目启动到现在，论文方向经历了 3 次重大调整。在用户要求"ultrathink"后、研究了 2025-2026 最新 benchmark 后，方向终于收敛到可发表路径。

### 方向 v0（项目启动 ~ 2026-04-20 左右）
**论题**: "TownAgent: Memory-Augmented SLMs with Emotion-Aware Generation for Multi-NPC Social Simulation"

**三大贡献**:
1. LoRA personality adaptation
2. Memory Prefix Injection (MPI)
3. Emotion Head (EH)

**假设**: 在 Qwen3.5-0.8B / 2B 上训这三个模块，能得到强于 base 的 NPC 对话系统。

### v0 执行结果
- **Stage 1 LoRA**: Qwen3.5-2B + 278 手写 → Val 0.344（看起来好）；Qwen3.5-0.8B + 8K → Val 1.89（过拟合）
- **Stage 2 MPI**: Val 1.196 → 0.645，Gate 涨到 0.758（看起来学到了）
- **Stage 3 EH**: 8 类情感分类 Val 72.0%（看起来不错）

### 被什么打脸
**严谨 benchmark（盲评 Claude + qwen3.5:9b 多 judge panel）证实**:
- Base Qwen3.5-0.8B: 3.81/5, 70% 合格
- +LoRA: 2.53/5, **0% 合格**（灾难性过拟合）
- +LoRA+MPI: 3.52/5（只是把 LoRA 的伤害拉回到 base 之下）
- Full (+EH): 3.52/5（EH 不改变生成文本）

**结论**: 我们的"改进"全部**不如什么都不做**。负面结果，发不出去。

### 方向 v1（曾短暂考虑）
**论题**: "Memory Faithfulness via DPO for SLM NPCs"

**依据**: benchmark 发现所有配置在 memory_use 维度都 2.85/5，这是真实 gap。

**被否决**: 范围太窄；用户反馈"应该聚焦游戏 NPC 扮演好"

### 方向 v2（当前，2026-04-24 确立）
**论题**: "Tool-Using Small Language Models for Interactive Game NPCs"

**核心主张**: 让 SLM 通过 Unity 游戏引擎 API 工具（play_animation, show_item, move_to 等）真正"行动"，而不仅仅是生成对话文本。

**为什么这条路对**:
1. **学术空白**: 没人做"游戏 NPC 工具调用训练"（通用 tool use 和纯对话 NPC 都有研究，但交集是空的）
2. **有完美 benchmark**: CPDC 2025（EMNLP Wordplay Workshop 2025.11）完全匹配我们的任务
3. **有清晰 Pareto 位置**: CPDC 获奖全用 Qwen3-14B + L40S，我们 0.8B + M4 占 17× 参数 + 消费硬件的空白
4. **BFCL V4 证据**: Qwen3-0.6B 在 agent score 上能到 0.880，说明小模型做 tool use 是 feasible 的
5. **产业界验证**: NVIDIA ACE 产品方向就是这个（但无 peer-reviewed paper）
6. **可 demo**: 真实 Unity 动画/物品互动比纯文字有说服力

### 假设 vs 现实总结

| 假设（v0） | 现实 | 教训 |
|-----------|------|------|
| LoRA 能让 SLM NPC 变强 | 0.8B 上过拟合到 0% 合格 | 小模型不适合小数据 SFT |
| 数据越多越好 | 278 手写 >> 81K 混合 | 数据质量决定性 |
| Memory Prefix 解决记忆问题 | Gate 涨到 0.76 但 memory_use 没变好 | 训练 loss ≠ 行为改善 |
| val loss 越低模型越强 | val 1.89 的模型 benchmark 0% | 必须直接测 benchmark |
| 我们是第一个做 SLM NPC 的 | Fixed-Persona SLMs (arXiv 2511.10277) 并行工作 | 必须查 concurrent work |

### 改变的决策

1. **主攻方向**: 从"记忆+情感架构"改为"工具调用 + 游戏 API"
2. **Benchmark 策略**: 从自建 benchmark 改为"标准 CPDC 2025 + 自建 tool-use benchmark"
3. **训练方法**: 从 LoRA SFT 为主改为 SFT + DPO（参考 CPDC 获奖方案用 GRPO）
4. **保留资产**: Unity 10 NPC 系统 + Memory Prefix（作为 ablation）+ 盲评协议
5. **舍弃**: Emotion Head（对生成无影响）、81K 混合数据、纯对话 benchmark

### 被重复浪费的时间

1. **Qwen3.5-0.8B Stage 1 v1/v2 两次训练** —— 应该在第一次 benchmark 不合格就止损
2. **81K mixed 数据训练** —— 应该先测数据质量才投入训练
3. **Stage 3 Emotion Head 多次调参** —— 对最终目标无贡献
4. **Qwen3.5-2B Stage 2/3 多次失败尝试** —— MPS dtype 问题应该先在小 case 诊断
5. **qwen3.5:9b 做 benchmark judge** —— 同族偏差应该更早意识到

### 下次注意（actionable lessons）

1. **开题前先查 arXiv 最近半年 concurrent work**（我们差点和 Fixed-Persona SLMs 撞车）
2. **先做 benchmark 基线再设计方法**（先测 Base 有多差再决定怎么补）
3. **用多家族 LLM judge 或 blind Claude eval**（避免 judge bias）
4. **val loss ≠ 目标指标**，必须有行为级评估
5. **SFT 前先问: "Base + prompt engineering 够了吗？"**
6. **遇到 MPS dtype 冲突先做 1 样本 minimal case 诊断**
7. **Emotion Head 这类"看起来合理但不改变核心输出"的模块要警惕**

---

## 后续更新

每次方向变更或重大发现，在下面追加新的复盘段落，保留时间戳。不要覆盖历史。
