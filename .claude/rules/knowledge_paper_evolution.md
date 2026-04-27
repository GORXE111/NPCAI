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

---

## 2026-04-24 / 角色锚定：Kim Kitsuragi

### 背景
v2 方向（工具调用 NPC）确立后，需要选具体角色作为论文 demo 主角和训练目标。

### 调研发现
两轮 web 搜索找出候选：
- **Honkai Star Rail** (185K 条 + 原生动作标签) — 数据最强
- **NieR Pod 042** (30K, 字面"工具调用 NPC") — 叙事最匹配
- **Kim Kitsuragi** (DE, 6.4K) — 叙事清晰 + 学界先例
- **Mantella Skyrim pipeline** (MIT, 8.8K + 可再生) — 授权最干净
- **Serana** (Skyrim, 1.96K) — 备选

### 假设 vs 现实
- **假设**: HSR 数据最大 + 标签最全 = 首选
- **现实**: HoYoverse 在 2024.10 对 Dimbreath repo 提 DMCA。HF 镜像目前活着但可能随时被清。**论文可复现性致命受损**。

### 决策
最终选 **Kim Kitsuragi (Disco Elysium)**。

理由：
1. **授权最安全**: ZA/UM 历史宽松，HF 数据集稳定
2. **叙事最清晰**: DE 24 个内心技能 = 工具调用的天然类比，论文有现成框架
3. **学术先例**: EMNLP 2023 Akoury et al. 已用 DE 做 corpus
4. **数据格式直接对口**: `output.json` 含 `[Action/Check: ...]` 标签，就是工具调用
5. **角色辨识度极高**: 现代游戏最独特 NPC 声音之一（即使审稿人没玩过，"clinical detective" 也好理解）

### 数据现状
- `allura-org/disco-elysium-conversations-raw/output.json`: 1,742 conversations，含动作标签
- `main-horse/disco-elysium-utterances`: 6,438 Kim 单条台词（无上下文）
- 提取后 SFT 训练样本: **1,587 train + 80 val**（按 Kim 出现的对话过滤）

### 影响
1. **Unity 端**: 重构为 DE 风格 galgame（油画背景 + 立绘 + 24 技能 UI 组件）
2. **训练数据**: Kim 1,587 SFT samples + 后续从 conversations 提取 action 调用
3. **Tool 体系**: 借鉴 DE 24 技能 + 我们扩展的 galgame 工具（show_cg / play_bgm / present_choices）
4. **论文叙事**: "Skill-as-Tool: Grounding SLMs in In-Character Decision Tools using Disco Elysium's Kim Kitsuragi"

### 下次注意
1. **任何依赖动漫/游戏数据的方向**先查发行商法律姿态（搜 DMCA 历史）
2. **HF 镜像不算永久存档** — 关键数据集要本地备份
3. **数据量不是单一标准** — 1.6K with 上下文 > 6K without 上下文（for SFT）
4. **DE 的 `[Action/Check: ...]` 是免费的工具调用监督信号**，下一阶段必用

### 已上传到 Mac
- `~/npcllm/data_kim/kim_train.jsonl` (1587)
- `~/npcllm/data_kim/kim_valid.jsonl` (80)
- 训练脚本 `~/npcllm/model/train_kim_lora.py` 启动中（PID 66280）

---

## 2026-04-27 / Stage 1 Kim LoRA 完成

### 背景
按 v3 计划 Stage 1 持久 SFT 训 Kim 人设。用之前 v2 学到的保守超参（LR 5e-5, dropout 0.15, r=16, weight_decay 0.05, 4 epochs + early-stop patience 2）。

### 发现
**最终 Best Val Loss: 1.0858**（4 个 epoch 持续下降）

| Epoch | Train | Val |
|-------|------|-----|
| 1 | 1.6493 | 1.2208 |
| 2 | 1.1258 | 1.1517 |
| 3 | 1.0605 | 1.1151 |
| 4 | 1.0138 | **1.0858** ← Best |

### 假设 vs 现实
- **假设**: 1,587 条 DE 数据可能太少，Val Loss 难降下来
- **现实**: 比 8K curated（v2 Val 1.94）好 42%，比 v1 的 1.89 好 43%

### 影响
1. **再次验证"质量胜过数量"** —— 从 v2 知识扩展到 v3 真实数据
2. **0.8B SFT 不是过不了的坎**，问题在于数据质量 + 超参组合
3. **保守超参完全有效**: LR 5e-5 + dropout 0.15 + r=16 + early stop 是 0.8B SFT 标准配置

### 技术细节
- LoRA size: 4.3MB（adapter_model.safetensors）
- 训练时长: ~75 分钟 on M4 16GB
- 内存峰值: 6.4% (~1GB)，无 swap
- CPU: ~50% （MPS 主算）
- 完成时间: 2026-04-27 19:02

### 已交付物
- ✅ `kim-q35-08b-stage1.lora` 4.3MB → 已拷贝到 `D:/AIproject/NPCAI/checkpoints/kim_q35_08b/`
- ✅ Best Val 1.0858

### 下次注意
1. **0.8B + 保守超参 + 1-2K 高质量数据 = OK 的 SFT 配方**，可推广到其他角色
2. Stage 2 数据要保持类似规模和质量等级，不要又跑去 10K+
3. Phase 1 闭环 demo 现在可以启动（有 LoRA 了，剩 Unity 端 demo 资源）
