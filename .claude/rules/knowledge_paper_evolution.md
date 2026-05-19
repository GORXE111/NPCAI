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

---

## 2026-04-28 / Stage 1 Persona 测试发现的数据格式问题

### 背景
Stage 1 LoRA Val 1.0858 看起来很好。跑了 10 场景 base vs LoRA 对比测试。

### 发现
**Val Loss 低 ≠ persona 学得好**。LoRA 出现 4 类问题：

1. **Metadata 泄漏**: 模型生成时输出 `[scene: [Action/Check: Condition: ...]]` 这种训练数据格式标签
2. **角色破裂**: "I'm just a *computer* -- I can't speak human language"（致命）
3. **第三人称混入**: "He nods at the bottles" 而不是 Kim 第一人称
4. **缺少招牌台词**: "Let me make a note of that" 完全没出现

### 根本原因
DE 原始 `output.json` 中 Kim 的 `dialogue` 字段**混合内容**：
```json
{
  "actor": "Kim Kitsuragi",
  "dialogue": "\"It's done.\" The lieutenant wipes his brow. [Action/Check: ...]"
}
```

我们 prepare_kim_data.py 把整段直接作为 assistant target，模型学会把：
- 引号内对话 + 第三人称动作描写 + metadata 标签
- 都当作"Kim 应该输出的内容"

**Val Loss 1.08 测的是 token-level perplexity，没法发现这种"格式污染"**。

### 假设 vs 现实
- **假设**: Val Loss 低 → persona 学得好
- **现实**: Val Loss 只反映"分布拟合"，不反映"输出可用性"

### 影响
1. **Stage 1 数据需要重做**：从 Kim 的 dialogue 字段里只提取**引号内直接对话**作为 target
2. **关于 metadata 标签**: 应该保留在 prior context 里，**绝不在 assistant target 里**出现
3. **训练目标格式必须明确**: 我们要的是 Kim **说**的话，不是 Kim **的全部行为描述**
4. **Stage 2 设计要前置考虑**: JSON schema 强制把 dialogue 和 tool_calls 分开，应该能从根本上避免这类问题

### 修复策略
重写 `prepare_kim_data.py`：
```python
def extract_kim_speech(dialogue):
    \"\"\"Extract only quoted speech from Kim's dialogue field.\"\"\"
    quotes = re.findall(r'\"([^\"]+)\"', dialogue)
    return ' '.join(quotes) if quotes else None  # skip if no speech

# Filter: keep only Kim turns where there's actual speech
samples = [s for s in raw_samples if extract_kim_speech(s.kim_line)]
```

预计这会把 1,587 → ~1,200 条（去掉纯叙述的 Kim 行）。
然后重训 Stage 1 v2。

### v2 重训结果（2026-04-28）
- **数据**: 1,127 条纯 Kim 引语（vs v1 的 1,587 含污染）
- **训练**: 同保守超参（LR 5e-5, dropout 0.15, r=16, 4 epochs）
- **Best Val**: **0.9445** (-13% vs v1 的 1.0858)
- **训练时长**: ~55 分钟
- 4 epoch 持续下降，未过拟合

| Epoch | Train | Val |
|-------|-------|-----|
| 1 | (high) | 1.022 |
| 2 | 1.013 | 0.995 |
| 3 | 0.956 | 0.964 |
| 4 | 0.919 | **0.945** ← Best |

**关键洞察**: 数据从 1,587 减到 1,127（-29%），Val Loss 从 1.086 降到 0.945（-13%）。**更少但更干净的数据 = 更好的模型**。这是"质量胜过数量"原则的又一次实证。

### v2 Persona 测试 (10 场景对比)

| 问题类型 | V1 | V2 | 修复率 |
|---------|:--:|:--:|:------:|
| Metadata 泄漏 | 7/10 | **0/10** | ✅ 100% |
| 第三人称错位 | 6/10 | **1/10** | ✅ 83% |
| 角色破裂 | 3/10 严重 | **1/10** | 🟡 67% |
| 缺 Kim 招牌口头禅 | 10/10 | 8/10 | 🟡 20% |

**V2 通过 Phase 1 准入门槛**：metadata 泄漏 + 第三人称错位 = 完全解决。
剩余 2 类瑕疵（modern break test 偶发、口头禅密度低）留给 Stage 2/3 修复。

**论文价值**: V1 vs V2 对比是论文 §B Appendix 的"data cleanup matters"案例。

---

## 2026-04-28 / Stage 2 训练 + 测试完成

### 背景
Stage 1 v2 通过后立即用 1,127 条 Stage 2 数据继续训练（`PeftModel.from_pretrained(..., is_trainable=True)` 接续 Stage 1 LoRA）。

### 训练结果
- **Best Val: 0.7246** (4 epochs，持续下降)
- Train: 1064, Val: 63
- 训练时长: ~75 分钟 on M4
- LoRA size: 4.3MB

### 测试结果（7 场景定性测试）
- **JSON schema validity: 100%** (7/7)
- **Persona drift partial**: "Calvert Junction" / "specialize in visual crimes" 等幻觉细节
- **Tool selection accuracy: ~25%** (1/4 期望调用触发)
  - ✅ show_character 在新角色入场时触发
  - ❌ skill_check 在 evidence/empathy 场景没触发
  - ❌ present_choices 在 branch 场景没触发

### 假设 vs 现实
- **假设**: Val 大幅下降 → 模型学会工具调用
- **现实**: 模型学到了 **format**（JSON 100%）但 **content selection** 弱（25%）
- **这正符合 SFT 的本质**: SFT 教 form，不教 correctness（见 knowledge_methodology.md §1.2）

### 这是论文期望的结果
**SFT form learned, RL correctness needed** 是论文核心叙事。Stage 2 数字做出来后正好印证了三阶段设计的必要性：

```
Stage 1: persona ↑                    | tools 0%
Stage 2: JSON valid 0% → 100%         | tool selection still ~25%   ← here
Stage 3 (DPO): tool selection → 80%+  | (核心 contribution)
```

§7 ablation 表会非常有说服力：每个 stage 各自解决一类问题。

### 影响
1. **Stage 2 完成签收**: L2 产出 ✅
2. **Stage 3 DPO 必须做**（验证了它的必要性）
3. **DPO 偏好对应该针对实测的失败模式**:
   - F-skill: skill_check 漏调（应该调没调）
   - F-choice: present_choices 漏调
   - F-persona: 关于自己的事实幻觉（Calvert Junction 等）
4. **不是所有 Stage 2 失败都是 bug**：empty tool_calls 在 small_talk 上是正确的，模型已学到这个

### 下次注意
1. **Stage 2 测试除了 valid JSON 还要测 tool semantic correctness** —— 工具调对了吗
2. **Stage 3 DPO 偏好对要覆盖每个 Stage 2 实测失败模式**，不要凭空设计
3. **Persona hallucination 也是 DPO 偏好对类别**（chosen=正确事实, rejected=Calvert Junction 之类）

### 已交付
- ✅ L2: `kim-q35-08b-stage2.lora` 4.3MB
- ✅ Stage 2 测试结果 JSON: `data/disco_elysium/stage2_tool_test.json`
- 📊 Phase 进度: 9/19 → **10/19 (53%)**

---

## 2026-05-06 / Stage 3 DPO v1 失败：Distributional Collapse

### 背景
按 methodology §3 计划，生成 2,789 条合成偏好对（9 种扰动），DPO 训练 1 epoch。

### 训练数字（看起来很美好）
- Step 662 (25%): Val Acc 98.57% (chosen > rejected) ← Best
- Step 2648 (100%): Val Acc 97.86%
- 训练时长: ~5 小时

### 实际 generation 测试（非常糟糕）

7 case 测试 Stage 2 vs Stage 3：

**正例 (应调工具)**:
- Stage 2: 1/4 (25%)  
- Stage 3: 0/4 (**0% — 倒退**)

**负例 (应空工具)**: 双方 100%

**典型失败**: `new_arrival` 场景，Stage 2 正确触发 `show_character`，Stage 3 改成空工具列表。

### 假设 vs 现实
- **假设**: DPO 训练 Val Acc 98% → tool selection 大幅提升
- **现实**: DPO 让模型 **过度保守**，倾向于不调工具
- **核心**: Val Acc 在偏好数据上 ≠ generation 行为质量

### 根因
合成偏好对**严重偏向"少调工具更好"**:

| 教 "remove tool" | 数量 | 占比 |
|-----------------|:----:|:---:|
| F4 出戏工具 | 526 | 19% |
| F6 多余工具 | 503 | 18% |
| F7 顺序错（chosen 顺序对，rejected 加 end_scene 在前）| 486 | 17% |
| F8 重复 | 261 | 9% |
| **小计** | **1,776** | **64%** |

教 "add tool when needed" 的扰动类型: **0 条**

模型完美学到"减少工具调用是好"，最终学成"什么都不调"。

### 修复策略 (DPO v2)
新增 F0_missing_tool + F0b_partial_drop 共 **724 条平衡对**：
- chosen = 原 sample（有正确工具）
- rejected = 同 sample 但 `tool_calls: []` 或 缺失部分工具

新分布：
- "Add tool when needed" (F0+F0b): 724 (20.6%)
- "Remove unnecessary tool" (F4+F6+F7+F8): 1,776 (50.5%)
- "Format/persona/skill correctness" (F1-F3, F5, F9): 837 (23.8%)

### 影响
1. **Stage 3 v1 不能用** —— 比 Stage 2 还差，论文不能用
2. **Stage 3 v2 训练中** (PID 62316, 3337 对, 预计 2.5 小时)
3. **论文价值新增**: §B Appendix "DPO data balance matters" 案例

### 下次注意
1. **DPO 偏好数据必须平衡**: "教什么时候做" + "教什么时候不做" 都要有
2. **Val Acc on preference pairs ≠ generation quality**: 必须用真实 generation 验证
3. **检查偏好数据 distribution**: 64% 同方向就是危险信号
4. **Distributional collapse 在 0.8B + 1 epoch 都会发生**，β=0.1 不足以 prevent，需要数据平衡

---

## 2026-05-11 / Stage 3 DPO v2 完成，发现 7-case 测试不足以判断

### 训练结果
- 3,337 平衡偏好对训练 1 epoch（~5.5 小时）
- Best Val Acc 96.59%（v1 是 98.57%，但 v1 是 collapse）
- Val Loss 0.234 (final)

### Generation 测试 (7 cases)

| 配置 | Positive cases (4) | Negative cases (3) | Total |
|------|:-----------------:|:------------------:|:------:|
| S2 | 1/4 (25%) | 3/3 (100%) | 4/7 (57%) |
| S3 v1 | 0/4 (0%) collapse | 3/3 | 3/7 (43%) |
| **S3 v2** | **0/4 (0%) 但 try to call** | 3/3 | 3/7 (43%) |

### 假设 vs 现实
- **假设**: 添加 F0_missing_tool 数据 → tool selection 提升
- **现实**: v2 vs v1 行为变了（开始尝试调工具）**但选错工具**
- **本质问题**: 7-case 测试集太小，3 positive case 不足以下结论

### 行为变化的细节
- v1: collapse — 几乎不调工具
- v2: chaos — 调工具但常调错
  - suspect_lying → 调了 show_character（应该是 Empathy skill_check）
  - investigation_branch → 调了 show_character（应该是 present_choices）

这暗示数据平衡**有效消除了 collapse**，但**精度还不够**。需要更大的测试集才能判断。

### 改变的决策
**不再用 7-case 微测试评估**。立刻构建 **DEBench** 正式测试集：
- 50+ 场景 × 多维度（persona / skill-tool / game-tool）
- ground-truth 工具调用标注
- 精度 / 召回 / F1 / multi-judge persona 评分

### 还有 persona 幻觉问题
S3 v2 输出 "A detective in Geneva"，Geneva 不在 DE 世界。需要在 DEBench 中加 **persona fact** 评估。

### 下次注意
1. **小测试集（<20 case）只能做 sanity check，不能下结论**
2. **正式评估必须有 ground truth + 多维度 + 统计显著性**
3. **DPO 的真实效果在 7-case 看不出来**，必须放到更大测试集
4. **Persona drift 需要单独追踪**：训 LoRA 时容易副作用

---

## 2026-05-11 / DEBench v1 设计 + 5 配置评估启动

### 背景
7-case 测试集太小判不了 S3 v1/v2 是否真的提升。立即构建正式 benchmark。

### DEBench v1 设计
**130 个场景，分 3 个 sub-benchmark**：

**A. Persona (50 scenarios)** — 测 Kim 声音的一致性
- 涵盖 DE 风格的多样化情境：crime scene / confession / introspection / break tests
- 每个场景有 rubric（不是 ground truth dialogue，是判断标准）
- 评分: JSON valid rate / no_break rate（关键词检测）/ brief rate
- **更复杂的 multi-judge LLM panel** 留作 Phase 4

**B. Tool Selection (50 scenarios)** — 测工具调用准确性
- 6 个类别：evidence (5), social (5), scene (5), branch (5), emotion (5), combined (3), filler (22)
- 每个有 expected_tools 集合 + expected_skills/actors 候选
- 评分: precision/recall/F1 on tool name + skill_arg_acc + actor_arg_acc

**C. Tool Suppression (30 scenarios)** — 测什么时候不调工具
- 纯闲聊 / 元 prompt / 简单问候
- 评分: empty_rate（应该为 0 的比例）

### 评估配置
5 个配置同步对比：
- `base` — Qwen3.5-0.8B 无任何 LoRA
- `stage1` — + Stage 1 v2 LoRA
- `stage2` — + Stage 2 LoRA
- `stage3_v1` — + DPO v1 (collapse 版本)
- `stage3_v2` — + DPO v2 (balanced)

### 评估在跑
- PID 64491
- 5 configs × 130 scenarios = 650 generation
- 预计 60-90 分钟

### 等到结果出来要看什么
这是论文 §7 的核心数据：

1. **Stage 1 vs Base**: persona 是否真的提升
2. **Stage 2 vs Stage 1**: JSON validity + tool 召回是否提升
3. **Stage 3 v1 vs Stage 2**: 是否真的 collapse（应该可以看到 tool recall 大降）
4. **Stage 3 v2 vs v1**: F0 balancing 是否管用
5. **Stage 3 v2 vs Stage 2**: DPO 总体是提升还是倒退 — **论文核心 claim**

### 决策树
- 如果 **S3 v2 vs S2 在 tool F1 显著提升 (+15pp+)** → 进 Phase 4 (CPDC + 论文写作)
- 如果 **持平或略升** → 补 Phase 3 Stage B 真实分布数据
- 如果 **倒退** → 用 Stage 2 作为最终版，论文重定位为"SFT alone is enough" 反向叙事

### 已交付
- ✅ L6: 3,337 balanced DPO 偏好对（含 F0_missing_tool 修正）
- ✅ L9: DEBench v1 (130 scenarios)
- ✅ L10: 评分脚本 `run_debench.py`
- 🟡 L3v2: Stage 3 v2 LoRA（DPO Val Acc 96.59%）
- 🟡 L10b: 5 配置评估（跑中）

### Phase 进度: 10/19 → **13/19 (68%)**

---

## 2026-05-11 / DEBench 评估结果：Stage 2 数据是根本瓶颈

### 5 配置 DEBench 完整结果

| Metric | base | stage1 | stage2 | s3_v1 | s3_v2 |
|--------|:----:|:------:|:------:|:-----:|:-----:|
| Tool Precision | 0.000 | 0.000 | 0.000 | 0.000 | **0.250** |
| Tool Recall | 0.000 | 0.000 | 0.000 | 0.000 | **0.038** |
| Tool F1 | 0.000 | 0.000 | 0.000 | 0.000 | **0.066** |
| Suppress Empty | 0.967 | 0.000 | 1.000 | 1.000 | 1.000 |
| JSON Valid | 0.940 | **0.120** | 1.000 | 1.000 | 1.000 |
| No-Break | 0.980 | 1.000 | 1.000 | 1.000 | 1.000 |

**Per-category tool F1** (50 scenarios across 6 categories):
- 只有 **scene 类别有效**：s3_v2 P1.0 / R0.4 — 学会了"新角色出现 → show_character"
- evidence / social / branch / emotion / combined / filler: **全 0**

### 假设 vs 现实
- **假设**: Stage 3 DPO 把 tool selection 25% → 80%+
- **现实**: Stage 2 实际就 0%，DPO v2 推到 3.8%，离 80% 差 21倍

### 根因诊断（深度分析）

**Stage 2 训练数据的工具触发模式 ≠ DEBench 评估的触发模式**：

训练时 `skill_check` 触发规则：
- **前置触发**：Skill actor (如 "Empathy") 先说话，Kim 之后回应

DEBench 期望的触发规则：
- **后置触发**：Detective 问 Kim 一个问题（"读他的脸"/"看现场"），Kim 应该**主动**调用工具

→ **模型学到的是"模仿训练数据里已经标注的工具"**，没有学会"从对话意图推断该调什么工具"。

类似失败模式：
- `present_choices` 训练触发：玩家选项跟在 Kim 之后；DEBench 触发：玩家问"该怎么办"
- `set_expression` 完全没在训练数据出现（推断而来）
- `skill_check` 训练分布：Inland Empire/Interfacing/Visual Calculus 都有，但模型没学到"什么场景该调什么"

唯一有效 = `show_character`，因为训练触发（新角色出现）和 DEBench 触发（新角色出现）**模式一致**。

### 影响
1. **Stage 1 v2 / Stage 2 / Stage 3 v1/v2 全部 LoRA 在 tool selection 上 ≈ 等效**（除了 show_character）
2. **不能直接发论文** — F1 0.066 远低于 14B 系统的 ~80%
3. **数据是核心瓶颈**，不是模型 / DPO / 超参问题
4. **Stage 1 LoRA 还破坏了 JSON 输出** (94% → 12%)，这是另一个 bug

### 决策：方案 A — 重做 Stage 2 数据

**核心改造**：让 Kim 学会"从 Detective 提问意图推断工具调用"

新数据触发规则：
- "Look at X" / "What do you see" → `skill_check(Visual Calculus / Perception / Logic)`
- "Is he lying?" / "Read his face" → `skill_check(Empathy / Authority)`
- "What's our next move?" / "How to handle?" → `present_choices`
- "Examine the lock" → `skill_check(Interfacing / Hand/Eye Coordination)`
- "What does this say?" → `skill_check(Encyclopedia / Logic / Rhetoric)`
- 等等

目标：3-5K 高质量主动触发训练样本。

### 下次注意
1. **训练数据触发模式必须匹配评估触发模式** — 不然 benchmark 看上去再好都没用
2. **空工具样本占比要谨慎**: Stage 2 v1 有 47% 空工具样本 → 模型学会"默认空"
3. **Per-category 评估优于 overall** — 暴露 scene-only 这种局部成功
4. **Val Loss / Val Acc 在 SFT 和 DPO 都不能预测 generation 质量** — 真实 benchmark 才是 ground truth
5. **训 LoRA 前先验证生成模式**: Stage 1 应该测一下输出 JSON 看会不会破坏 format

---

## 2026-05-11 / Stage 2 v3 数据重构（intent-driven）

### 背景
DEBench 显示 Stage 2 v1/v2 触发模式错位：训练数据靠"前置触发"（skill actor 先说），DEBench 要求"主动触发"（从 Detective 问题推断）。

### 数据策略：Skill Mining
新方法：每个 DE 语料里的 skill actor 行（如 `Empathy: "His eyes flicker"`）都是 ground truth tool fire 点。把这些**重新组织**：
- INPUT: 移除 skill actor 行的 context + Kim 的回应
- OUTPUT: `skill_check(skill=该 skill 名)` + Kim 对话

模型被迫**从 Detective 的提问推断**该调什么 skill，因为 skill actor 不再出现在 context 里。

### 数据规模
- 487 train + 30 val（vs v2 的 1064，更少）
- 但**多样性更高**:
  - 22 个 DE 技能均衡分布（top: Rhetoric 26, Inland Empire 21, Encyclopedia 13, Authority 12 ...）
  - 35% skill_check / 31% show_character / 30% empty / 4% present_choices

### 关键改造
1. **`exclude_skills=True`** when building context for skill_check samples
2. **Garbage filter**: 过滤 task state markers (`auto.X.Y`, `TASK.X`)
3. **Balanced sampling**: 70% positive + 30% negative

### 训练已启动
PID 65077，预计 30-50 分钟

### 期望结果（vs v2）
- skill_check 触发率从 0% → 30%+
- present_choices 触发率从 0% → 20%+
- 其他工具维持或提升

### 下次注意
1. **训练数据触发模式 = 评估触发模式** 是 NPC 工具调用的硬约束
2. **Skill mining 是高效率数据生成策略**（用语料自带信号，零标注成本）
3. **Garbage filter 必须早做**: training data 里的 `auto.X.Y` 类标记会污染 generation

---

## 2026-05-14 / 撞墙：DPO 第三次 collapse，需要战略转向

### 完整实验矩阵（7 个系统）
| 系统 | Tool F1 | Suppress | 问题 |
|------|:-------:|:--------:|------|
| Stage 2 v1 | 0.000 | 1.00 | 啥也不调 |
| Stage 2 v2 | 0.000 | 1.00 | 啥也不调 |
| Stage 3 v1 DPO | 0.000 | 1.00 | collapse |
| Stage 3 v2 DPO | 0.066 | 1.00 | 几乎不调 |
| **Stage 2 v3** | **0.105** | 0.27 | 过度调用（best F1）|
| Stage 2 v3.1 | 0.069 | 0.90 | 过度保守 |
| Stage 3 v3 DPO | 0.000 | 1.00 | **又 collapse** |

### 假设 vs 现实（累计 3 次）
- **假设**: DPO 用平衡偏好对能修复 tool selection
- **现实**: DPO 三次（v1/v2/v3）都 collapse 到"啥也不调"
- **根因**: 偏好数据**净方向**总是偏向"少调工具"
  - v1: 64% "remove tool"
  - v3: F_overcall 36% vs F0 18% = 2:1 偏向 empty
- **更深**: 0.8B + DE 语料无法泛化主动工具选择到 DEBench 分布

### 关键认知转变
**DPO 不是这个问题的解药**。即使数据完美平衡，0.8B 在有限领域数据上学不会泛化的 proactive tool calling。

best 结果 = Stage 2 v3 SFT 的 F1 0.105，离论文需要的 0.5+ 差 5 倍。

### 决定性对照实验
必须测 **Base Qwen3.5-0.8B + few-shot prompting** on DEBench：
- base+fewshot >> LoRA → 论文转向 "ICL > fine-tuning at 0.8B for NPC tools"
- base+fewshot 也烂 → limits study，0.8B 天花板
- 决定整个论文方向

### 论文价值（即使转向）
这 7 个失败实验本身是 contribution：
1. **DPO collapse pattern**（3 次复现）→ 方法论警示
2. **数据触发模式失配** → benchmark 设计教训
3. **质量胜数量**（多次验证）
4. **0.8B SFT/DPO 在工具泛化上的实证天花板**

### 下次注意
1. **DPO 偏好数据方向偏差是反复踩的坑** — 必须严格 1:1 平衡 add/remove
2. **撞墙时先做 cheap 对照实验**（base+fewshot 30 分钟）再决定大方向，别一直堆训练
3. **训练 7 个系统才意识到要测 base+fewshot** — 应该第一天就做这个 baseline
4. **DPO checkpoint 按 Val Loss 存而非 Val Acc** — Acc 在 noisy 偏好数据上不动，Loss 才反映学习

---

## 2026-05-15 / 战略转向：放弃 0.8B，切 Qwen3.5-2B 从 0 重做

### 背景
0.8B 上 7 个系统全部 F1 ≤ 0.105，DPO 3 次 collapse。调研确认 DeepSeek V4 无 <3B 小模型（284B 起步），DeepSeek 路线断。

### 决策
**全面切 Qwen3.5-2B，从 Stage 1 重新开始。**

理由：
1. **DeepSeek V4 排除**：最小 V4-Flash 284B，M4 训不了；唯一 sub-3B 是 R1-Distill-Qwen-1.5B（math/think 导向，不适合 NPC）
2. **Qwen3.5-2B 是唯一现实升级**：有 `qwen35_mps_fix.py` patch，SFT 能在 M4 跑（~12GB float32）
3. **0.8B 可能是容量天花板**：base+fewshot 实验未跑完（M4 太慢），但 7 个训练系统全失败强烈暗示容量不足
4. **2B 已在本对话验证可训**（之前 Stage 2/3 v0 尝试加载成功，dtype patch 有效）

### 复用资产（不浪费前期工作）
- Stage 1 数据: `data_kim_v2/kim_train.jsonl` (1,127 cleaned Kim)
- Stage 2 数据: `data_kim_v3_1` (intent-driven 50/50, 677) — 之前最好的 Stage 2 数据
- Stage 3 DPO: `data_kim_dpo_v3` (1,861 balanced + F_overcall)
- 所有 prepare/train 脚本（改 MODEL_NAME 即可）

### M4 上 Qwen3.5-2B 的已知约束
- 2B float32 ≈ 8GB 权重 → SFT LoRA 跑得动（~12GB，会 swap 但能完成）
- **DPO 跑不了**: ref+policy 双模型 = 16GB 仅权重，装不下 → Stage 3 可能要：
  - 跳过 DPO（纯 SFT），或
  - 用 reference-free 方法（ORPO/SimPO，单模型），或
  - 上 H100

### H100 备选已评估
- 单卡 H100 80GB：全参 ≤7B / LoRA ≤20B / QLoRA ≤70B
- 能跑正经 GRPO（CPDC 第一名方法）+ 2-9B + 双模型 DPO
- 云租 ~$2-3/h，我们训练量分钟级
- **若 2B SFT 仍不达标 → 上 H100 是明确下一步**

### 当前状态
- Qwen3.5-2B Stage 1 训练中（PID 2717, 1.88B params 确认, 1,064 train）
- print 标签 "Qwen3.5-0.8B" 是硬编码遗留字符串，实际模型是 2B（param count 已验证）

### 下次注意
1. **撞墙时尽早做 cheap baseline + 调研可选项**，别堆 7 个失败训练才转向
2. **print 标签用变量不要硬编码**，避免误判
3. **2B DPO 在 M4 不可行** — 提前规划 H100 或 reference-free
4. **0.8B → 2B 是这个 paper 的关键 scale 实验**，结果本身有论文价值（"NPC tool-calling 的最小可行 SLM size"）

---

## ⏯️ 恢复点 (2026-05-16, 关机后续作)

### Mac 网络注意
- `huggingface.co` 不通时，训练脚本会因 tokenizer 联网检查失败
- **解法**: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 前缀（缓存已全，无需联网）
- HF 缓存位置: `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B`（齐全）

### 当前任务流（Qwen3.5-2B pivot）
```
🟡 Stage 1 (Qwen3.5-2B)  训练中 PID 1542
   脚本: ~/npcllm/model/train_kim_2b_s1.py
   数据: ~/npcllm/data_kim_v2/kim_train.jsonl (1064)
   输出: ~/npcllm/checkpoints/kim_q35_2b/lora
   日志: ~/npcllm/2b_s1_train.log
   上次中断在 E3 Step532 Val 0.9167（已重启跑完整 4 epochs）

⏳ Stage 2 (待做): 数据 ~/npcllm/data_kim_v3_1/ (intent-driven 50/50, 677)
   需创建 train_kim_2b_s2.py（改 train_stage2_v3.py 的 MODEL_NAME + 接 2b s1 lora）

⏳ Stage 3 (待定): M4 跑不了 2B DPO → 选项: 跳过/ORPO/H100

⏳ DEBench: run_debench.py 加 kim_q35_2b 配置后评估
```

### 关键文件
- 训练脚本模板都在 `~/npcllm/model/`，改 MODEL_NAME + 路径即可
- DEBench: `~/npcllm/benchmarks/debench_v1.json` + `run_debench.py`
- 所有 0.8B negative result checkpoint 已存 GitHub

### 期望
2B Stage 1 Val 应 ≤ 0.9（0.8B 是 0.945）。Stage 2 后跑 DEBench 看 tool F1 是否突破 0.8B 的 0.105 天花板。

### Stage 1 结果 (2026-05-16 完成)
- **Best Val 0.8886**（vs 0.8B v2 的 0.945, -6%）
- 4 epoch 持续降，无过拟合
- Persona 测试：
  - ✅ in-character 质量明显优于 0.8B（"I'm Kim Kitsuragi -- lieutenant, RCM Precinct 41"，无 Geneva/Calvert 幻觉）
  - ✅ 程序化语气地道（anti_authority/introspection 完美）
  - 🔴 break-test 仍失败 2/10（"It's 2025" / "I am a computer program"）—— base 指令跟随泄漏，已知难点
- 结论：可接受，进 Stage 2

### Stage 2 启动 (2026-05-16)
- PID 28227，train_kim_2b_s2.py
- 接 Stage 1 LoRA + data_kim_v3_1（intent-driven 50/50, 677 train）
- 关键实验：2B 能否突破 0.8B tool F1 0.105 天花板
- 输出: ~/npcllm/checkpoints/kim_q35_2b_stage2/lora

### 下次注意
1. **任何对 LLM 输出有形式要求的训练**: SFT 前必须先验 1-2 条样本生成是否符合预期
2. **Val Loss 是必要不充分条件** —— 必须配合定性 generation 测试
3. **DE 数据特别**: dialogue 字段不是纯对话，提取时要严格筛选
4. **Stage 2 JSON schema 是更根本的解药**: 强制 dialogue/tool_calls 分离
