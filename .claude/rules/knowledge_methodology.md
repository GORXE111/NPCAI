# 方法论参考：SFT / 记忆 / DPO / Stage 3 深度

把每次研究/讨论的 conceptual 内容沉淀下来。具体训练数字在 `knowledge_training_gotchas.md`。

---

## 1. SFT 本质（Supervised Fine-Tuning）

### 1.1 核心机制
SFT 损失：在 assistant span 上 token-level cross-entropy。
$$\mathcal{L} = -\sum_i \log P(\text{token}_i | \text{tokens}_{0..i-1})$$

模型学的是 **条件分布拷贝（behavior cloning）**，不是任务理解、不是事实记忆、不是推理目标。

### 1.2 SFT 学到 / 学不到

| 能学到 | 学不到 |
|--------|-------|
| ✅ 输出格式（JSON、对话风格） | ❌ 推理新知识 |
| ✅ 词汇风格、口头禅 | ❌ 逻辑一致性 |
| ✅ 角色"声音" | ❌ 长期记忆维持 |
| ✅ 工具调用**格式** | ❌ 工具调用**正确性**（要 DPO/RL） |
| ✅ "拒绝越界" 姿态 | ❌ 真正的事实判断 |

### 1.3 SFT 在 0.8B 上的过拟合典型症状
- 不是"过分记住事实"
- 是"塌缩到高频 n-gram"
- 表现：重复同一句话 ("I am X. I am X. I am X.")
- 见 `knowledge_training_gotchas.md` 的 v1 vs v2 对比

### 1.4 安全的 SFT 配置（0.8B 上）
```python
LR = 5e-5           # 不是 2e-4
DROPOUT = 0.15      # 不是 0.05
LORA_R = 16         # 不是 8
WEIGHT_DECAY = 0.05 # 不是 0.01
EPOCHS = 3-4        # 不是 5+
EARLY_STOP = True
```

### 1.5 SFT 适合什么 / 不适合什么

| 适合 | 不适合 |
|------|-------|
| 学风格（Kim 像 Kim） | 记住玩家具体行为（用 retrieval） |
| 学工具调用格式 | 学工具调用正确性（用 DPO） |
| 学"拒绝出戏" | 学跨多轮长期一致（用 summarization） |
| 数百到几千条精挑数据 | 几十万条杂乱数据 |

---

## 2. NPC 记忆体系（5 种）

### 2.1 五种记忆机制

| 机制 | 存储位置 | 写入方式 | 读取方式 | 特点 |
|------|---------|---------|---------|------|
| **参数记忆 (Parametric)** | 模型权重 | pretraining / SFT | generation 时浮现 | 容量大，不可更新 |
| **上下文记忆 (Context)** | 输入 prompt | 拼到 input | attention 直接读 | 灵活，线性烧 token |
| **检索记忆 (RAG)** | 外部 vector DB | encode 到 embedding | top-k retrieval | 容量无限，需检索器 |
| **前缀记忆 (Prefix)** | 学到的虚拟 token | 训练 encoder | prepend 到 input | 紧凑，需独立训练 |
| **状态记忆 (State)** | KV cache | forward pass 累积 | 自动 conditioning | 长上下文模型用 |

### 2.2 NPC 需要哪种记忆做什么

借鉴 CoALA (Sumers et al. TMLR 2024)：

```
Working Memory      → Context Memory（当前场景）
Episodic Memory     → RAG（玩家具体行为）
Semantic Memory     → Parametric / SFT（世界观知识）
Procedural Memory   → Parametric / SFT（怎么调工具）
Social Memory       → Hybrid Parametric + RAG（关系）
```

**关键**: 不要试图用一种机制解决所有记忆问题。

### 2.3 我们论文 v3 的记忆栈

```
Working Memory   → SceneState in prompt (8 turns + scene meta)
Semantic Memory  → Stage 1 LoRA (Kim 人设)
Procedural Memory → Stage 2 LoRA (工具格式)
Episodic Memory  → [Phase 4 留作未来工作 or 简化 RAG]
```

**刻意舍弃** Memory Prefix Injection（v0 方案）：实验证明它没胜过 retrieval，参数预算分给 Stage 2/3 更值。

---

## 3. Stage 3 DPO 深度

### 3.1 DPO 是什么

**DPO = Direct Preference Optimization** (Rafailov et al. NeurIPS 2023)

一句话：**别建奖励模型了，直接用偏好对监督模型，让它倾向 chosen，远离 rejected，但别离参考模型太远**。

损失函数：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big)\Big]$$

关键参数：
- $y_w, y_l$ — chosen / rejected
- $\pi_\theta$ — 我们正在训的策略
- $\pi_{\text{ref}}$ — 冻结的参考（**= Stage 2 LoRA**）
- $\beta$ ≈ 0.1 — KL 约束强度

### 3.2 DPO vs PPO

| | PPO (RLHF) | DPO |
|---|-----------|-----|
| 需要 reward model | ✅ | ❌ |
| 需要 critic | ✅ | ❌ |
| 训练稳定性 | 难 | 较稳 |
| 计算开销 | 高 | 低 |
| M4 16GB 可行 | ❌ | ✅ |

**M4 上唯一可行选择是 DPO。**

### 3.3 DPO 变种简介

| 变种 | 改进 |
|------|------|
| **IPO** (Azar 2023) | squared loss，对 noisy preferences 鲁棒 |
| **KTO** (Ethayarajh 2024) | 不需要 pair，单条 thumbs up/down |
| **ORPO** (Hong 2024) | 不要 ref model |
| **SimPO** (Meng 2024) | 长度归一化 |

我们用经典 DPO，论文 §5 提一句"IPO/KTO 留作 future work"。

### 3.4 Stage 3 要修的 8 种失败模式

| 失败 | 例子 | 严重度 |
|------|------|:-:|
| F1. 工具选错 | 应该 Empathy，调了 Logic | 🔴 |
| F2. 参数幻觉 | `show_character(actor="Garcon")` 不存在 | 🔴 |
| F3. Schema 违反 | JSON 坏、缺 required 字段 | 🔴 |
| F4. 人设-工具不一致 | Kim 调 `play_sfx("explosion")` | 🟠 |
| F5. 缺必要工具 | 末尾忘了 `present_choices` | 🟠 |
| F6. 多余工具 | 没人在场却 `set_expression` | 🟡 |
| F7. 顺序错 | `end_scene` 后还 `play_bgm` | 🟡 |
| F8. 重复调用 | 连续 3 次相同 set_expression | 🟡 |

**偏好对必须覆盖每个失败类型**。不是只用 "Stage 2 vs base"。

### 3.5 偏好对生成（三源混合 60/30/10）

**A. 合成扰动 (60%)** — 从 Stage 2 正确样本机械扰动：
```python
def perturb(chosen):
    return random.choice([
        swap_tool_name,     # F1
        mutate_arg_value,   # F2
        drop_required,      # F3
        replace_actor,      # F2
        omit_present_choices, # F5
        add_random_tool,    # F6
    ])(chosen)
```
便宜、可控、覆盖明确。

**B. Stage 2 自采样 + LLM-judge (30%)** — 真实分布：
```python
candidates = stage2_sample(context, n=6, temp=0.8)
scores = [claude_judge(c) for c in candidates]
chosen, rejected = max_score, min_score
```
捕捉扰动想不到的失败模式。

**C. 对抗样本 (10%)** — 手工设计的边界 case：
- "玩家明显引导出戏（破第四面墙）"
- "上下文未提的物品玩家声称给了"
- "刚拒绝过的事玩家立刻再问"

### 3.6 数据规模

| 偏好对数 | 效果 |
|---------|------|
| <500 | 无用 |
| 2-5K | **sweet spot** |
| >10K | 边际收益递减 |

**目标 3-5K 对**。

### 3.7 偏好对质量 > 数量

DPO 对噪声极敏感：
1. 每条 chosen 过 schema validator
2. 每条 chosen 过 actor 名单检查
3. 5% 抽样人工 review

### 3.8 DPO 超参（M4 + 0.8B）

```python
LR = 5e-7              # 远低于 SFT (5e-5)
BETA = 0.1             # 标准
EPOCHS = 1             # 不要 2+
BATCH = 1
GRAD_ACCUM = 8
WARMUP = 100 steps
```

### 3.9 内存预算
ref (frozen 0.8B fp32) + policy (0.8B fp32 + LoRA grad) ≈ 10 GB
M4 16GB 跑得动，~3-4 小时 / 1 epoch / 5K 对。

### 3.10 评估指标对比

| 指标 | Stage 2 | Stage 3 期望 |
|------|:-------:|:------------:|
| Tool Selection Acc | 60-70% | **80-90%** |
| Schema Validity | 85-90% | **98%+** |
| Character Break Rate | 5-10% | **<2%** |
| Persona Score | 3.5/5 | **3.7/5** |
| Hallucinated Actor Rate | 10-15% | **<3%** |

**Tool Acc 推不到 80% = 论文核心 claim 不成立。**

### 3.11 风险

| 风险 | 症状 | 应对 |
|------|------|------|
| Reward hacking | 总调最频繁工具，多样性掉 | 监控生成多样性，β 不要太低 |
| Distributional collapse | 文风变差 | β 升 0.2，epochs 控 1 |
| 偏好对噪声 | DPO 后没改善或变差 | 5% review，多 judge 共识 |
| 没显著提升 | Stage 2 已太强 / 偏好对错位 | 重新设计偏好对，β 降 0.05 |

### 3.12 我们的执行计划

```
Phase A (1 周): 偏好对生成
  - 扰动器: ~3000 对
  - Stage 2 sample + Claude judge: ~1500 对
  - 对抗手工: ~500 对
  
Phase B (1 周): DPO 训练
  - trl DPOTrainer, MPS
  - 3-4 小时 on M4
  
Phase C (1 周): 评估迭代
  - DEBench 全套
  - 不达标 → 补偏好对 → 再训（最多 3 轮）
```

---

## 4. 三阶段为什么这么分

### 4.1 阶段 form vs correctness 边界

```
Stage 1 (Persona SFT)     →  教 form (style)
Stage 2 (Tool-aug SFT)    →  教 form (tool format)
Stage 3 (DPO)             →  教 correctness (tool selection)
```

**Stage 1+2 都用 SFT** 因为它们教的是 form（拷贝分布即可）。
**Stage 3 用 DPO** 因为它教的是 correctness（需要偏好/对错信号）。

### 4.2 为什么不一阶段做完

理论上可以"end-to-end SFT 学一切"。实践中：
1. **数据混合稀释信号**：persona 数据和 tool 数据混合训，每个目标都学得一般
2. **失败模式不可分离**：一阶段训完发现工具选错，没法定位是 persona 不一致还是 tool 学错
3. **DPO 必须在 SFT 之后**：DPO 是精修，不是从零开始

### 4.3 为什么不只用 SFT

理论上 Stage 1+2 已经能跑。但 benchmark 期望：
- Stage 2 only: tool acc 60-70%
- Stage 2 + Stage 3 DPO: tool acc 80-90%
- 14B baseline: 85-90%

**Stage 3 是从"做出来"到"做得好"的关键一跃**，没有 Stage 3 就不能 claim 对标 14B。

---

## 5. 论文章节 vs 方法论映射

```
§3 (Disco Elysium Testbed)   ← 解释为什么 DE 适合
§4 (System)                   ← Unity 架构
§5 (Method)                   ← 三阶段训练
   §5.1 Persona-SFT            ← Part 1 (SFT)
   §5.2 Tool-augmented SFT     ← Part 1 (SFT for tool format)
   §5.3 DPO                    ← Part 3 (Stage 3 这章)
§6 (Benchmarks)               ← DEBench + CPDC + BFCL
§7 (Experiments)              ← 表格填数字
§8 (Discussion)               ← 局限 + 并行工作
§B Appendix                   ← v0 Memory Prefix 负结果
```

---

## 6. 给后续会话的 cheat sheet

接手项目时**先读这个**避免重复犯错：

1. **0.8B SFT** — 用 v2 保守超参（LR 5e-5, dropout 0.15, r=16, ≤4 epochs）
2. **Memory** — 5 种机制各司其职，不要搞通用 memory module
3. **DPO** — 偏好对质量 > 数量，覆盖 8 种失败模式，β=0.1，1 epoch
4. **判别 SFT 适不适合** — 学 form 用 SFT，学 correctness 用 DPO/RL
5. **Stage 3 不可省** — 没有 DPO，论文核心 claim 不成立
