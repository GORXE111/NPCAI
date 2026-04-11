# TownAgent Benchmark Specification

## 概述
6个评估维度，覆盖NPC对话系统的所有核心能力。
每个维度有明确的协议、指标、样本量和自动化评估方法。

---

## 维度 1：Personality Consistency（人格一致性）

**目标**: 评估NPC是否在多轮对话中保持角色人设

**协议**:
1. 对5个NPC各问5轮递进式问题（自我介绍→价值观→看法→改变→恐惧）
2. N = 5 NPC × 5 questions = 25 samples per condition

**指标**:
- C1: Occupation Mention Rate（职业提及率）
- C2: Speech Style Adherence（说话风格一致率）
- C3: Backstory Consistency（背景故事一致率）
- C4: No Character Break（无出戏率）

**自动化评估**: LLM-as-judge（GPT-4/Claude评分）+ 关键词匹配辅助

**对比**: Base vs LoRA vs LoRA+Memory

---

## 维度 2：Memory Recall Accuracy（记忆召回准确率）

**目标**: 评估NPC能否正确引用过去的事件

**协议**:
1. 给NPC注入3条记忆事件（如"玩家昨天买了剑"、"北方有土匪"、"面粉短缺"）
2. 问NPC相关问题，检查是否引用了正确的记忆
3. 问NPC无关问题，检查是否不会幻觉出假记忆
4. N = 10 NPC × 3 memories × 2 questions (related + unrelated) = 60 samples

**指标**:
- Recall Rate: 相关问题中正确引用记忆的比例
- Precision: 引用的记忆中正确的比例（无幻觉）
- False Memory Rate: 无关问题中编造记忆的比例

**对比**: Prompt-based memory vs Memory Prefix Injection

---

## 维度 3：Information Propagation（信息传播）

**目标**: 评估信息在NPC社交网络中的传播准确性

**协议**:
1. Pre-test: 用虚构专有名词问所有NPC（baseline）
2. Seed: 告知一个NPC
3. Hop-1/Hop-2: 让NPC之间传播
4. Post-test: 再次询问所有NPC
5. N = 3 scenarios × 9 NPCs = 27 samples per phase

**指标**:
- False Positive Rate (pre-test): 传播前的假阳性（应≈0%）
- Propagation Rate (post-test): 传播后知晓率
- Keyword Preservation Rate: 关键信息在每跳后的保留率
- Semantic Distortion Score: 语义偏移度（cosine similarity原始vs复述）

**评估方法**: 语义判断（deny vs confirm分类）而非简单关键词匹配

---

## 维度 4：Emotion Coherence（情感连贯性）

**目标**: 评估NPC情感状态的合理性和跨轮连贯性

**协议**:
1. 给NPC一个情感刺激（如"你的店被烧了！"、"国王赏你金币！"、"你的朋友死了"）
2. 后续3轮对话检测情感是否连贯
3. N = 5 NPC × 4 emotion stimuli × 4 turns = 80 samples

**刺激类型**:
- Positive: "国王赏赐你！" → expected: happy → 后续应维持正面
- Negative: "你的店被烧了！" → expected: angry/sad → 后续应不突然变happy
- Threat: "敌人来了！" → expected: fearful → 后续应维持警觉
- Neutral: "今天天气不错" → expected: neutral → 应保持平稳

**指标**:
- Emotion Transition Validity: 情感转换是否合理（人工定义合理转换矩阵）
  - happy → 坏消息 → sad/angry ✅
  - happy → 坏消息 → happy ❌
- Emotion Duration: 强烈情感是否在3轮内保持（不突然跳回neutral）
- Emotion Head Agreement: Emotion Head预测 vs 人工标注的一致率
- Ground Truth Agreement: 与人工标注的加权F1

**合理转换矩阵**:
```
FROM\TO    neutral  happy  angry  sad  fearful
neutral    ✅       ✅     ✅     ✅   ✅       (任何都合理)
happy      ✅       ✅     ✅     ✅   ✅       (收到坏消息可变)
angry      ✅       ❌     ✅     ✅   ✅       (angry→happy不自然)
sad        ✅       ❌     ✅     ✅   ✅       (sad→happy不自然)
fearful    ✅       ❌     ✅     ✅   ✅       (fearful→happy不自然)
```

---

## 维度 5：Social Dynamics（社交动态）

**目标**: 评估NPC之间的关系是否合理演化

**协议**:
1. 初始化社交图谱（铁匠和面包师是好友，卫兵和扒手对立等）
2. 运行30分钟自主交互（约50-60次NPC-NPC对话）
3. 检查社交图谱变化 + 对话内容

**指标**:
- Relationship Consistency: 好友间对话是否友好（affinity > 0的NPC对话sentiment应为正）
- Trust Propagation Accuracy: A信任B，B信任C → A是否逐渐获得C的信息？
- Conflict Detection: 敌对NPC（如Brynn vs Finn, affinity < 0）对话是否反映紧张/对抗
- Affinity Change Validity: 关系变化方向是否合理（友好对话 → affinity上升）
- Social Graph Stability: 30分钟后图谱是否趋向合理（不会全变成0或全变成1）

**评估方法**:
- 对话sentiment分析（positive/negative/neutral）
- 对比初始图谱 vs 终态图谱的变化合理性
- 检查信息传播路径是否沿高trust边传播

---

## 维度 6：Response Latency & Throughput（响应延迟与吞吐）

**目标**: 评估端侧推理的实时性

**协议**:
1. 单NPC对话延迟：连续20次对话，记录每次响应时间
2. 并发吞吐：3个NPC同时对话，记录完成时间
3. 队列压力：10个NPC同时请求，记录排队等待时间

**指标**:
- Single NPC P50/P95 Latency
- Concurrent P50/P95 (parallel=3)
- Queue Wait Time (10 concurrent)
- Throughput: dialogues/minute

**硬件**: Apple M4 16GB, Qwen3.5-9B Q4_K_M via Ollama

---

## 总分表（论文Table格式）

| Dimension | Metric | Base | +LoRA | +Memory | +Emotion | +All |
|-----------|--------|------|-------|---------|----------|------|
| Personality | Avg C1-C4 | | | | | |
| Memory Recall | F1 | | | | | |
| Propagation | Rate/Preservation | | | | | |
| Emotion | Transition Validity | | | | | |
| Social | Relationship Consistency | | | | | |
| Latency | P50 (seconds) | | | | | |
