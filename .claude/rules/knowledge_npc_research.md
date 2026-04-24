# NPC 研究知识库

这里记录项目过程中对 NPC / SLM / 工具调用领域的关键发现、并行工作、可用资源。

---

## 1. Benchmark 景观（2025-2026）

### Tier 1 必跑（直接可用，与本项目完全对口）

#### CPDC 2025 — Commonsense Persona-Grounded Dialogue Challenge
- **时间/场所**: Wordplay Workshop @ EMNLP 2025（苏州，2025年11月9日）
- **数据**: 公开在 AIcrowd
- **任务**: 3 个任务 × 4 轴 LLM-Judge（Scenario Adherence / NPC Believability / Persona Consistency / Dialogue Flow）
- **SOTA**: 全部是 Qwen3-14B + LoRA/GRPO + L40S GPU
- **对标论文**:
  - arXiv 2511.20200 (MSRA_SC) — Context Engineering + GRPO
  - arXiv 2510.13586 (Deflanderization) — Qwen3-14B + SFT + LoRA
  - arXiv 2511.01720 (Multi-Expert NPC Agent) — 三 LoRA 头
  - arXiv 2509.24229 (Model Fusion) — Qwen3-14B + 3 LoRA via vLLM
- **为什么是我们的核心 benchmark**: 任务完全匹配（NPC 对话 + 工具调用），获奖方案全是 14B，我们 0.8B 有清晰的 Pareto 空白

#### BFCL V4 (Berkeley Function Calling Leaderboard)
- **链接**: gorilla.cs.berkeley.edu/leaderboard.html
- **关键数字**: Qwen3-0.6B, Qwen3-4B, lfm2.5:1.2b, phi4-mini:3.8b 都在 **0.880 agent score** 上并列
- **结论**: 参数量不是强预测因子；架构+训练数据更重要
- **可用性**: github.com/ShishirPatil/gorilla，Inspect-Evals 一键跑

#### τ²-Bench (Sierra Research)
- **论文**: arXiv 2506.07982 (2025年6月)
- **设置**: 多轮 agent-user 对话 + dual-control 共享世界状态
- **代码**: github.com/sierra-research/tau2-bench
- **数字**: GPT-4 级 agent 都 <50% success，pass^8 约 25%

#### ToolSandbox (Apple, NAACL 2025 Findings)
- **论文**: arXiv 2408.04682
- **卖点**: 状态相关工具（send_message 在关蜂窝时失败）
- **代码**: github.com/apple/ToolSandbox (Apache 2.0)

### Tier 2 关键并行工作（必须引用+区分）

#### Fixed-Persona SLMs with Modular Memory (arXiv 2511.10277, 2025.11)
- **和我们几乎同方向**: SLM + NPC + 消费硬件 + 模块化记忆
- **他们有**: DistilGPT-2 / TinyLlama-1.1B / Mistral-7B
- **他们没有**: 工具调用、Unity 集成、M4 on-device 数字
- **定位**: 直接对标，必须 cite 并说清差别

#### Small LMs for Efficient Agentic Tool Calling (arXiv 2512.15943, 2025.12, Amazon)
- **最强证据**: 350M OPT fine-tuned 打赢 500× 大模型（77.55% pass on ToolBench）
- **对我们的价值**: 直接支持"SLM + targeted fine-tune"论点

#### Orak (ICLR 2026, arXiv 2506.03610)
- **12 个游戏的 LLM agent benchmark** + 基于 MCP 的 plug-and-play 接口
- **数据集公开**，可能作为训练数据源

### Tier 3 相关上下文
- lmgame-Bench (ICLR 2026) — 6 游戏 + 13 SOTA 模型
- BALROG (ICLR 2025) — 6 RL 环境（NetHack 等）
- AgentBench v3 (2025.10 revision)
- Gaia2 (OpenReview 2025) — 动态异步 agent 环境
- ToolHop (ACL 2025) — 995 查询 × 3912 工具，多跳
- NesTools (COLING 2025) — 嵌套工具调用
- LongMemEval (ICLR 2025, arXiv 2410.10813) — 长期对话记忆
- RPGBench (arXiv 2502.00595) — LLM 作为 RPG 引擎（反向视角）
- SLM-Bench (arXiv 2508.15478) — 15 SLM × 9 任务 × 23 数据集

### 产业界（无 peer review，但可引）
- **NVIDIA ACE (CES 2025)**: 在 PUBG / inZOI / NARAKA 用自主 NPC
- **Inworld AI**: 商业 middleware
- **LLM-Driven NPCs Cross-Platform (arXiv 2504.13928)**: Unity + Discord

---

## 2. 论文 Gap Analysis

2025-2026 landscape 的**未被占据的 Pareto 位置**：

| Gap | 证据 | 我们占据 |
|-----|------|---------|
| 没有 benchmark 用 <1B + 工具调用 + on-device runtime | CPDC 获奖全是 14B + L40S | Qwen3.5-0.8B + M4 Mac |
| CPDC 方向没人系统研究 SLM | 所有获奖是大模型 | 我们是第一篇 "how small can we go?" |
| 工具调用都是抽象 API | Tau²/ToolSandbox/CPDC 都是文本 tool | 我们绑定真实 Unity 执行（动画/物品/镜头） |
| 记忆 + 工具调用没整合 | LongMemEval 没工具、BFCL V4 没人格 | MPI + EH + Tool-calling 是新组合 |
| 没有 "M4 Mac 上 NPC 工具调用" 数字 | Edge-First Inference 只通用 NLP | 独家报告 Apple Silicon latency |

---

## 3. 论文方向演化（避免回到老路）

### v0 (最初): "三阶段 SLM NPC 架构"
- Stage 1 LoRA + Stage 2 Memory Prefix + Stage 3 Emotion Head
- **失败原因**: Base Qwen3.5-0.8B 已 3.81/5，我们训完 3.52 —— 负面结果

### v1: "Memory Faithfulness via DPO"
- 聚焦记忆保真度问题（benchmark 证实所有配置 memory_use ≈ 2.85/5）
- **被用户否决**: 范围太窄

### v2 (**当前**): "Tool-Using SLM NPCs in Games"
- LLM + 游戏内 API 工具调用
- 学术空白（CPDC 2025 有 benchmark 但没 SLM 方向）
- 可以结合 Unity demo + 真实动画/物品交互

---

## 4. 核心技术坑（已解决，下次直接用）

### Qwen3.5 MPS dtype 冲突
- **症状**: `MPSNDArrayMatrixMultiplication.mm:4140 failed assertion: Destination NDArray and Accumulator NDArray cannot have different datatype`
- **根因**: `Qwen3_5GatedDeltaNet.forward` 第 71 行 `.float()` 调用产生混合 dtype
- **解法**: `model/qwen35_mps_fix.py` 的 `patch_qwen35_for_mps()`，强制整个模型 float32 + 包装 GatedDeltaNet forward
- **适用**: Qwen3.5 全系列（0.8B / 2B 均验证）

### Ollama Qwen3.5 thinking mode 污染
- **症状**: API response 字段是空，内容跑到 `thinking` 字段
- **解法**: 请求加 `"think": false`，或 fallback 读 `thinking` 字段

### MPS 内存限制（M4 16GB）
- **0.5B float32**: ~2GB，稳定
- **0.8B float32**: ~3.2GB，稳定
- **2B float32**: ~8GB + forward pass 激活 → 12GB，紧张但能跑
- **3B float32**: ~12GB + 激活 → OOM → swap → 训练慢到不可用
- **3B float16 + LoRA merge**: dtype 混合 → NaN
- **结论**: M4 上 SFT/MPI/EH 最大稳定跑 2B，3B 需要外部 GPU

### LoRA 过拟合（0.8B）
- **症状**: 8K curated 数据 × 5 epochs × r=8 → val loss 1.89 但 benchmark 0% 合格
- **表现**: 响应疯狂重复同一句话 "I am X. I am X. I am X."
- **解法 (v2)**: LR 5e-5、dropout 0.15、r=16、epochs=3、weight_decay=0.05、mid-epoch eval + patience=2
- **v2 结果**: val 1.94（比 v1 高），但 benchmark 可能更好（还没跑）
- **更根本的**: 0.8B 可能根本不适合 LoRA NPC 适配；2B 可以（278 条 → val 0.344）

---

## 5. 数据质量 > 数量（量化证据）

| 数据集 | 样本量 | Val Loss (Qwen3.5-2B) | Val Loss (Qwen3.5-0.8B) |
|--------|--------|----------------------|-------------------------|
| 278 手写精标 | 278 | **0.344** | - |
| 8K curated (LIGHT + amaydle 过滤) | 8,131 | 1.757 | **1.89** (过拟合) |
| 81K mixed (未过滤) | 81,036 | 2.011 | - |

**结论**: NPC 对话 SFT 数据质量碾压数量。针对性清洗（用关键词验证情感标签、剔除非中世纪对话）比盲目收集更有效。

---

## 6. LLM-as-Judge 靠谱性

### qwen3.5:9b 做 judge 的问题
- **同族偏差**: 判断 qwen3.5 系列输出时有倾向
- **噪声**: C 和 D 配置响应文本完全一致，qwen3.5 给 3.31 vs 3.40（应该相同）
- **thinking mode 污染**: 默认会把 JSON 评分写进 thinking 而非 response

### Claude 盲评更可靠
- **验证**: C=D 的响应文本一致时，Claude 正确给出 3.52 = 3.52
- **方法**: 打散标签 + 匿名化 + 单次独立评分
- **成本**: 160 条评分约 500k tokens，一次完成

### 最佳实践（从 PingPong / CharacterEval）
- 多 judge 面板（不同家族）
- 人工小样本校准
- Blind randomized order

---

## 7. 可复用资产盘点

| 资产 | 位置 | 在新方向还能用吗 |
|------|------|----------------|
| Unity 10 NPC 系统 | `unity/Assets/Scripts/NPC/` | ✅ 直接用，加 ToolExecutor |
| OllamaClient | `unity/Assets/Scripts/LLM/` | ✅ 改 JSON 结构化输出 |
| NPCMemory / SocialGraph | `unity/Assets/Scripts/NPC/` | ✅ 暴露为工具 (recall, notify_npc) |
| 278 条手写对话 | `data/training_data/` | ⚠️ 扩展成"对话+工具" |
| Qwen3.5-0.8B base | Ollama cache | ✅ 直接基座 |
| qwen35_mps_fix.py | `model/` | ✅ 继续用 |
| 盲评 benchmark 协议 | `benchmarks/` | ✅ 作为 methodology |
| Memory Prefix 架构 | `model/` | ⚠️ 作为 ablation 可能有用 |
| Stage 1/2/3 checkpoint | `checkpoints/` | ⚠️ 可能需要重跑（新 tool-use 数据） |
| GitHub 仓库 | https://github.com/GORXE111/NPCAI | ✅ 继续推 |

---

## 8. 下一步 Roadmap（当前共识）

### 立刻
1. Clone CPDC 2025 数据集（AIcrowd）
2. Base Qwen3.5-0.8B 在 CPDC Task 1/3 上跑 zero-shot（取基线数字）
3. 读 arXiv 2511.20200 / 2510.13586 / 2511.01720（获奖论文）

### 第 1 周
4. 设计 Tool API（20-25 个工具 JSON schema）
5. Unity ToolExecutor 组件
6. 端到端 demo：玩家问话 → LLM 返回对话+tool → Unity 执行动画

### 第 2-3 周
7. 合成训练数据（Claude 生成 3-5K 条 tool-use NPC traces）
8. SFT Qwen3.5-0.8B on tool traces
9. DPO 精调

### 第 4 周
10. 跑 CPDC 2025 + BFCL V4 + τ²-Bench 对比
11. 写论文 + demo 视频

---

## 9. 要避免的陷阱

1. **不要再无 benchmark 先训练** — 每次方向变更先跑基线
2. **不要只看 val loss** — 用 CPDC 4 轴 judge 或 benchmark 分数
3. **不要忽略 concurrent work** — 搜 arXiv 最近半年用关键词
4. **不要用同族 LLM 做唯一 judge** — 多 judge 面板
5. **不要在 M4 上跑 3B+ float32 训练** — swap 到死，用 0.8B 或 2B
6. **不要合并 LoRA 在 float16 上** — 产生混合 dtype 导致 NaN
