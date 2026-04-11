# NPC-LLM 项目知识库

## 项目概况
- **标题方向**: TownAgent: Memory-Augmented Small Language Models with Emotion-Aware Generation for Multi-NPC Social Simulation
- **目标**: arXiv 预印本 → IEEE Transactions on Games (SCI Q1/Q2)
- **硬件**: Apple M4 16GB (训练+推理), RTX 3060 12GB (备用), 后续 H100 (大模型)
- **引擎**: Unity 6000.3.11f1
- **项目路径**: Mac ~/npcllm/, Windows D:\AIproject\npcllm_paper\

## 核心贡献点（IEEE TG 级别）
1. **Memory Prefix Injection**: 将NPC记忆编码为可学习prefix tokens注入transformer输入，替代prompt-based记忆。Gate机制控制注入强度。
2. **Emotion Head**: 从hidden state分类NPC情感状态，情感向量注入下一轮生成，实现跨轮情感连续性。可驱动Unity表情/动画。
3. **三阶段训练策略**: Stage1 LoRA → Stage2 Memory → Stage3 Emotion，分阶段冻结避免互相干扰。
4. **NPCAgent框架**: 10个NPC的完整社交仿真（FSM行为、社交图、自主对话、日夜周期、信息传播）。
5. **端侧全链路**: 训练+推理全在消费级硬件（M4 16GB），证明可行性。

## 五层架构
```
Layer 4: Unity 游戏层
  ├── TownManager (10个NPC, 2排摊位, 2D俯视)
  ├── DayClock (日夜周期: 1实分=1游戏时)
  ├── NPCBehavior (FSM: Idle/Working/Walking/Chatting/Trading)
  ├── SpeechBubble (气泡对话+打字机效果+名字标签)
  ├── SocialDirector (自主NPC对话: 走近→聊天→走回)
  └── NPCScheduler (优先级队列, maxConcurrent=3)

Layer 3: NPCAgent 框架
  ├── NPCBrain (人格+记忆+LLM推理整合)
  ├── NPCMemory (四层记忆+Ebbinghaus遗忘曲线)
  ├── SocialGraph (关系网络: affinity/trust/sharedMemories)
  ├── TradeInventory (物品+价格)
  └── TestBridge (文件通信+MCP双模式)

Layer 2: Memory Prefix Injection
  ├── MemoryPrefixEncoder (记忆文本→prefix tokens)
  ├── Gate (可学习门控, sigmoid初始0.5)
  └── 注入位置: input embeddings前拼接

Layer 1: Emotion Head
  ├── EmotionHead (hidden state → 8类情感分类)
  ├── EmotionEmbedding (情感向量→下一轮注入)
  └── 8类: neutral/happy/angry/sad/fearful/surprised/disgusted/contemptuous

Layer 0: 模型层
  ├── Base: Qwen2.5-0.5B / Qwen2.5-3B / Qwen3.5-2B
  ├── LoRA: rank 8, target q/k/v/o_proj
  └── 推理: Ollama (qwen3.5:9b Q4_K_M) / PyTorch MPS
```

---

## 实验结果汇总

### Stage 1: LoRA 对话适配

| 模型 | 发布年 | 参数 | 数据量 | Best Val Loss | M4训练时间 |
|------|--------|------|--------|---------------|-----------|
| Qwen2.5-0.5B + LoRA | 2024 | 0.5B | 154条 | 0.891 | ~2min |
| Qwen2.5-3B + LoRA | 2024 | 3B | 154条 | 0.418 | ~10min |
| Qwen3.5-2B + LoRA | 2026 | 2B | 154条 | 0.442 | ~5min |
| **Qwen3.5-2B + LoRA (大数据)** | 2026 | 2B | **278条** | **0.344** | ~8min |

**关键发现**:
- Qwen3.5-2B (2026) 用更少参数(2B vs 3B)达到了和 Qwen2.5-3B 几乎相同的效果
- 数据量从154→278条，Val Loss从0.442→0.344（-22%）
- LoRA显著减少角色出戏（Base的Aldric提到"latest technologies"，LoRA不会）

### Stage 1 对话质量对比（Aldric 铁匠）

| 问题 | Base 3B | LoRA 3B |
|------|---------|---------|
| Who are you? | 正确但平淡 | "Just a man who melts steel" — 铁匠口吻 |
| Sell? | 泛泛而谈 | "If you're made of steel, I've got something for you" — 锻造比喻 |
| Afraid? | "losing my skills...technologies" 出戏 | "That someone will tear this town apart" — 保持人设 |

### Stage 2: Memory Prefix Injection

| 配置 | 数据量 | Best Val Loss | Gate |
|------|--------|---------------|------|
| 0.5B + 小数据 | 20条 | 2.790 | 0.500 |
| **0.5B + 大数据** | **168条** | **1.467** (-47%) | 0.498 |

**发现**: Loss持续下降20 epochs无过拟合，证明Memory Prefix架构有效。Gate稳定在~0.5（初始值），需更大模型才能看到gate的显著学习。

### Stage 3: Emotion Head

| 配置 | 数据量 | Best Train Acc | Best Val Acc |
|------|--------|----------------|-------------|
| 0.5B | 48条(6类) | 67% | 58.3% |

**发现**: 架构验证通过（train acc 67%证明hidden state含情感信息），但数据太少导致过拟合。需要更多数据+更大模型。

### NPC系统运行测试

| 测试项 | 结果 |
|--------|------|
| NPC对话（Ollama qwen3.5:9b） | Aldric 6.8s, Elara 9.4s，人设完美保持 |
| 自主NPC-NPC对话 | SocialDirector每30s触发，NPC走近后聊天 |
| 信息传播V1 | 9/9 NPC"知道"（关键词匹配过宽） |
| 信息传播V2（虚构名词） | 语义分析后假阳性0%，传播机制需加强 |
| 人格一致性（5轮） | 5个NPC全部保持人设 |

---

## 训练硬件实测

| 硬件 | 可训练模型 | 限制 |
|------|-----------|------|
| Mac M4 16GB (MPS) | 0.5B float32 ✅, 3B float16 ✅, Qwen3.5-2B float16 ✅ | 4B+ OOM |
| Mac M4 16GB (MLX) | 全部OOM（macOS 26 Metal bug） | 不可用 |
| RTX 3060 12GB (CUDA) | 0.5B ✅, 需要clean venv | 9B需QLoRA+unsloth |
| Gemma 4 E4B | Mac OOM, peft不支持ClippableLinear | 太新 |
| Gemma 4 E2B | Mac OOM（多模态模型实际>2B） | 太新 |
| Qwen3.5 (MPS) | MPS dtype冲突（linear_attn层） | 用float32可解决0.5B |

---

## 核心参考文献

### Tier 1: 必引（奠基+直接相关）

[1] Park et al. "Generative Agents: Interactive Simulacra of Human Behavior." **UIST 2023** (CCF-A)
- 25个LLM agent沙盒小镇，记忆流+反思+规划。我们的差异：端侧+Memory Prefix+Emotion Head

[2] Zhong et al. "MemoryBank: Enhancing LLMs with Long-Term Memory." **AAAI 2024** (CCF-A)
- Ebbinghaus遗忘曲线。我们的NPCMemory直接参考。

[3] Liu et al. "Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory." **AAAI 2024** (CCF-A)
- 记忆中思考范式。

[4] Sumers et al. "CoALA: Cognitive Architectures for Language Agents." **TMLR 2024**
- 四层记忆理论基础。

[5] Zhou et al. "CharacterGLM: Customizing Chinese Conversational AI Characters." **ACL 2024** (CCF-A)
- 角色一致性评估。

[6] Dettmers et al. "QLoRA: Efficient Finetuning of Quantized Language Models." **NeurIPS 2023** (CCF-A)
- 4bit量化+LoRA。

### Tier 2: 重要参考

[7] Vezhnevets et al. "Concordia." Google DeepMind, arXiv 2023.
[8] Wang et al. "Voyager." **NeurIPS 2023 Spotlight** (CCF-A)
[9] Li et al. "CAMEL." **NeurIPS 2023** (CCF-A)
[10] Wang et al. "Humanoid Agents." **EMNLP 2023 Demo** (CCF-B)
[11] Altera AI. "Project Sid." arXiv 2024.
[12] Zhu et al. "CALYPSO." **AIIDE 2023**
[13] Jeon, Ha, Kim. "LRAgent: KV Cache Sharing for Multi-LoRA Agents." arXiv:2602.01053, 2026.
[14] Braas, Esterle. "Fixed-Persona SLMs with Modular Memory." arXiv:2511.10277, 2025.
[15] "SimWorld." **NeurIPS 2025 Spotlight**
[16] Xie et al. "AgentBench." **ICLR 2024** (CCF-A)

---

## 文件索引

### Mac ~/npcllm/
```
Assets/Scripts/
  LLM/OllamaClient.cs          # Unity↔Ollama通信
  NPC/NPCBrain.cs               # 人格+记忆+LLM
  NPC/NPCMemory.cs              # 四层记忆+遗忘曲线
  NPC/SocialGraph.cs             # 社交关系网络
  NPC/NPCScheduler.cs            # 推理调度+缓存
  NPC/NPCBehavior.cs             # FSM行为状态机
  NPC/SocialDirector.cs          # 自主NPC对话
  NPC/DayClock.cs                # 日夜周期
  NPC/TradeInventory.cs          # 交易系统
  NPC/TownManager.cs             # 10NPC小镇管理
  NPC/TestBridge.cs              # 测试桥接
  NPC/NPCLocations.cs            # 位置数据
  UI/SpeechBubble.cs             # 气泡对话+名字
  Editor/AutoRefresh.cs          # 自动刷新
  Editor/AutoPlayMode.cs         # 文件信号Play控制
  Editor/MarketSceneSetup.cs     # 2D场景搭建

model/
  npc_model.py                   # 完整NPC模型架构
  train_stage1.py                # Stage 1 LoRA
  train_stage2.py                # Stage 2 Memory Prefix
  train_stage3.py                # Stage 3 Emotion Head
  train_gemma4.py                # Gemma 4尝试（peft不兼容）

training_data/
  _combined/                     # 原始172条
  _combined_large/               # 扩展278条
  _memory_large/                 # 记忆数据168+42条
  aldric/ elara/ finn/ ...       # 每NPC单独数据

checkpoints/
  stage1/                        # 0.5B LoRA (Val 0.89)
  stage1_3b/                     # 3B LoRA (Val 0.42)
  stage1_qwen35_2b/              # Qwen3.5-2B LoRA (Val 0.44)
  stage1_qwen35_2b_large/        # Qwen3.5-2B大数据 (Val 0.34)
  stage2_memory/                 # Memory 0.5B小数据 (Val 2.79)
  stage2_memory_large/           # Memory 0.5B大数据 (Val 1.47)
  stage3_emotion/                # Emotion Head (Acc 58%)

experiment_data/
  experiment_*.json              # 传播+人格实验数据
```

### Windows D:\AIproject\npcllm_paper\
```
draft_v1.md                      # 论文草稿
knowledge_base.md                # 本文件
model/npc_model.py               # 模型架构代码
model/train_*.py                 # 训练脚本
training_data/                   # 训练数据副本
```

### MCP桥接
- 配置: D:\AIproject\.mcp.json → SSH stdio → Mac MCP server
- Unity MCP: localhost:8090 (Mac)
