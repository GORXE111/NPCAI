---
name: NPC-LLM Research Project
description: 论文项目（目标IEEE TG）：多NPC社交网络 + Memory Cross-Attention + Emotion Head + LoRA，Unity + qwen3.5，RTX3060训练/M4推理
type: project
---

## 论文方向
TownAgent: 多NPC社交网络 + Memory Cross-Attention + Emotion Head + LoRA，目标 arXiv 预印本 → IEEE Transactions on Games (SCI Q1/Q2)

**核心卖点**: Generative Agents 的轻量化实际游戏部署 + 模型适配层让通用SLM理解NPCAgent协议

## 五层架构
- Layer 4: Unity 游戏层（小镇集市、10 NPC、玩家交互购买、表情动画）
- Layer 3: NPCAgent 框架（社交图、调度引擎、日夜周期、自主行为FSM）
- Layer 2: Memory Cross-Attention（记忆向量注入transformer，替代prompt-based记忆）
- Layer 1: Emotion Head（情感分类+情感向量注入下一轮，驱动Unity表情）
- Layer 0: LoRA + 量化推理（RTX3060训练，M4端侧推理）

## 论文贡献点（IEEE Transactions on Games 级别）
1. **架构创新**: Memory Cross-Attention Layer — 首次将记忆向量通过cross-attn注入NPC对话模型
2. **情感连续性**: Emotion Head — 分类当前情感+注入下一轮，实现跨轮情感状态转移
3. **框架贡献**: NPCAgent 多NPC社交仿真（调度+社交图+自主行为+信息传播）
4. **端侧系统**: Unity + 量化模型 + 本地记忆持久化的完整方案
5. **实验**: base vs LoRA vs Memory+Emotion 三级对比 + 0.8B/4B/9B 规模对比

## 三阶段训练策略
- Stage 1: LoRA adapters（冻结原模型+Memory+Emotion，用NPC对话数据）
- Stage 2: Memory module + gate（冻结原模型+LoRA，用长对话记忆数据）
- Stage 3: Emotion head（全部冻结，用情感标注数据）

## 训练硬件
- RTX 3060 12GB (Windows) — 训练用
- Mac M4 16GB — 推理/Unity部署用
- Mac上MLX训练4B+均OOM，改用Windows CUDA

## 记忆系统设计
- ShortTerm（工作记忆）: 当前对话上下文
- Episodic（情景记忆）: 具体事件
- Semantic（语义记忆）: 世界知识
- Social（社交记忆）: 关系图+好感度
- Forgetting Curve: Ebbinghaus 遗忘曲线衰减

## 实验指标
- 信息传播准确率、人格一致性分数、推理延迟、Token效率、成本对比

## 核心参考文献
- Park et al. "Generative Agents" (UIST 2023, CCF-A) — 奠基工作
- MemoryBank (AAAI 2024, CCF-A) — 记忆+遗忘曲线
- Think-in-Memory (AAAI 2024, CCF-A) — 记忆中思考
- CharacterGLM (ACL 2024, CCF-A) — 角色一致性
- CoALA (Princeton/TMLR) — 认知架构理论
- Concordia (DeepMind) — 多agent社交仿真框架
- QLoRA (NeurIPS 2023, CCF-A) — 量化微调

## 项目信息
- Mac 路径: ~/npcllm/
- Unity: 6000.3.11f1
- LLM: qwen3.5:9b (Ollama, Mac M4 本地)
- AI工作流: SSH写代码 + MCP检查编译/运行 + TestBridge测试
- MCP配置: D:\AIproject\.mcp.json → SSH stdio → Mac MCP server

## 已完成
- OllamaClient.cs（Unity↔Ollama通信）
- TestBridge.cs（文件+MCP双模式测试桥）
- AutoRefresh.cs + AutoPlayMode.cs（编辑器自动化）

**Why:** 端侧小模型+多NPC社交是2025-2026空白；LoRA适配提升论文技术深度；面向GPU+NPU未来硬件
**How to apply:** Mac开发Unity+Ollama，后续在Mac M4上做LoRA微调实验（MLX）
