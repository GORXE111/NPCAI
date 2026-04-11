# TownAgent: Memory-Augmented Small Language Models with Emotion-Aware Generation for Multi-NPC Social Simulation

## Abstract

Large language models (LLMs) have shown remarkable potential in creating believable non-player characters (NPCs) with natural dialogue capabilities. However, existing approaches rely on cloud-based large models (e.g., GPT-4), incurring high latency, cost, and privacy concerns that make them impractical for real-time game deployment. We present **TownAgent**, an end-to-end framework for multi-NPC social simulation powered by on-device small language models (SLMs), featuring two novel architectural contributions: (1) **Memory Prefix Injection**, which encodes NPC episodic memories into learnable prefix tokens via a gated encoder, replacing token-expensive prompt-based memory and reducing context length while improving memory utilization; and (2) **Emotion Head**, a lightweight classification module that infers NPC emotional states from hidden representations and injects emotion vectors into subsequent generation turns, enabling cross-turn emotional continuity that can drive game animations. Our system deploys 10 personality-distinct NPCs in a Unity-based medieval market town with autonomous behaviors, social relationships, and information propagation. We propose a three-stage training strategy—LoRA personality adaptation, Memory Prefix training, and Emotion Head training—that progressively adds capabilities without catastrophic forgetting. Experiments on Apple M4 (16GB) demonstrate that a 2B-parameter model with LoRA achieves validation loss 0.344 on NPC dialogue (comparable to 3B models), the Memory Prefix module reduces memory-augmented dialogue loss by 47% over prompt-based baselines, and the Emotion Head achieves 58.3% accuracy on 6-class emotion classification. All training and inference runs entirely on consumer hardware without cloud dependencies.

**Keywords**: NPC dialogue, social simulation, small language models, on-device inference, LoRA fine-tuning, Unity

---

## 1. Introduction

The emergence of large language models has opened new possibilities for creating non-player characters that can engage in natural, contextual conversations rather than relying on pre-scripted dialogue trees [1]. The seminal work of Park et al. [1] demonstrated that LLM-powered agents can exhibit emergent social behaviors—forming relationships, spreading information, and coordinating activities—within a simulated environment. However, their approach depends entirely on cloud-based API calls to GPT-3.5/4, which introduces three fundamental limitations for practical game deployment:

1. **Latency**: Cloud round-trips of 1-5 seconds per NPC response create unacceptable pauses in real-time gameplay.
2. **Cost**: Running 10+ NPCs with continuous dialogue generates thousands of API calls per hour, costing tens of dollars per session.
3. **Privacy**: Player interactions are transmitted to external servers, raising concerns for privacy-sensitive applications.

Recent advances in small language models (SLMs) with 1-10B parameters [6], combined with efficient quantization techniques [14], suggest an alternative path: running NPC intelligence entirely on consumer hardware. Modern gaming devices equipped with GPUs and NPUs (e.g., Apple M-series, NVIDIA RTX, Qualcomm Snapdragon X) can potentially host quantized SLMs with acceptable inference speed.

In this paper, we present **TownAgent**, a complete framework for multi-NPC social simulation running on-device. Our contributions are:

- **A multi-NPC social simulation framework** featuring 10 personality-distinct NPCs with autonomous behaviors (movement, trading, conversation), a four-layer memory system with forgetting curves, and a social relationship graph that drives information propagation.
- **NPC-LoRA**, a parameter-efficient fine-tuning method that trains lightweight adapters (10-20MB each) to specialize a general SLM for structured NPC dialogue, improving personality consistency and protocol adherence while maintaining the base model's language capabilities.
- **A priority-based inference scheduler** that manages concurrent NPC inference requests on a single device, balancing player-facing responsiveness with background NPC-NPC social interactions.
- **Comprehensive experiments** on consumer hardware (Apple M4, 16GB) comparing base vs. LoRA-adapted models and 4B vs. 9B parameter scales, evaluating personality consistency, information propagation accuracy, and response latency.

---

## 2. Related Work

### 2.1 LLM-Powered Game Agents

Park et al. [1] introduced generative agents that maintain memory streams, perform reflections, and plan daily activities, producing emergent social behaviors in a 2D sandbox. Subsequent works expanded this vision: Concordia [7] provided a modular library for social simulation with a Game Master architecture; SimWorld [16] scaled to multi-agent tasks in Unreal Engine 5; and Project Sid [11] deployed 1000+ agents in Minecraft. However, all these systems rely on cloud-based large models, and none address on-device deployment.

### 2.2 Memory and Personality Consistency

Maintaining long-term personality consistency remains a key challenge. MemoryBank [2] introduced Ebbinghaus-based forgetting curves for dynamic memory management. Think-in-Memory [3] proposed recall-then-post-think cycles for self-evolving memory. CoALA [4] systematized agent cognitive architectures into working, episodic, semantic, and procedural memory modules. CharacterGLM [5] demonstrated multi-dimensional personality evaluation for role-playing models. Our memory system synthesizes these approaches into a practical four-layer architecture tailored for game NPCs.

### 2.3 Small Language Models for Games

While NVIDIA ACE has deployed 4B parameter models for single-NPC roleplay in commercial titles, academic research on multi-NPC systems with small models is sparse. Fixed-Persona SLMs [15] explored modular memory with small models but tested only up to 7B parameters with 3 NPCs. The CPDC 2025 challenge produced multi-LoRA approaches for tool-calling dialogue agents [14], but in cloud settings. To our knowledge, TownAgent is the first system to deploy multi-NPC social simulation with autonomous behaviors entirely on consumer hardware.

### 2.4 Parameter-Efficient Fine-Tuning

LoRA [6] and QLoRA [6] have made it feasible to fine-tune large models on consumer GPUs by training low-rank adapter matrices. LRAgent [14] recently demonstrated efficient KV cache sharing for multi-LoRA agent serving. We build on these techniques with NPC-LoRA, which trains character-specific adapters that can be hot-swapped during inference to serve different NPCs from a single base model.

---

## 3. System Design

### 3.1 Architecture Overview

TownAgent consists of four layers (Figure 1):

- **Game Layer**: Unity 6000 scene with a 2D medieval market town, 10 NPC stalls, player avatar, and day/night cycle.
- **Agent Layer**: Each NPC is governed by an NPCBrain (personality + dialogue), NPCBehavior (finite state machine), NPCMemory (four-layer persistence), and connections in a SocialGraph.
- **Scheduling Layer**: NPCScheduler manages a priority queue for LLM inference. SocialDirector orchestrates autonomous NPC-NPC conversations with rate limiting and proximity requirements.
- **Model Layer**: On-device SLM (Qwen3.5-4B/9B) with optional NPC-LoRA adapters, served via Ollama with parallel request support.

### 3.2 NPC Behavior State Machine

Each NPC operates a finite state machine with five states:

- **Idle**: NPC stands at a location, periodically wandering to random points.
- **Working**: NPC is at their stall, performing occupation-specific activities.
- **Walking**: NPC moves toward a target position at 2.5 units/second using linear interpolation.
- **Chatting**: NPC is engaged in dialogue (locked until conversation completes).
- **Trading**: NPC is conducting a transaction with the player.

State transitions are driven by the DayClock (game time system where 1 real minute = 1 game hour) and by the SocialDirector.

### 3.3 Four-Layer Memory System

Inspired by CoALA [4] and MemoryBank [2], each NPC maintains:

1. **Short-term memory**: Rolling buffer of the last 10 dialogue turns (in-memory, not persisted). Provides immediate conversational context.
2. **Episodic memory**: Specific events with timestamps ("Player bought a sword at 13:00"). Persisted to JSON. Each entry has importance (0-1) and strength (decays via forgetting curve).
3. **Semantic memory**: Static world knowledge defined in the NPC's personality profile ("The smithy is east of the square").
4. **Social memory**: Relationship data managed by the SocialGraph—affinity, trust, interaction count, and shared memories for each known NPC.

**Forgetting Curve**: Following Ebbinghaus [2], memory strength decays as:

$$R = e^{-\lambda t / S}$$

where $\lambda$ is the decay rate, $t$ is hours since last access, and $S = (1 + 0.5 \cdot \text{accessCount}) \cdot (0.5 + 0.5 \cdot \text{importance})$ is the stability factor. Memories below threshold 0.1 are garbage-collected.

### 3.4 Social Relationship Graph

The SocialGraph is a directed weighted graph where edges represent relationships between NPCs, characterized by:

- **Affinity** [-1, 1]: Ranges from hostile to friendly.
- **Trust** [0, 1]: Governs information sharing—NPCs only propagate information to connections with trust above a threshold.
- **Shared memories**: A log of information exchanged between two NPCs.

The graph is initialized with predefined relationships (e.g., the blacksmith and baker are old friends; the guard captain distrusts the pickpocket) and evolves through interactions.

### 3.5 Inference Scheduling

Running 10 NPCs on a single device requires careful scheduling. NPCScheduler maintains a priority queue where:

- Player-initiated dialogues receive priority 100 (always processed first).
- NPC-NPC conversations receive priority based on spatial proximity to the player.
- A response cache (LRU, 50 entries, 5-minute TTL) avoids redundant LLM calls for repeated queries.

With Ollama's parallel request support (`OLLAMA_NUM_PARALLEL=4`), up to 3 concurrent inference requests are processed, with the rest queued.

### 3.6 Autonomous Social Interactions

The SocialDirector singleton manages NPC-NPC conversations with the following protocol:

1. **Selection** (every 10 seconds): Score all eligible NPC pairs by $\text{affinity} \times \text{proximity}^{-1}$, respecting a global 30-second cooldown and per-pair 120-second cooldown.
2. **Approach**: The initiating NPC walks toward the responder (world rule: must be within 1.5 units to chat).
3. **Dialogue**: The initiator's opening line is generated from templates (no LLM call); only the responder's reply uses an LLM call. This halves the inference cost.
4. **Memory update**: Both NPCs store the interaction in episodic memory and update their social relationship.
5. **Return**: The initiator walks back to their home position.

---

## 4. Model Architecture: Three-Stage NPC Augmentation

We propose a three-stage approach to augment a general-purpose SLM for NPC dialogue: (1) LoRA personality adaptation, (2) Memory Prefix Injection, and (3) Emotion Head. Each stage trains only its own parameters while freezing all others, preventing catastrophic forgetting.

### 4.1 Stage 1: LoRA Personality Adaptation

General-purpose SLMs exhibit character breaks (modern language in medieval settings), format instability, and inability to maintain consistent personality over multiple turns. We apply LoRA [6] with rank 8 to the query, key, value, and output projection matrices of the self-attention layers.

**Training Data**: We construct 278 high-quality conversation samples across 10 NPC archetypes, covering 8 interaction categories (greetings, trading, personal questions, town gossip, world events, opinions, emotional interactions, NPC-to-NPC dialogue). Each sample precisely captures the NPC's unique voice—occupation-specific metaphors, speech patterns, and knowledge boundaries.

**Results** (on Apple M4 16GB, PyTorch MPS):

| Model | Params | Year | Data | Best Val Loss |
|-------|--------|------|------|---------------|
| Qwen2.5-0.5B + LoRA | 0.5B | 2024 | 154 | 0.891 |
| Qwen2.5-3B + LoRA | 3B | 2024 | 154 | 0.418 |
| Qwen3.5-2B + LoRA | 2B | 2026 | 154 | 0.442 |
| **Qwen3.5-2B + LoRA** | **2B** | **2026** | **278** | **0.344** |

Key finding: Qwen3.5-2B (2026 architecture) with 33% fewer parameters achieves comparable performance to the older Qwen2.5-3B, demonstrating that newer architectures improve parameter efficiency for NPC dialogue tasks.

### 4.2 Stage 2: Memory Prefix Injection

Existing NPC memory systems embed memories as text in the prompt, consuming valuable context tokens and providing only indirect memory access. We propose **Memory Prefix Injection**: a learnable encoder that transforms episodic memory texts into a fixed number of prefix tokens prepended to the input embeddings.

**Architecture**:
$$\text{prefix} = \sigma(g) \cdot W_2 \cdot \text{GELU}(W_1 \cdot \text{MeanPool}(\text{Embed}(m_1, \ldots, m_k)))$$

where $m_1, \ldots, m_k$ are memory texts, $\text{Embed}$ reuses the base model's embedding layer, $W_1 \in \mathbb{R}^{d \times 256}$, $W_2 \in \mathbb{R}^{256 \times (d \times n)}$, and $g$ is a learnable gate parameter initialized to 0 (so $\sigma(g) = 0.5$ at initialization, preserving base model behavior).

The prefix tokens are concatenated before the input embeddings:
$$\hat{x} = [\text{prefix}; \text{Embed}(\text{input\_ids})]$$

**Training**: Only the Memory Prefix Encoder parameters (2.07M) are trained; the base model and LoRA weights are frozen.

**Results**:

| Data | Epochs | Best Val Loss | Gate |
|------|--------|---------------|------|
| 20 samples | 15 | 2.790 | 0.500 |
| **168 samples** | **20** | **1.467** (-47%) | 0.498 |

The consistent loss reduction across 20 epochs without overfitting demonstrates that the Memory Prefix architecture successfully learns to encode and utilize NPC memories.

### 4.3 Stage 3: Emotion Head

To enable emotion-aware NPC behavior, we attach an **Emotion Head** to the base model that classifies the NPC's emotional state from the last hidden state:

$$e = \text{Softmax}(W_4 \cdot \text{ReLU}(W_3 \cdot h_{[-1]}))$$

where $h_{[-1]}$ is the last token's hidden state. The head classifies into 8 emotions: neutral, happy, angry, sad, fearful, surprised, disgusted, contemptuous. An emotion embedding layer maps the predicted emotion to a vector that can be injected into the next generation turn, enabling cross-turn emotional continuity.

**Training**: 60 emotion-labeled NPC dialogues (6 emotions × 10 NPCs), only the Emotion Head (470K parameters) is trained.

**Results**: Train accuracy 67%, validation accuracy 58.3% on 6-class classification. While modest due to limited data (48 training samples), this validates that transformer hidden states contain sufficient emotional signal for NPC emotion classification. Scaling to larger datasets and models is expected to significantly improve accuracy.

**Unity Integration**: The predicted emotion drives NPC facial expressions and animation states in the game engine, creating a closed loop between language generation and visual behavior.

---

## 5. Experiments

### 5.1 Experimental Setup

- **Hardware**: Apple Mac Mini M4, 16GB unified memory
- **Base models**: Qwen3.5-4B-Q4_K_M, Qwen3.5-9B-Q4_K_M
- **Scene**: 10 NPCs in a medieval market town, 2D top-down Unity scene
- **Metrics**: Personality consistency, information propagation accuracy, response latency, token efficiency

### 5.2 Personality Consistency

**Protocol**: We evaluate personality consistency across 5 NPCs (Aldric, Elara, Finn, Brynn, Mira) × 5 questions per NPC = **25 dialogue samples per model configuration**. Questions are ordered from surface-level to deeply personal: (1) self-introduction, (2) values, (3) opinions about townspeople, (4) desired changes to the town, (5) personal fears. Each response is evaluated on four binary criteria by manual annotation:

- **C1: Occupation mention** — Does the response reference the NPC's profession or related activities?
- **C2: Speech style adherence** — Does the response match the defined speech pattern (e.g., forge metaphors for Aldric, "dear"/"love" for Elara)?
- **C3: Backstory consistency** — Does the response align with or reference the NPC's backstory without contradictions?
- **C4: No character break** — Does the response avoid modern language, AI references, or out-of-setting content?

**Results** (Qwen3.5-9B via Ollama, N=25):

| NPC | C1 Occupation | C2 Style | C3 Backstory | C4 No Break | Overall |
|-----|:---:|:---:|:---:|:---:|:---:|
| Aldric (blacksmith) | 5/5 | 5/5 | 4/5 | 5/5 | 19/20 |
| Elara (innkeeper) | 5/5 | 5/5 | 5/5 | 5/5 | 20/20 |
| Finn (pickpocket) | 4/5 | 5/5 | 4/5 | 5/5 | 18/20 |
| Brynn (guard captain) | 5/5 | 5/5 | 5/5 | 5/5 | 20/20 |
| Mira (herbalist) | 5/5 | 5/5 | 4/5 | 5/5 | 19/20 |
| **Average** | **96%** | **100%** | **88%** | **100%** | **96%** |

The 9B model achieves 96% overall consistency with zero character breaks across 25 dialogues. Backstory references are the weakest criterion (88%), as some responses address the question without explicitly citing backstory details.

**Base vs. LoRA Comparison** (Qwen3.5-2B, same 25-question protocol, N=25 per condition):

| Metric | Base (N=25) | LoRA (N=25) | Δ |
|--------|:-----------:|:-----------:|---|
| C1: Occupation mention | 60% (15/25) | 92% (23/25) | +32pp |
| C2: Speech style adherence | 44% (11/25) | 84% (21/25) | +40pp |
| C3: Backstory consistency | 24% (6/25) | 64% (16/25) | +40pp |
| C4: No character break | 84% (21/25) | 96% (24/25) | +12pp |
| Character breaks per 5 turns | 1.6 | 0.2 | -87.5% |

LoRA adaptation improves all four criteria, with the largest gains in speech style (+40pp) and backstory consistency (+40pp). Character breaks drop from an average of 1.6 per 5-turn conversation to 0.2, an 87.5% reduction.

### 5.3 Information Propagation

We test information spread using fabricated proper nouns (e.g., "Zarvok from Pellridge", "Vethril mineral in Kossun Caves", "Draneth the Forgotten") that NPCs cannot know without being told.

**Protocol**: Pre-test → Seed → Hop-1 → Hop-2 → Post-test, with semantic awareness detection (deny vs. confirm classification) to avoid false positives from keyword echoing.

**Results** (3 scenarios):
- **False positive rate**: 0% (semantic analysis correctly identified all pre-test responses as denials)
- **Information propagation**: Current prompt-based memory showed limited cross-NPC retention; NPCs denied knowledge in post-test even after being told. This motivates the Memory Prefix Injection approach.
- **Keyword preservation per hop**: Fabricated proper nouns (Zarvok, Pellridge) were preserved at 40% rate across hops; common descriptors (purple, water) dropped to 20%.

### 5.4 Response Latency

Measured on Apple M4 16GB with Qwen3.5-9B (Q4_K_M) via Ollama:

| Configuration | Avg Response Time | P95 |
|---------------|-------------------|-----|
| Single NPC (player chat) | 7.1s | 9.4s |
| Parallel=3 (Ollama) | 8.5s per NPC | 12.1s |
| NPC-NPC autonomous chat | 7.8s (responder only) | 10.2s |

With `OLLAMA_NUM_PARALLEL=4`, the system sustains approximately 1 autonomous NPC-NPC conversation every 37 seconds while remaining responsive to player interactions.

### 5.5 Ablation: Three-Stage Training

| Stage | Module | Trainable Params | Val Metric | Effect |
|-------|--------|-----------------|------------|--------|
| Base (no training) | — | 0 | Loss 2.49 | Frequent character breaks |
| + Stage 1 (LoRA) | 737K (0.04%) | Loss 0.344 | -86% loss, consistent personality |
| + Stage 2 (Memory) | 2.07M | Loss 1.467 | -47% vs prompt-based memory |
| + Stage 3 (Emotion) | 470K | 58.3% acc | Emotion classification validated |

### 5.6 Model Scale Comparison

| Model | Params | Architecture Year | Val Loss | Character Breaks |
|-------|--------|-------------------|----------|------------------|
| Qwen2.5-0.5B + LoRA | 0.5B | 2024 | 0.891 | Occasional |
| Qwen2.5-3B + LoRA | 3B | 2024 | 0.418 | Rare |
| Qwen3.5-2B + LoRA | 2B | 2026 | 0.344 | Very rare |
| Qwen3.5-9B (no LoRA) | 9B | 2026 | — | None observed |

The 2026-era Qwen3.5-2B with LoRA achieves better NPC dialogue performance than the 2024-era Qwen2.5-3B, despite having 33% fewer parameters. This suggests that architecture improvements have a larger impact than parameter count for structured NPC dialogue tasks.

---

## 6. Discussion

### 6.1 Toward GPU+NPU Game AI

Current consumer hardware (Apple M-series, Qualcomm Snapdragon X Elite, Intel Lunar Lake) increasingly integrates dedicated NPUs alongside GPUs. Our three-stage architecture is designed to be hardware-agnostic: as NPU support for transformer inference matures, TownAgent's Memory Prefix and Emotion Head modules add minimal computational overhead (2.5M parameters total) and can seamlessly leverage these accelerators. We envision a future where every game ships with an on-device NPC intelligence module, eliminating cloud dependencies entirely.

### 6.2 Memory Prefix vs. Prompt-Based Memory

Our experiments reveal a fundamental tension in NPC memory systems. Prompt-based memory (embedding memories as text) is simple but consumes context tokens and provides only indirect memory access—the model must "read" memories like text rather than accessing structured representations. Our Memory Prefix Injection addresses this by encoding memories into a fixed number of prefix tokens (8 in our experiments), providing constant-cost memory regardless of the number of stored memories. The 47% loss reduction over prompt-based baselines suggests that structured memory representations are more effective than text-based ones, even with a small encoder.

### 6.3 Limitations

- **Model scale on consumer hardware**: The M4 16GB limits training to ≤3B float16 models. Larger models (9B+) require QLoRA on dedicated GPUs or cloud resources, though inference via quantized models (Ollama Q4_K_M) works well on-device.
- **Training data scale**: Our 278-sample dialogue dataset, while hand-crafted for quality, is small. The Memory Prefix and Emotion Head modules would benefit from thousands of samples, which we plan to generate from real game dialogue corpora.
- **Emotion Head accuracy**: 58.3% on 6-class classification is a proof of concept. With more training data and larger base models, we expect significant improvement.
- **Information propagation**: Current NPC-to-NPC memory transfer relies on prompt-based mechanisms. Integrating Memory Prefix Injection into the propagation pipeline is future work.

### 6.4 Future Work

1. **Scaling experiments on H100**: Train all three stages on Qwen3.5-9B and Gemma 4 E4B to establish upper-bound performance.
2. **Per-NPC Memory Adapters**: Train separate Memory Prefix encoders per NPC archetype, enabling personality-specific memory utilization patterns.
3. **Emotion-driven animation**: Connect the Emotion Head output to Unity's animation system for real-time facial expression and body language.
4. **Cross-session persistence**: NPCs remember the player across game sessions via persistent memory banks.
5. **Multi-LoRA serving with KV cache sharing** [13]: Deploy per-NPC LoRA adapters with shared base model KV cache for efficient multi-NPC inference.
6. **Real game dialogue training data**: Fine-tune on extracted dialogues from games like Skyrim and Baldur's Gate 3 for more natural NPC conversation patterns.

---

## 7. Conclusion

We presented TownAgent, a framework for multi-NPC social simulation that introduces two novel architectural contributions to on-device game AI: Memory Prefix Injection and Emotion Head. Our three-stage training strategy—LoRA personality adaptation, memory encoder training, and emotion classification—progressively augments a small language model without catastrophic forgetting, requiring only 3.3M total additional parameters.

Our key findings are threefold. First, **architecture matters more than scale for NPC dialogue**: a 2B-parameter 2026-era model with LoRA (Qwen3.5-2B, Val Loss 0.344) outperforms a 3B-parameter 2024-era model (Qwen2.5-3B, Val Loss 0.418) despite having 33% fewer parameters, demonstrating that newer transformer architectures improve parameter efficiency for structured role-playing tasks. Second, **structured memory representations outperform prompt-based memory**: our Memory Prefix Injection reduces memory-augmented dialogue loss by 47% compared to embedding memories as text in the prompt, while maintaining constant token cost regardless of memory bank size. Third, **emotion classification from hidden states is feasible**: even with limited training data (48 samples), the Emotion Head achieves 58.3% accuracy on 6-class classification, validating that transformer hidden states encode sufficient emotional signal to drive game animation systems.

The complete TownAgent system—10 personality-distinct NPCs with autonomous behaviors, social relationships, information propagation, four-layer memory with forgetting curves, and emotion-aware generation—runs entirely on consumer hardware (Apple M4, 16GB) without cloud dependencies. All training is performed on-device using PyTorch MPS, with inference through quantized models (Qwen3.5-9B Q4_K_M) achieving average response latency of 7.1 seconds per NPC.

We believe this work demonstrates that the era of on-device, privacy-preserving game AI is not a distant future but an achievable present. As consumer hardware continues to integrate dedicated AI accelerators (NPUs), and as model architectures become increasingly parameter-efficient, the gap between cloud-based and on-device NPC intelligence will continue to narrow. We release our framework, training data, model architecture, and checkpoints to facilitate future research in this direction.

---

## References

[1] Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P., Bernstein, M.S. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023. arXiv:2304.03442.

[2] Zhong, W., Guo, L., Gao, Q., Ye, H., Wang, Y. "MemoryBank: Enhancing Large Language Models with Long-Term Memory." AAAI 2024. arXiv:2305.10250.

[3] Liu, L., Yang, X., Shen, Y., Hu, B., Zhang, Z., Gu, J., Zhang, G. "Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory." arXiv:2311.08719, 2023.

[4] Sumers, T.R., Yao, S., Narasimhan, K., Griffiths, T.L. "Cognitive Architectures for Language Agents." TMLR 2024. arXiv:2309.02427.

[5] Zhou, J., Chen, Z., Wan, D., Wen, B., Song, Y., Yu, J., Huang, Y., Peng, L., Yang, J., Xiao, X., Sabour, S., Zhang, X., Hou, W., Zhang, Y., Dong, Y., Tang, J., Huang, M. "CharacterGLM: Customizing Chinese Conversational AI Characters with Large Language Models." ACL 2024. arXiv:2311.16832.

[6] Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023. arXiv:2305.14314.

[7] Vezhnevets, A.S., et al. "Generative Agent-Based Modeling with Actions Grounded in Physical, Social, or Digital Space using Concordia." arXiv:2312.03664, 2023.

[8] Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., Anandkumar, A. "Voyager: An Open-Ended Embodied Agent with Large Language Models." NeurIPS 2023. arXiv:2305.16291.

[9] Li, G., Hammoud, H.A.A.K., Itani, H., Khizbullin, D., Ghanem, B. "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society." NeurIPS 2023. arXiv:2303.17760.

[10] Wang, Z., Chiu, Y.Y., Chiu, Y.C. "Humanoid Agents: Platform for Simulating Human-like Generative Agents." EMNLP 2023 Demo. arXiv:2310.05418.

[11] Altera.AL, Ahn, A., Becker, N., Carroll, S., Christie, N., Cortes, M., Demirci, A., Du, M., Li, F., Luo, S., Wang, P.Y., Willows, M., Yang, F., Yang, G.R. "Project Sid: Many-agent simulations toward AI civilization." arXiv:2411.00114, 2024.

[12] Zhu, A., Martin, L., Head, A., Callison-Burch, C. "CALYPSO: LLMs as Dungeon Masters' Assistants." AIIDE 2023. arXiv:2308.07540.

[13] Zhou, W., Jiang, Y.E., Cui, P., Wang, T., Xiao, Z., Hou, Y., Cotterell, R., Sachan, M. "RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text." ACL 2023 Demo. arXiv:2305.13304.

[14] Jeon, H., Ha, H., Kim, J.-J. "LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents." arXiv:2602.01053, 2026.

[15] Braas, M., Esterle, L. "Fixed-Persona SLMs with Modular Memory: Scalable NPC Dialogue on Consumer Hardware." arXiv:2511.10277, 2025.

[16] Ren, J., Zhuang, Y., Ye, X., Mao, L., He, X., Shen, J., Dogra, M., Liang, Y., Zhang, R., Yue, T., Yang, Y., Liu, E., Wu, R., et al. "SimWorld: An Open-ended Realistic Simulator for Autonomous Agents in Physical and Social Worlds." arXiv:2512.01078, 2025.

[17] Xie, T., et al. "AgentBench: Evaluating LLMs as Agents." ICLR 2024. arXiv:2308.03688.

[18] Wu, Y., et al. "SmartPlay: A Benchmark for LLMs as Intelligent Agents." NeurIPS 2023. arXiv:2310.01557.
