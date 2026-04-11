# TownAgent: Memory-Augmented Small Language Models with Emotion-Aware Generation for Multi-NPC Social Simulation

[![Paper](https://img.shields.io/badge/Paper-Draft-blue)](paper/draft_v1.md)
[![Unity](https://img.shields.io/badge/Unity-6000.3.11f1-green)](unity/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#license)

## Overview

TownAgent is an end-to-end framework for multi-NPC social simulation powered by on-device small language models. It features:

- **Memory Prefix Injection**: Encodes NPC episodic memories into learnable prefix tokens, replacing token-expensive prompt-based memory
- **Emotion Head**: Classifies NPC emotional states from hidden representations for cross-turn emotional continuity
- **Three-Stage Training**: LoRA → Memory → Emotion, progressively adding capabilities without catastrophic forgetting
- **10 Personality-Distinct NPCs**: Blacksmith, innkeeper, pickpocket, guard captain, herbalist, hunter, priestess, fisherman, baker, merchant
- **Fully On-Device**: All training and inference on Apple M4 16GB, no cloud dependencies

## Architecture

```
Layer 4: Unity Game Layer (10 NPCs, market town, day/night cycle)
Layer 3: NPCAgent Framework (social graph, scheduler, autonomous behavior)
Layer 2: Memory Prefix Injection (memory encoder → prefix tokens → input)
Layer 1: Emotion Head (hidden state → 8-class emotion → next turn injection)
Layer 0: Base Model + LoRA (Qwen2.5/Qwen3.5, on-device inference)
```

## Key Results

| Experiment | Result |
|------------|--------|
| LoRA personality adaptation (Qwen3.5-2B, 278 samples) | Val Loss **0.344** |
| Data quality vs quantity | 278 hand-crafted (0.34) >> 8K filtered (1.76) >> 81K mixed (2.01) |
| Memory Prefix Injection | **-47%** loss vs prompt-based memory |
| Emotion Head (8-class) | **58.3%** validation accuracy |
| NPC personality consistency (5 rounds) | **96%** across 4 criteria |
| Response latency (Qwen3.5-9B, M4) | **7.1s** average |

## Project Structure

```
NPCAI/
├── unity/                    # Unity project files
│   └── Assets/Scripts/       # C# NPC system
│       ├── LLM/              # OllamaClient
│       ├── NPC/              # NPCBrain, Memory, SocialGraph, Behavior, Scheduler
│       ├── UI/               # SpeechBubble
│       └── Editor/           # AutoRefresh, AutoPlayMode, MarketSceneSetup
├── model/                    # Python model architecture & training
│   ├── npc_model.py          # NPCModel (LoRA + Memory Prefix + Emotion Head)
│   ├── train_stage1.py       # Stage 1: LoRA training
│   ├── train_stage2.py       # Stage 2: Memory Prefix training
│   └── train_stage3.py       # Stage 3: Emotion Head training
├── data/                     # Training datasets
│   └── training_data/
│       ├── curated/          # 8K high-quality filtered data
│       ├── stage1_large/     # 81K mixed data
│       ├── stage2_large/     # 10K memory data
│       └── stage3_large/     # 12K emotion data
├── checkpoints/              # Trained model checkpoints
│   ├── stage1/               # 0.5B LoRA
│   ├── stage1_3b/            # 3B LoRA
│   ├── stage1_qwen35_2b/     # Qwen3.5-2B LoRA (best)
│   ├── stage2_memory/        # Memory Prefix encoder
│   └── stage3_emotion/       # Emotion Head
├── experiments/              # Experiment scripts & results
│   ├── experiment_v2.py      # Information propagation
│   ├── benchmark_emotion.py  # Emotion coherence benchmark
│   └── benchmark_social.py   # Social dynamics benchmark
├── benchmarks/               # Benchmark specs & test sets
│   └── benchmark_spec.md     # 6-dimension evaluation protocol
├── paper/                    # Paper draft & knowledge base
│   ├── draft_v1.md           # Full paper draft
│   └── knowledge_base.md     # References & technical notes
└── .claude/                  # Claude Code memory & rules
    └── memory/               # Project memory files
```

## Hardware Requirements

- **Training**: Apple M4 16GB (MPS) or NVIDIA GPU 8GB+ (CUDA)
- **Inference**: Apple M4 16GB with Ollama (Qwen3.5-9B Q4_K_M)
- **Unity**: Unity 6000.3.11f1

## Quick Start

### 1. Setup Unity Project
```bash
# Open unity/ folder in Unity Hub (6000.3.11f1)
# MCP Unity package is pre-configured
```

### 2. Start Ollama
```bash
ollama serve
ollama run qwen3.5:9b  # First time: downloads model
```

### 3. Train LoRA (Stage 1)
```bash
cd model/
python train_stage1.py  # Uses MPS on Mac, CUDA on Windows
```

### 4. Run Experiments
```bash
# Start Unity Play Mode, then:
python experiments/experiment_v2.py      # Propagation test
python experiments/benchmark_emotion.py  # Emotion benchmark
python experiments/benchmark_social.py   # Social dynamics
```

## 10 NPCs

| NPC | Role | Speech Style | Key Trait |
|-----|------|-------------|-----------|
| Aldric | Blacksmith | Forge metaphors, short | Protective, mourning wife |
| Elara | Innkeeper | "Dear"/"love", chatty | Gossip, warm-hearted |
| Finn | Pickpocket | Slang, humor | Secretly kind, feeds orphans |
| Brynn | Guard Captain | Military formal | Honor, duty-focused |
| Mira | Herbalist | Plant metaphors, gentle | Senses forest darkness |
| Thorne | Hunter | Few words, precise | Observant, lone wolf |
| Sister Helene | Priestess | Calm, scripture | Troubled visions |
| Old Bertram | Fisherman | Sea metaphors, slow | Philosophical, saw serpent |
| Lydia | Baker | Food metaphors, motherly | Protects children |
| Garrett | Merchant | Numbers, smooth | Shrewd but generous |

## References

See [paper/draft_v1.md](paper/draft_v1.md) for full references. Key papers:

1. Park et al. "Generative Agents" (UIST 2023) - Foundation work
2. Zhong et al. "MemoryBank" (AAAI 2024) - Ebbinghaus forgetting curves
3. Dettmers et al. "QLoRA" (NeurIPS 2023) - Efficient fine-tuning
4. Braas & Esterle. "Fixed-Persona SLMs" (arXiv 2025) - NPC dialogue on consumer hardware

## License

MIT License
