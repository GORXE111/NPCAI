# Skill-as-Tool: Grounding Small Language Models in In-Character Decision Tools for Game NPCs

[![Paper](https://img.shields.io/badge/Paper-v3_Draft-blue)](paper/draft_v3.md)
[![Model](https://img.shields.io/badge/Base-Qwen3.5--0.8B-orange)](https://huggingface.co/Qwen/Qwen3.5-0.8B)
[![Hardware](https://img.shields.io/badge/Trained%20on-Apple%20M4%2016GB-purple)](https://www.apple.com/mac-mini/)
[![Unity](https://img.shields.io/badge/Unity-6000.3.11f1-green)](unity/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#license)

## TL;DR

We train **Qwen3.5-0.8B** to play [Lieutenant Kim Kitsuragi](https://discoelysium.fandom.com/wiki/Kim_Kitsuragi) (*Disco Elysium*) as a **tool-using game NPC** entirely on consumer hardware (Apple M4 Mac, 16GB). The model emits structured `{dialogue, tool_calls}` JSON that drives a Unity visual-novel environment — including character expression changes, BGM cross-fades, scene transitions, and the eponymous **Disco Elysium skill checks** mapped 1:1 to LLM tool calls. We position against **CPDC 2025** winners (Qwen3-**14B** on data-center GPUs) at **17× fewer parameters**.

---

## What Makes This Different

| | Cloud NPC systems (CPDC 2025 winners) | TownAgent (this work) |
|---|---|---|
| Base model | Qwen3-**14B** | Qwen3.5-**0.8B** |
| Hardware | L40S / H100 GPUs | Apple M4 Mac, 16GB |
| Tool grounding | Abstract APIs | **Real Unity engine execution** (animation/BGM/portraits) |
| Tool taxonomy | Hand-designed | **Mined from a published RPG** (24 *Disco Elysium* skills + 9 VN engine tools) |
| Anchor character | Generic NPCs | Kim Kitsuragi (one of gaming's most distinctive NPCs) |

**Why Disco Elysium?** The game's 24 internal skills (Logic, Empathy, Visual Calculus, Shivers, Half Light, …) are *literally* tool calls from a system perspective: each dialogue branch is gated by a skill check that returns success/fail. The community-released conversation corpus encodes these inline as `[Action/Check: Condition: …]` tags — giving us **free supervision** for tool-calling training.

---

## Architecture

```
L4. Unity VN Environment           Disco Elysium-styled visual novel
                                   (oil-painting BG + character portraits + dialogue box + choice panel)
       ↑ executes
L3. ToolRegistry & ToolExecutor    18 tools registered, JSON dispatch + schema validation
       ↑ {dialogue, tool_calls} JSON
L2. NPCAgent (Kim)                 SceneState + memory + LLM orchestration
       ↑ chat with tool schema in system prompt
L1. Qwen3.5-0.8B + LoRA            Stage 1 (persona) + Stage 2 (tools) + Stage 3 (DPO)
                                   Served via PyTorch MPS / Ollama
```

## Three-Stage Training (the methodology)

| Stage | Method | What it teaches | Data | Result |
|-------|--------|-----------------|:-----:|--------|
| **1: Persona-SFT** | LoRA r=16 | Kim's **voice** (form: speech style) | 1,127 cleaned Kim utterances from `output.json` | Val Loss **0.945** |
| **2: Tool-aug SFT** | LoRA continued | Tool-call **format** (form: structured JSON) | 1,064 dialogue+tool_calls samples | Val Loss **0.725**, JSON 100% valid, tool selection ~25% |
| **3: DPO** | preference optimization | Tool-call **correctness** (when to call, when not to) | 3,337 balanced preference pairs | _Currently training with corrected data balance_ |

**Why three stages?** SFT optimizes *form*; it teaches Kim how to *speak* and how to *format* tool calls. **Correctness** — knowing when to invoke `Empathy` vs `Logic`, when to refuse a tool entirely — requires preference signal, which we provide via DPO. See [paper/draft_v3.md §5](paper/draft_v3.md) for details.

---

## Skill-as-Tool API (18 tools)

Tools are categorized by source and game effect:

```
🧠 Skill (Disco Elysium, ×24)
   skill_check(skill, message)        — invoke any of Logic, Empathy, Visual Calculus,
                                        Shivers, Half Light, Inland Empire, Authority,
                                        Rhetoric, Drama, Conceptualization, Encyclopedia,
                                        Volition, Esprit de Corps, Suggestion, Endurance,
                                        Pain Threshold, Physical Instrument, Electrochemistry,
                                        Hand/Eye Coordination, Perception, Reaction Speed,
                                        Savoir Faire, Interfacing, Composure

🎭 Character (×3)
   set_expression(actor, emotion)     — swap portrait sprite
   show_character(actor, slot)        — bring on screen at left/center/right
   hide_character(actor)              — remove from screen

🖼️ Scene (×1)
   set_background(location)           — switch backdrop with crossfade

🔊 Audio (×2)
   play_bgm(track)                    — cross-fade BGM
   play_sfx(sound)                    — one-shot sound effect

💬 Flow (×3)
   present_choices(options)           — display 2-5 player choices
   narrate(text)                      — narrator voice (no character)
   end_scene(next_scene)              — scene transition
```

Full JSON schema in `unity/Assets/Scripts/Agent/BuiltinToolHandlers.cs` and [paper §A](paper/draft_v3.md).

---

## Key Findings

### 1. Data quality dominates quantity (replicated across 3 generations)
| Setup | Samples | Val Loss |
|------|---------|---------:|
| 81K mixed (Persona-Chat + WoW + RPG dump) | 81,036 | 2.011 |
| 8K curated (LIGHT + amaydle filtered) | 8,131 | 1.757 |
| 1.6K Kim raw (with metadata pollution) | 1,587 | 1.086 |
| **1.1K Kim cleaned (quoted speech only)** | **1,127** | **0.945** |

### 2. SFT teaches *form*, not *correctness*
- Stage 2 → 100% JSON schema validity
- Stage 2 → only ~25% tool selection accuracy on positive cases
- Conclusion: SFT alone is insufficient for game NPC tool use; need RL/DPO.

### 3. DPO data balance is critical (negative result, paper §B)
Our first DPO run (2,789 pairs, 64% taught "remove tool") triggered **distributional collapse**: tool selection dropped from 25% to 0%.
Fix: add `F0_missing_tool` perturbations to teach "when context demands a tool, you must call it" (724 balancing pairs added).

### 4. MPS dtype quirk in Qwen3.5
Qwen3.5's `Qwen3_5GatedDeltaNet` (linear-attention) layer triggers MPS mixed-dtype matmul crashes. We release [`model/qwen35_mps_fix.py`](model/qwen35_mps_fix.py) — a forward-pass wrapper that coerces float32 throughout the layer.

---

## Project Structure

```
NPCAI/
├── paper/
│   ├── draft_v3.md                  # Current paper (Skill-as-Tool framework)
│   ├── draft_v2.md                  # Prior version (TownAgent multi-NPC)
│   └── draft_v1.md                  # First version (preserved for history)
│
├── unity/Assets/Scripts/
│   ├── VN/                          # Visual novel UI (DialogueBox, Portrait, Choice, Audio, BG)
│   ├── Agent/                       # ToolRegistry, ToolDefinition, AgentResponse, PromptBuilder, BuiltinToolHandlers
│   ├── LLM/                         # OllamaClient with structured JSON output
│   └── _Archive/                    # Old market-town code (paper v0/v1)
│
├── model/
│   ├── qwen35_mps_fix.py            # Apple MPS compat patch
│   ├── train_kim_lora.py            # Stage 1: persona SFT
│   ├── train_stage2_kim.py          # Stage 2: tool-augmented SFT
│   ├── train_dpo_kim.py             # Stage 3: DPO
│   ├── test_kim_persona.py          # Stage 1 reproduction test
│   └── test_stage2_tool.py          # Stage 2/3 tool selection test
│
├── data/disco_elysium/
│   ├── output.json                  # 1,742 raw conversations (allura-org HF)
│   ├── prepare_kim_data_v2.py       # Stage 1 data prep (quoted speech extraction)
│   ├── prepare_stage2_data_v2.py    # Stage 2 data prep (tool inference rules)
│   ├── generate_dpo_synthetic.py    # Phase 3A: 9 perturbations
│   ├── generate_dpo_balanced.py     # Phase 3A fix: F0_missing_tool balancing
│   ├── training_v2/                 # 1,127 Kim SFT samples
│   ├── training_stage2/             # 1,064 tool-augmented samples
│   └── training_dpo/                # 3,337 balanced preference pairs
│
├── checkpoints/
│   ├── kim_q35_08b_v2/lora/         # Stage 1 v2 (Val 0.945)
│   ├── kim_q35_08b_stage2/lora/     # Stage 2 (Val 0.725)
│   ├── kim_q35_08b_stage3/lora/     # Stage 3 v1 (collapsed; preserved as negative result)
│   └── kim_q35_08b_stage3_v2/lora/  # Stage 3 v2 (training in progress)
│
├── benchmarks/                      # CPDC 2025 / BFCL / DEBench (DEBench in progress)
│
└── .claude/rules/                   # Project knowledge base (read first when joining)
    ├── README.md                    # Index
    ├── retrospective_process.md     # Self-audit rules
    ├── knowledge_paper_evolution.md # v0 → v3 direction history
    ├── knowledge_methodology.md     # SFT / Memory / DPO deep dive
    ├── knowledge_npc_research.md    # 2025-2026 benchmark landscape
    ├── knowledge_training_gotchas.md # MPS / LoRA / data quality gotchas
    └── knowledge_deliverables.md    # 19-item delivery tracker (~53% complete)
```

---

## Quick Start

### 1. Run inference with the trained Kim agent (Mac M4)

```bash
# Apply MPS patch + load Stage 2 (or Stage 3 v2 once finished)
python model/test_stage2_tool.py
# Outputs: 7 DE-style scenarios, base vs LoRA tool selection comparison
```

### 2. Train your own from raw data

```bash
# Step 1: extract Kim quoted speech from output.json
python data/disco_elysium/prepare_kim_data_v2.py

# Step 2: persona SFT
python model/train_kim_lora.py
# → checkpoints/kim_q35_08b_v2/lora (Val ≈ 0.95)

# Step 3: tool-augmented SFT
python data/disco_elysium/prepare_stage2_data_v2.py
python model/train_stage2_kim.py
# → checkpoints/kim_q35_08b_stage2/lora

# Step 4: DPO on balanced preferences
python data/disco_elysium/generate_dpo_synthetic.py
python data/disco_elysium/generate_dpo_balanced.py
python model/train_dpo_kim.py
# → checkpoints/kim_q35_08b_stage3_v2/lora
```

### 3. Open the Unity VN demo

```bash
# Open unity/ in Unity Hub (6000.3.11f1+)
# Configure OllamaClient to point to your local Ollama OR a Python inference server hosting the trained LoRA
# Hit Play in the demo scene
```

### 4. Run the benchmarks

```bash
# CPDC 2025 (clone separately)
git clone https://gitlab.aicrowd.com/cpdc-2025
python benchmarks/run_cpdc.py --model checkpoints/kim_q35_08b_stage3_v2/lora

# BFCL V4 (multi-turn function calling)
python benchmarks/run_bfcl.py --model checkpoints/kim_q35_08b_stage3_v2/lora

# DEBench (ours, Disco Elysium-grounded)
python benchmarks/run_debench.py
```

---

## Hardware Requirements

- **Training**: Apple M4 16GB (MPS) — all stages tested.
  - Stage 1: ~75 min / 1,127 samples / 4 epochs
  - Stage 2: ~75 min / 1,064 samples / 4 epochs
  - Stage 3: ~150 min / 3,337 pairs / 1 epoch (DPO with frozen ref + trainable policy)
- **Inference**: Apple M4 16GB. Average per-turn latency: ~1.5s with Stage 2 LoRA.
- **Unity**: Unity 6000.3.11f1+

---

## Benchmarks

We evaluate on three benchmarks (one ours, two external):

1. **DEBench (ours)** — Disco Elysium-grounded scenarios scoring (a) persona fidelity, (b) skill-tool selection, (c) game-tool call F1.
2. **[CPDC 2025](https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025)** — *Commonsense Persona-Grounded Dialogue Challenge* (EMNLP 2025 Wordplay Workshop). 4-axis LLM-as-Judge: Scenario Adherence / NPC Believability / Persona Consistency / Dialogue Flow. Winner systems use Qwen3-14B + LoRA / GRPO on data-center GPUs ([MSRA_SC](https://arxiv.org/abs/2511.20200), [Deflanderization](https://arxiv.org/abs/2510.13586), [Multi-Expert](https://arxiv.org/abs/2511.01720)).
3. **[BFCL V4](https://gorilla.cs.berkeley.edu/leaderboard.html)** — Berkeley Function Calling Leaderboard, multi-turn agentic categories.

---

## Concurrent Work

We acknowledge two concurrent works in the same niche; our paper differentiates as follows:

- **[Fixed-Persona SLMs with Modular Memory](https://arxiv.org/abs/2511.10277)** (Braas & Esterle, Nov 2025) — Same goal (SLM NPCs on consumer hardware), but uses DistilGPT-2 / TinyLlama / Mistral-7B with prompt-based memory, **no tool calling**, no game-engine grounding. We add tool-augmented training, DPO faithfulness optimization, and Unity engine execution.
- **[Akoury et al.](https://aclanthology.org/2023.findings-emnlp.151.pdf)** (EMNLP 2023 Findings) — Used Disco Elysium's 1.1M-word script with GPT-4 for dialogue infilling and player perception study. We are first to use the corpus for **SLM training** and **tool-use grounding**.

---

## References

See [paper/draft_v3.md](paper/draft_v3.md) for the full reference list. Key papers:

1. Park et al. "Generative Agents." UIST 2023. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
2. Rafailov et al. "Direct Preference Optimization." NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
3. Hu et al. "LoRA: Low-Rank Adaptation." ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
4. Tu et al. "CharacterEval." ACL 2024. [arXiv:2401.01275](https://arxiv.org/abs/2401.01275)
5. Yan et al. "Berkeley Function Calling Leaderboard V4." 2025-2026. [Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
6. Yao et al. "τ²-bench: Multi-turn Conversational Agents under Dual Control." [arXiv:2506.07982](https://arxiv.org/abs/2506.07982)

---

## Citation

```bibtex
@misc{npcai2026skillastool,
  title  = {Skill-as-Tool: Grounding Small Language Models in In-Character
            Decision Tools for Game NPCs},
  author = {[Authors]},
  year   = {2026},
  url    = {https://github.com/GORXE111/NPCAI}
}
```

---

## License

Code and data: MIT License. *Disco Elysium* dialogue text is © ZA/UM and is used under fair use for academic research; the community-extracted corpus on which we train is publicly hosted at `allura-org/disco-elysium-conversations-raw` and `main-horse/disco-elysium-utterances` on HuggingFace. Trained checkpoints are released under MIT.

## Acknowledgments

To ZA/UM for *Disco Elysium*. To the Disco Elysium community for the data-mined conversation corpora that made this work possible.
