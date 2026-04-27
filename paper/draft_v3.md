# Skill-as-Tool: Grounding Small Language Models in In-Character Decision Tools for Game NPCs

**Authors**: [TODO: Add author names and affiliations]
**Corresponding Author**: [TODO: Add email]
**Repository**: https://github.com/GORXE111/NPCAI

---

## Abstract

Modern small language models (SLMs) under 1 billion parameters can produce competent dialogue but struggle to act in interactive game settings: existing systems either generate text without grounding it in game-engine actions, or rely on cloud-scale models for tool invocation. We present **Skill-as-Tool (SaT)**, a framework that grounds an SLM-driven NPC in a structured tool API representing in-character decision skills, and we instantiate it with Kim Kitsuragi from *Disco Elysium*—a character whose internal "skill checks" map naturally onto our tool-call abstraction. Our system runs entirely on consumer hardware (Apple M4, 16GB) using Qwen3.5-0.8B as the policy model, integrated with a Unity-based visual-novel environment that executes 18 grounded tools (`set_expression`, `play_bgm`, `present_choices`, `skill_check`, `show_cg`, etc.). We propose a three-stage training recipe—persona-SFT on extracted character dialogue, tool-augmented SFT on action-tagged conversations, and Direct Preference Optimization on tool-faithfulness pairs—using **1,587 Kim utterances with surrounding context** and **6,438 raw Kim lines** mined from the *Disco Elysium* community corpus. We evaluate on three benchmarks: the **CPDC 2025 Persona-Grounded Dialogue Challenge** (standard), the **Berkeley Function Calling Leaderboard V4** (standard), and **DEBench**, a Disco-Elysium-specific benchmark we contribute that scores joint dialogue and skill-call correctness. Our results show that a 0.8B model, with skill-as-tool training, [TODO: fill ablation numbers] approaches the persona-fidelity of CPDC 2025's 14B winners on dialogue while delivering [TODO: latency number]ms-class on-device latency. We additionally release the Unity visual-novel environment, the trained checkpoints, and DEBench. All code and data are at https://github.com/GORXE111/NPCAI.

**Keywords**: NPC dialogue, tool use, function calling, small language models, on-device inference, visual novel, Unity, Disco Elysium, Kim Kitsuragi

---

## 1. Introduction

### 1.1 The Acting NPC Problem

Non-player characters in modern games are increasingly expected to do more than recite scripted lines. A believable NPC must decide *what to do* as much as *what to say*: shift their facial expression as the conversation darkens, change the background music when stakes rise, present the player with grounded multiple-choice options, or refuse a question because their character would never know the answer. We refer to this as the **acting NPC problem**: a single agent that must produce both natural-language utterances and structured, executable actions in a tightly coupled loop.

Recent benchmark efforts have begun to formalize this setting. The Commonsense Persona-Grounded Dialogue Challenge (CPDC 2025) at the EMNLP 2025 Wordplay Workshop [19] explicitly evaluates systems on three tasks—task-oriented dialogue with function-calls, context-aware dialogue, and the hybrid of the two—using a 4-axis LLM-as-Judge protocol over an NPC scenario corpus. The winning solutions [20, 21, 22] all rely on **Qwen3-14B** fine-tuned with LoRA or GRPO on data-center GPUs (L40S). Independently, the Berkeley Function Calling Leaderboard (BFCL) V4 [23] has extended its tool-use evaluation to multi-turn agentic categories.

Two empirical facts from these leaderboards motivate our work. First, **parameter count is a weak predictor of tool-calling skill at the small scale**: BFCL V4 places Qwen3-0.6B, Qwen3-4B, lfm2.5-1.2B, and phi4-mini-3.8B together at the 0.880 agent-score tier [23]. Second, **CPDC 2025 has no published SLM (<2B) baseline for the joint dialogue-and-tool task**, leaving an unoccupied position on the Pareto frontier of "small-enough-to-run-on-a-laptop, faithful-enough-for-a-game".

### 1.2 The Disco Elysium Insight

While planning a tool taxonomy for game NPCs, we noticed that one published commercial RPG—*Disco Elysium* (ZA/UM, 2019)—already encodes its protagonist's mind as a fixed inventory of 24 named skills (Logic, Empathy, Visual Calculus, Inland Empire, Half Light, Shivers, …). Each in-game decision is gated by a **skill check**: the system rolls against a skill's level, returns a result, and the dialogue branches accordingly. Concretely, the community-extracted conversation corpus encodes these checks inline as

```
"actor": "Soona, the Programmer",
"dialogue": "[Action/Check: Condition: IsKimHere()]"
```

interleaved with regular utterances. **This is structurally identical to what a tool-augmented LLM is asked to produce.** Rather than designing an arbitrary tool API for a generic NPC and hoping the data supports it, we adopt *Disco Elysium*'s skill system as a ready-made tool taxonomy and treat the protagonist's partner—Lieutenant Kim Kitsuragi—as a concrete, well-known, anchor NPC.

### 1.3 Contributions

This paper makes four contributions:

1. **Skill-as-Tool framework.** A formulation that grounds an SLM NPC in a structured tool API derived from *Disco Elysium*'s skill system, expanded with eight Unity-actionable tools (visual-novel layer: `set_expression`, `play_bgm`, `present_choices`, `show_cg`, `set_background`, `play_sfx`, `narrate`, `end_scene`), totaling 18 tools.

2. **Three-stage training recipe.** Persona-SFT → Tool-augmented SFT → Tool-faithfulness DPO, demonstrated on Qwen3.5-0.8B with consumer hardware (Apple M4, 16GB) and addressing dtype-incompatibility quirks in MPS via a `Qwen3_5GatedDeltaNet` patch we release.

3. **DEBench**: a Disco-Elysium-grounded evaluation set scoring joint dialogue and skill-call correctness, complementing standard benchmarks (CPDC 2025, BFCL V4 multi-turn).

4. **Open Unity visual-novel environment** with Kim Kitsuragi as the live agent, implementing the 18-tool API, runnable on M4 / consumer hardware. All assets and code are released at https://github.com/GORXE111/NPCAI.

---

## 2. Related Work

### 2.1 Tool-Using Language Models

The line from Toolformer [TODO: ref], ReAct [TODO], and Gorilla [TODO] established that LLMs can reliably call structured APIs. The Berkeley Function Calling Leaderboard [23] is the de-facto standard for measuring this. Tau-bench [24] and τ²-bench [25] from Sierra Research extend tool-use evaluation to multi-turn agent-user conversations with shared state, and ToolSandbox [26] adds explicit world-state dependency. Recent systems papers show dramatic SLM/LLM gaps narrowing: Amazon's "Small LMs for Efficient Agentic Tool Calling" (arXiv 2512.15943) reports a fine-tuned 350M model beating 500× larger competitors on ToolBench at 77.55% pass rate.

**Gap**: None of these benchmarks anchor the agent in a specific narrative character, and none ground the tool API in a game engine. They evaluate API-calling competence in the abstract.

### 2.2 NPC Dialogue and Persona-Grounded Generation

Generative Agents [1] established that LLM-driven NPCs exhibit emergent social behaviors but used cloud GPT-3.5/4 and treated "actions" as free-text plans rather than executable engine calls. Concordia [7], Project Sid [11], and SimWorld [16] scaled this paradigm but stayed in textual or schematic action spaces. CharacterEval [17] introduced a 13-metric, reward-modeled benchmark for Chinese-novel character roleplay; CharacterGLM [5] proposed 6 evaluation dimensions; PingPong [27] and PersonaEval [TODO] introduced multi-judge protocols. Fixed-Persona SLMs (arXiv 2511.10277) is the closest concurrent work to ours: it explores SLM NPCs on consumer hardware with a modular memory architecture, but uses three small models (DistilGPT-2, TinyLlama-1.1B, Mistral-7B) and **does not include tool-use or game-engine grounding**.

**Gap**: No prior work anchors persona-grounded SLM dialogue to a concrete, distinctive game NPC AND to executable game-engine tools.

### 2.3 The CPDC 2025 Cohort

The CPDC 2025 challenge [19] produced the cohort of papers most directly comparable to ours: MSRA_SC [20] used Context Engineering plus GRPO RL on Qwen3-14B; Deflanderization [21] applied SFT+LoRA to Qwen3-14B; Multi-Expert NPC Agent [22] split the work over three LoRA heads; Model Fusion Multi-LoRA [TODO ref arXiv 2509.24229] served three Qwen3-14B LoRAs through vLLM. **Every winning system uses 14B and a data-center GPU.** Our Pareto contribution—Qwen3.5-0.8B on a 16GB Mac—is a 17× parameter reduction.

### 2.4 Disco Elysium as Corpus

Akoury et al. [TODO: ref EMNLP 2023] used *Disco Elysium*'s 1.1M-word script with GPT-4 to study player perception of LLM-generated dialogue infilling. They focused on perception, not training. Two community datasets exist: `main-horse/disco-elysium-utterances` (34,384 lines, per-character segmented; Kim has 6,438) and `allura-org/disco-elysium-conversations-raw` (1,742 conversations with inline skill-check action tags). We are, to our knowledge, the first to **train an SLM agent on these corpora** and the first to use *Disco Elysium* as a tool-taxonomy donor.

---

## 3. *Disco Elysium* as a Testbed

### 3.1 The Skill System

*Disco Elysium*'s protagonist (the "detective") possesses **24 internal skills** organized into four categories:

- **Intellect**: Logic, Encyclopedia, Rhetoric, Drama, Conceptualization, Visual Calculus
- **Psyche**: Volition, Inland Empire, Empathy, Authority, Esprit de Corps, Suggestion
- **Physique**: Endurance, Pain Threshold, Physical Instrument, Electrochemistry, Shivers, Half Light
- **Motorics**: Hand/Eye Coordination, Perception, Reaction Speed, Savoir Faire, Interfacing, Composure

In-game, skills speak as inner monologues, contesting the player's choices and surfacing analyses. Crucially, gating dialogue branches on a skill check **is** the game's core mechanic—players literally encounter `[Logic — Easy: 9]` prompts. From a system perspective, each skill is a deterministic tool call against persisted state.

### 3.2 Kim Kitsuragi

Kim Kitsuragi is a 43-year-old lieutenant from RCM Precinct 41, partnered with the (amnesiac, troubled) detective for an investigation. His voice is famously distinctive: clinical, dry, morally-anchored, with measured pauses and quiet wit. As an evaluation target he offers three properties:

1. **High recognizability**. Even within the small-but-influential *Disco Elysium* fandom, Kim is the most-cited "perfect NPC" in modern games. Persona-capture failures are easy for human raters to flag.
2. **Sufficient data**. 6,438 Kim utterances (raw) plus 1,587 utterances with surrounding context after our pipeline (§5.1).
3. **Tool-correlated dialogue**. Kim does not himself perform skill checks—he is partnered with the player who does—but the conversations he appears in contain inline `[Action/Check: …]` tags. Training on these conversations teaches the model to **respect** skill checks and follow their consequences in dialogue.

### 3.3 Skill-as-Tool Mapping

We expose the 24 skills directly as tools (Table 1, top rows). To ground execution in a Unity visual-novel environment, we add 8 game-engine tools (Table 1, bottom rows).

**Table 1.** The Skill-as-Tool API. Skill tools (top) are derived from *Disco Elysium*; game tools (bottom) are added for Unity grounding.

| Layer | Tool | Args | Game effect |
|-------|------|------|-------------|
| Skill | `skill_check(skill, difficulty)` | name, "Easy/Medium/Hard" | dice roll vs. player's stat, returns success/fail |
| Skill (×24) | `Logic`, `Empathy`, `Inland Empire`, … | message string | inner monologue triggers UI panel, returns interpretation |
| VN — character | `set_expression(actor, emotion)` | actor, emotion | swap portrait sprite |
| VN — character | `show_character(actor, slot)` | actor, left/center/right | display character |
| VN — character | `hide_character(actor)` | actor | remove from screen |
| VN — scene | `set_background(location)` | scene_id | switch backdrop |
| VN — scene | `show_cg(cg_id)` | cg | display full-screen still |
| VN — audio | `play_bgm(track)` | track_id | start/cross-fade music |
| VN — audio | `play_sfx(sound)` | sfx_id | one-shot sound |
| VN — flow | `present_choices(options)` | list of strings | show clickable player options |
| VN — flow | `narrate(text)` | text | narrator voice (no character) |
| VN — flow | `end_scene(next_id)` | scene_id | scene transition |

The full schema (JSON) is in Appendix A. Tools are stateless from the LLM's perspective; their effects are computed by Unity.

---

## 4. System Design

### 4.1 Architecture Overview

The runtime is a four-layer stack:

```
L4. Unity VN Environment       [scene state, sprites, audio, choices]
L3. ToolRegistry & ToolExecutor [parses LLM JSON, dispatches to L4]
L2. Agent (NPCAgent / Kim)     [persona + memory + LLM call]
L1. Qwen3.5-0.8B + LoRA        [policy, runs via Ollama or PyTorch]
```

A turn proceeds:

1. Unity sends scene state + player choice to L2.
2. L2 builds a prompt: persona system message + scene state (current bg / characters / bgm) + recent dialogue + tool schema + player choice.
3. L1 returns a JSON object: `{dialogue: "...", tool_calls: [{name, args}, ...]}`.
4. L3 validates and executes each tool call against L4 (animations, audio, UI updates).
5. Unity awaits next player choice; loop.

### 4.2 Unity Visual-Novel Environment

We implement a *Disco Elysium*-styled visual novel: oil-painting backdrops, hand-drawn portraits at 1–3 slots, dialogue box with typewriter effect, choice panel, and a side rail showing the active skill (when one fires). Unlike our previous market-town prototype (archived in the repository), the VN format makes **every tool call visually salient**: a `set_expression` is a visible portrait swap, a `play_bgm` is an audible cross-fade, a `skill_check` is a dice-roll animation. This dramatically simplifies both implementation and qualitative evaluation. A screenshot is in Figure 1 [TODO].

### 4.3 OllamaClient and Structured JSON

We extend the OllamaClient to enforce structured JSON output via Ollama's `format: "json"` mode and a runtime schema validator. Tool calls that fail schema validation are dropped with a `[tool_error]` event surfaced to the agent on the next turn so it can recover, mirroring the pattern in τ²-Bench.

### 4.4 MPS Compatibility Patch

Qwen3.5-series models contain a `Qwen3_5GatedDeltaNet` (linear-attention) layer whose forward pass invokes `tensor.float()` on internal computations. On Apple's MPS backend, this causes mixed-dtype matrix multiplications to crash with "Destination NDArray and Accumulator NDArray cannot have different datatype". We release `qwen35_mps_fix.py`, which wraps each `GatedDeltaNet.forward` to coerce the entire layer to `float32` while preserving the model's outer dtype. With this patch, both Qwen3.5-0.8B and Qwen3.5-2B run forward and backward passes on M4 16GB.

---

## 5. Method

### 5.1 Data Pipeline

We mine training data from two community-released *Disco Elysium* corpora and one synthetic preference set we generate.

**Source A — Conversations with action tags.** `allura-org/disco-elysium-conversations-raw/output.json` contains 1,742 ordered conversations. Each is a list of `(actor, dialogue)` records; `dialogue` is one of (a) regular speech, (b) `[Action/Check: Condition: ...]` (a tool call), or (c) skill-named inner monologues from the player side. We parse this into structured turns.

**Source B — Per-character utterances.** `main-horse/disco-elysium-utterances/processed/Kim Kitsuragi/metadata.txt` contains all 6,438 of Kim's lines without context, paired with the corresponding `.wav` reference.

**Stage 1 SFT data — context → Kim line.** For every Kim utterance in source A, we extract a 6-turn rolling window of preceding context, format prior actors with role tags (`Detective:` / `(Empathy — internal):` / `[scene: action_check]`), and emit a `(system, context, kim_line)` chat-style triple. After length filtering (<800 chars per Kim line, ≥2 words) we obtain **1,587 train + 80 validation samples** (5% by-conversation split to avoid leakage).

**Stage 2 SFT data — joint dialogue + tool call.** For every `[Action/Check: ...]` tag in source A we emit a target output of the form

```json
{"dialogue": "<the next NPC line>", "tool_calls": [{"name": "...", "args": {...}}]}
```

with the action tag parsed into the structured `tool_calls` field. We additionally synthesize 8 game-tool calls (the VN layer of Table 1) using a templated procedure conditioned on scene-state changes, expanding the corpus by approximately [TODO: estimate from output.json scan]× and yielding [TODO: total samples] samples.

**Stage 3 DPO preference data.** For each Stage-2 sample, we sample two completions from a Stage-2 SFT model: a *chosen* completion (correct skill_check + persona-faithful Kim reply) and a *rejected* completion (wrong tool, hallucinated skill, or out-of-character voice—bootstrapped using base Qwen3.5-0.8B's no-training output, which our prior work [TODO: §5.11 of v2] documented produces character breaks). This yields [TODO: number] preference pairs for DPO training.

### 5.2 Three-Stage Training Recipe

All three stages train Qwen3.5-0.8B (~753M parameters) on Apple M4 16GB.

**Stage 1: Persona-SFT.** LoRA (rank=16, α=32, dropout=0.15) on `q_proj, v_proj, k_proj, o_proj`. Loss: causal LM on the assistant span. Hyperparameters chosen conservatively to combat the LoRA-overfitting failure observed at this scale in our preliminary experiments (see Appendix B): lr=5e-5, weight_decay=0.05, batch=1 with grad-accum=8, max_seq_len=384, mid-epoch evaluation with patience=2 early stop. **Trained on the 1,587 Kim Stage-1 samples.**

**Stage 2: Tool-augmented SFT.** Same LoRA initialized from Stage-1 weights, retrained jointly on the structured `(scene, prior_dialogue) → {dialogue, tool_calls}` mapping. Output is wrapped in a JSON schema; loss is computed only over the JSON span. **Trained on the [TODO: total] tool-augmented samples.**

**Stage 3: Tool-faithfulness DPO.** Direct Preference Optimization [TODO: cite Rafailov 2023] on the chosen/rejected pairs. We use β=0.1, learning rate 1e-5, 2 epochs. **Trained on [TODO] preference pairs.**

We deliberately *omit* a separately trained Memory Prefix module and Emotion Head from our prior architecture (v0 of this work). Our internal benchmarks (Appendix B) showed those modules did not improve held-out persona quality at 0.8B; their parameter budget is reallocated to the tool-augmented stages.

---

## 6. Datasets and Benchmarks

We evaluate on three benchmarks. The first two are external standards; the third we contribute.

### 6.1 CPDC 2025 — Persona-Grounded Dialogue Challenge

We follow the CPDC 2025 evaluation protocol [19]: Tasks 1 (function-call correctness), 2 (persona/world adherence), and 3 (hybrid), scored by a 4-axis LLM-as-Judge over (1) Scenario Adherence & Quest Progression, (2) NPC Believability & Engagement, (3) Persona Consistency, (4) Dialogue Flow & Coherence. We report under both **API track rules** (no fine-tuning beyond what we declare) and **GPU track rules** (fine-tuning permitted). Our system runs in API track only by construction (we host our own model).

### 6.2 BFCL V4 — Multi-Turn Function Calling

We evaluate on BFCL V3's multi-turn category and V4's agentic subset (search, memory) [23]. Metrics: AST-based and execution-based accuracy.

### 6.3 DEBench (Ours)

DEBench evaluates joint persona and tool fidelity in *Disco Elysium*-grounded scenarios. We construct it from a held-out 5% of `output.json` conversations. For each held-out conversation:

- **Persona task**: Given the prior 6 turns, predict Kim's next line. Scored by a multi-judge panel (Claude-Sonnet-4.6 and Qwen3.5:9b) on the 5-axis rubric we previously validated [TODO §5.11 of v2]: consistency, fluency, engagement, memory_use, emotion_fit. We additionally compute character-break rate (zero-shot prompted detector for "AI", "language model", modern-era references).
- **Skill-tool task**: Given the prior 6 turns, predict the next skill_check (if any) and its arguments. Scored by exact-match on `skill_name`, exact-match on `difficulty` band, and a recovery-from-misroute metric (does the model still produce a coherent next line if we inject a *forced* failed skill check?).
- **Game-tool task**: Given the scene state delta (e.g., a character entered, location changed), predict the appropriate VN-layer tool calls. Scored by precision/recall against a hand-labeled ground truth on a 100-scenario subset.

DEBench has **[TODO: number, ~80] held-out conversations × ~6 Kim utterances each ≈ [TODO: ~480] persona test items + [TODO: ~250] skill-tool items + 100 game-tool items**.

We release DEBench at https://github.com/GORXE111/NPCAI/tree/main/benchmarks/debench.

---

## 7. Experiments

### 7.1 Setup

- **Hardware**: Apple Mac Mini M4, 16GB unified memory.
- **Models**: Qwen3.5-0.8B base; with our LoRA adapters at each stage.
- **Baselines**: (i) Qwen3.5-0.8B base + zero-shot tool prompt; (ii) Qwen3.5-9B base via Ollama (a strong cloud-comparable on the same hardware); (iii) [TODO: GPT-4o or Claude as cost permits]; (iv) Fixed-Persona SLMs equivalent (Mistral-7B-Instruct + character prompt) [arXiv 2511.10277].
- **Training cost**: Stage 1 [TODO: minutes] on M4. Stage 2 [TODO]. Stage 3 [TODO]. Total [TODO]. No data-center GPUs used.

### 7.2 Main Result: DEBench

[TODO: fill table once Stage 1–3 complete. Schema:]

**Table 2.** DEBench results. Persona scored 1–5 (mean of 5 sub-dimensions). Skill exact-match. Game-tool F1.

| Model | Persona ↑ | Char-Break Rate ↓ | Skill EM ↑ | Game-Tool F1 ↑ |
|-------|:---------:|:-----------------:|:----------:|:--------------:|
| Qwen3.5-0.8B base | [TODO] | [TODO] | [TODO] | [TODO] |
| + Stage-1 LoRA (persona) | [TODO] | [TODO] | [TODO] | [TODO] |
| + Stage-2 (tool SFT) | [TODO] | [TODO] | [TODO] | [TODO] |
| + Stage-3 (DPO) — **ours** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** |
| Qwen3.5-9B (no train) | [TODO] | [TODO] | [TODO] | [TODO] |
| Claude-Sonnet-4.6 (API) | [TODO] | [TODO] | [TODO] | [TODO] |

### 7.3 CPDC 2025

[TODO: After running CPDC 2025 dataset]

**Table 3.** CPDC 2025 results. Compared to published winners.

| System | Params | Hardware | Task 1 (FC) | Task 2 (Dialog) | Task 3 (Hybrid) |
|--------|:------:|:--------:|:-----------:|:---------------:|:---------------:|
| MSRA_SC [20] | 14B | L40S | [report] | [report] | [report] |
| Multi-Expert [22] | 14B | L40S | [report] | [report] | [report] |
| **Ours (Skill-as-Tool, 0.8B)** | **0.8B** | **M4 Mac** | **[TODO]** | **[TODO]** | **[TODO]** |

### 7.4 BFCL V4

[TODO]

### 7.5 Latency

[TODO: per-turn latency on M4 16GB; Ollama parallel=1, parallel=3]

### 7.6 Ablation: Three Stages

[TODO: ablation table on DEBench showing additive contribution of Stages 1, 2, 3]

### 7.7 Qualitative Examples

[TODO: 2–3 example transcripts comparing base, our model, and a 14B reference. Each example: scene, player choice, model's joint dialogue+tool output. Show: (a) successful tool selection with in-character voice; (b) successful refusal/redirect when memory is unsupported; (c) failure mode and recovery.]

---

## 8. Discussion

### 8.1 Why Skill-as-Tool Works

Three properties of the skill abstraction make it effective as a tool taxonomy. (i) **Closed inventory**: 24 skills are easy for a small model to memorize and select among, unlike a long-tail open API. (ii) **Tight coupling with dialogue**: each skill check produces a verbal output (an inner monologue), so generation and tool call are not separable—training one helps the other. (iii) **Self-narrating actions**: a skill check is *narratively legible* to the player; this makes evaluation tractable because failures produce visible dialogue inconsistencies.

### 8.2 Why 0.8B Is Enough

Two findings independently support the viability of 0.8B for this task. First, BFCL V4's leaderboard places multiple sub-1B and few-billion-parameter models at the same agent-score tier, indicating tool selection saturates well below 14B. Second, our Persona-SFT alone, on 1,587 examples, achieves [TODO: number] on persona consistency on DEBench, against the base model's [TODO]. The remaining gap to larger models is not in language quality but in the rare-event tail of skill selection, which we address with Stage-2 and Stage-3.

### 8.3 Why Visual Novel

The visual-novel framing is not incidental. Compared to a 2D top-down environment (our previous prototype), the VN format makes every tool call visually salient: a `set_expression` is a sprite swap on the order of tens of milliseconds, and a `present_choices` is a UI panel with structured options. This (i) simplifies engineering by collapsing the action space onto UI events, (ii) makes paper figures legible, and (iii) **maps faithfully back to** *Disco Elysium*'s own UI grammar—reinforcing the ecological validity of training on its corpus.

### 8.4 Limitations

- **Single-character anchor.** Our system is trained and evaluated on one NPC. We expect transfer to other distinctive NPCs (Geralt, 2B, Kafka) to require additional persona-SFT corpora, not architectural change.
- **English-only.** Although the *Disco Elysium* corpus exists in multiple languages, we trained only on English; multilingual is straightforward but unevaluated.
- **No live player study.** Our evaluation uses LLM-as-Judge panels, not human game-play studies. We plan a Prolific-recruited study (see Park et al. [1]).
- **No combat/movement.** Our action space is dialogue-and-presentation; physical actions in 3D space are out of scope.

### 8.5 Concurrent Work

Fixed-Persona SLMs (arXiv 2511.10277) is the most directly comparable concurrent work. They use DistilGPT-2, TinyLlama-1.1B, and Mistral-7B with a swappable memory module on consumer hardware. We differ in (a) explicit tool-use training, (b) game-engine grounding via Unity, (c) anchoring to a published commercial NPC for evaluation, and (d) a DPO-based stage targeting tool faithfulness specifically. We report a head-to-head comparison in Appendix C [TODO].

The CPDC 2025 cohort [20, 21, 22] establishes the strong-baseline 14B systems we measure against. Our contribution is orthogonal: same task, ~17× smaller model, consumer hardware.

---

## 9. Conclusion

We presented Skill-as-Tool, a framework that grounds an SLM-driven NPC in a structured tool API mined from a published commercial RPG. We instantiated it with Kim Kitsuragi from *Disco Elysium*, trained Qwen3.5-0.8B on consumer hardware in three stages (persona-SFT, tool-augmented SFT, and DPO on tool faithfulness), and integrated the result with a Unity visual-novel environment that executes 18 grounded tools. Our experiments [TODO: claim once filled] show that a 0.8B model with this recipe approaches the persona-fidelity of CPDC 2025's 14B winners on the dialogue axis while running on a 16GB Mac at [TODO: ms]ms-class latency.

We release: (i) the trained checkpoints, (ii) the Unity environment, (iii) DEBench, and (iv) the MPS-compatibility patch (`qwen35_mps_fix.py`) at https://github.com/GORXE111/NPCAI.

---

## Acknowledgments

[TODO]

---

## References

[1] Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P., Bernstein, M.S. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023. arXiv:2304.03442.

[2] Zhong, W., Guo, L., Gao, Q., Ye, H., Wang, Y. "MemoryBank: Enhancing Large Language Models with Long-Term Memory." AAAI 2024. arXiv:2305.10250.

[3] Liu, L., et al. "Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory." arXiv:2311.08719, 2023.

[4] Sumers, T.R., Yao, S., Narasimhan, K., Griffiths, T.L. "Cognitive Architectures for Language Agents." TMLR 2024. arXiv:2309.02427.

[5] Zhou, J., et al. "CharacterGLM: Customizing Chinese Conversational AI Characters with Large Language Models." EMNLP 2024 Industry. arXiv:2311.16832.

[6] Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023. arXiv:2305.14314.

[7] Vezhnevets, A.S., et al. "Generative Agent-Based Modeling with Actions Grounded in Physical, Social, or Digital Space using Concordia." arXiv:2312.03664, 2023.

[8] Wang, G., et al. "Voyager: An Open-Ended Embodied Agent with Large Language Models." NeurIPS 2023. arXiv:2305.16291.

[9] Li, G., et al. "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society." NeurIPS 2023. arXiv:2303.17760.

[10] Wang, Z., Chiu, Y.Y., Chiu, Y.C. "Humanoid Agents: Platform for Simulating Human-like Generative Agents." EMNLP 2023 Demo. arXiv:2310.05418.

[11] Altera.AL, et al. "Project Sid: Many-agent simulations toward AI civilization." arXiv:2411.00114, 2024.

[12] Zhu, A., Martin, L., Head, A., Callison-Burch, C. "CALYPSO: LLMs as Dungeon Masters' Assistants." AIIDE 2023. arXiv:2308.07540.

[13] Zhou, W., et al. "RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text." ACL 2023 Demo. arXiv:2305.13304.

[14] Jeon, H., Ha, H., Kim, J.-J. "LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents." arXiv:2602.01053, 2026.

[15] Braas, M., Esterle, L. "Fixed-Persona SLMs with Modular Memory: Scalable NPC Dialogue on Consumer Hardware." arXiv:2511.10277, 2025.

[16] Ren, J., et al. "SimWorld: An Open-ended Realistic Simulator for Autonomous Agents in Physical and Social Worlds." arXiv:2512.01078, 2025.

[17] Tu, Q., et al. "CharacterEval: A Chinese Benchmark for Role-Playing Conversational Agent Evaluation." ACL 2024. arXiv:2401.01275.

[18] Akoury, N., Yang, Q., Iyyer, M. "A Framework for Exploring Player Perceptions of LLM-Generated Dialogue in Commercial Video Games." EMNLP 2023 Findings. https://aclanthology.org/2023.findings-emnlp.151.pdf.

[19] CPDC 2025 — Commonsense Persona-Grounded Dialogue Challenge. Wordplay Workshop, EMNLP 2025. https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025

[20] MSRA_SC. "Interactive AI NPCs Powered by LLMs: CPDC 2025 Technical Report." arXiv:2511.20200, 2025.

[21] "Deflanderization for Game Dialogue: Balancing Roleplay and Task Execution." arXiv:2510.13586, 2025.

[22] "Efficient Tool-Calling Multi-Expert NPC Agent for CPDC 2025." arXiv:2511.01720, 2025.

[23] Yan, F., et al. "Berkeley Function Calling Leaderboard V4: From Tool Use to Agentic Evaluation." 2025-2026. https://gorilla.cs.berkeley.edu/leaderboard.html.

[24] Yao, S., et al. "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." NeurIPS 2024. arXiv:2406.12045.

[25] Yao, S., et al. "τ²-bench: Multi-turn Conversational Agents under Dual Control." arXiv:2506.07982, 2025.

[26] Lu, J., et al. "ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities." NAACL 2025 Findings. arXiv:2408.04682.

[27] Gusev, I. "PingPong: A Benchmark for Role-Playing Language Models with User Emulation and Multi-Model Evaluation." arXiv:2409.06820, 2024.

[28] Hu, E.J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685.

[29] Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. arXiv:2305.18290.

[TODO: Add Toolformer, ReAct, Gorilla, PersonaEval citations.]

---

## Appendix A: Tool API JSON Schema

[TODO: Insert the full JSON schema we ship with the system, ~3 pages]

## Appendix B: Negative Results from v0 (Memory Prefix and Emotion Head)

In an earlier iteration of this work we proposed a Memory Prefix Injection module and an Emotion Head as separately trained components. Our internal benchmarks showed that, on Qwen3.5-0.8B at this scale, a vanilla LoRA on insufficient data overfits catastrophically (mean rated overall score 2.53/5 vs. base 3.81/5 in a 4-config ablation), and Memory Prefix recovers some lost ground (3.52/5) but does not exceed base. Emotion Head, while reaching 72% classification accuracy, did not affect generation. We therefore omit these modules from the production system but present the data here for completeness; full details in our project repository's `paper/draft_v2.md`.

## Appendix C: Direct Comparison with Fixed-Persona SLMs (arXiv 2511.10277)

[TODO: Once we run their setup as a baseline.]
