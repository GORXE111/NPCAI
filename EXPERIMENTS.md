# Experiment Map — Skill-as-Tool

A navigable index of every training run in this repository. Each row links a
**system** to its data-prep script, training script, dataset directory, and
DEBench result. All 2B results use Qwen3.5-2B + Stage-1 persona LoRA; all scripts
are under [`data/disco_elysium/`](data/disco_elysium/).

Metric legend: **F1** = Tool-selection F1 · **Sup** = tool-suppression (empty
rate on chit-chat; higher = better) · **emo** = emotion (`set_expression`)
per-category F1. Full per-category numbers and the narrative are in
[`paper/draft_v3.md` §7.2 / §7.6 / §7.7](paper/draft_v3.md).

---

## Headline: a Pareto frontier, not a single winner

The two most capable 2B systems are **Pareto-incomparable**:

| System | Path | **F1** | Sup | emotion | evidence | What it's best at |
|--------|------|:------:|:---:|:-------:|:--------:|-------------------|
| **DPO v3.1.D** | v3.1 → KL-DPO | **0.680** | 0.30 | 0 | 0.75/0.60 | **max F1**, all learned categories held |
| **v5.0** | all-tool SFT + synthetic | 0.641 | 0.10 | **0.20** | **1.00/1.00** | **max coverage**, only non-zero emotion |

Every attempt to *merge* their strengths degraded rather than combined (§7.7).

---

## Phase A — 0.8B era (capacity floor)

Seven configs, none exceeded F1 = 0.11. Establishes the parameter-count
threshold. (Scripts: `prepare_stage2_v3_1_skillmining.py`, `generate_dpo_*.py`,
`train_dpo_v3_1_D.py` predecessors; checkpoints under `checkpoints/kim_q35_08b_*`.)

| System | Method | F1 | Note |
|--------|--------|:--:|------|
| base / stage1 / stage2 | — | 0.000 | no tool calls or collapsed |
| Stage 2 v3 (intent 70/30) | SFT | **0.105** | best 0.8B |
| Stage 3 v1/v2/v3 | DPO | 0.000–0.066 | directional collapse (paper §7.7.4) |

**Finding:** at 0.8B, no SFT/DPO recipe crosses F1 = 0.11 → pivot to 2B.

---

## Phase B — 2B Stage 2 (SFT base policies)

| System | data-prep | dataset | F1 | Sup | Finding |
|--------|-----------|---------|:--:|:---:|---------|
| **v3.1** (base) | `prepare_stage2_v3_1_skillmining.py` | `training_stage2_v3_1/` | **0.639** | 0.30 | 6× over 0.8B — the scale result |
| v3.2 | `prepare_stage2_v3_2.py` | `training_stage2_v3_2/` | 0.000 | 1.00 | dup negatives → collapse |
| v3.3 | `prepare_stage2_v3_3.py` | `training_stage2_v3_3/` | 0.102 | 0.90 | synthetic too short (6.5 w) |
| v3.4 | `prepare_stage2_v3_4.py` | `training_stage2_v3_4/` | 0.578 | 0.73 | length-matched, still worse |
| v4.0 | `prepare_stage2_v4_0.py` | `training_stage2_v4_0/` | 0.653 | 0.13 | 2× data, but over-calls |

**Finding:** v3.1 sits at a near-saturated local optimum; adjacent recipes are
worse on ≥1 axis (paper §7.6).

---

## Phase C — 2B Stage 3 refinement study (the 20-config map)

### C1. In-place refinement: SFT-continuation vs KL-DPO

| System | data-prep | train | dataset | F1 | Sup | Finding |
|--------|-----------|-------|---------|:--:|:---:|---------|
| SFT-cont. v3.1.1 | `prepare_stage3_v3_1_1.py` | `train_kim_2b_s3_v3_1_1.py` | `training_stage3_v3_1_1/` | 0.073 | 0.97 | continued SFT → over-suppression collapse |
| SFT-cont. v3.1.2 | `prepare_stage3_v3_1_2.py` | `train_kim_2b_s3_v3_1_2.py` | `training_stage3_v3_1_2/` | 0.107 | 0.17 | rebalanced → other-way collapse |
| **DPO v3.1.D** | `prepare_dpo_v3_1_D.py` / `_D2.py` | `train_dpo_offline_v3_1_D2.py` | `training_dpo_v3_1_D2/` | **0.680** | 0.30 | **KL-DPO refines & holds everything** ⭐ |
| DPO D5 (recall) | `prepare_dpo_d5.py` | `train_dpo_offline_d5.py` | `training_dpo_d5/` | 0.673 | 0.27 | recall unmovable (generalization gap) |

**Finding:** KL-anchored DPO dominates continued SFT for refinement; the KL term
is the anti-forgetting mechanism SFT lacks. But DPO **cannot lift a ≈0-reference-
probability tool** (branch/emotion stay 0) — a property of `log(π/π_ref)`.

### C2. Warm-start SFT → DPO (introduce emotion, then align)

| System | data-prep | train | F1 | Sup | emo | Finding |
|--------|-----------|-------|:--:|:---:|:---:|---------|
| warm-start SFT | `prepare_warmstart_sft_d3.py` | `train_warmstart_sft_d3.py` | 0.096 | 0.0 | **0.83** | pure-positive SFT injects emotion but "always calls" |
| DPO D3 | `train_dpo_offline_d3.py` (ref=warm) | `precompute_ref_d3.py` | 0.122 | 0.0 | 0.80 | DPO can't fix a defective reference |
| DPO D4 | `prepare_dpo_d4.py` / `train_dpo_offline_d4.py` | | 0.094 | 0.0 | 0.80 | heavier suppress pairs still can't escape ref |
| DPO D7 | `prepare_dpo_d7.py` / `train_dpo_offline_d7.py` (ref=v5.0) | | 0.621 | 0.10 | 0.20 | DPO anchored to over-caller stays over-calling |

**Finding (2nd form):** DPO **cannot repair a defect that lives in its own
reference model** — the KL anchors the policy *to* the flawed prior.

### C3. Comprehensive all-tool SFT + synthetic emotion (the breakthrough & the chaos)

| System | data-prep | dataset | empty% | F1 | Sup | emo | Finding |
|--------|-----------|---------|:------:|:--:|:---:|:---:|---------|
| **v5.0** | `prepare_stage2_v5_0.py` | `training_stage2_v5_0/` | 44% | 0.641 | 0.10 | **0.20** | length-matched synthetic **works** (emo 0→0.20, evidence→1.0) ⭐ |
| v5.1 | `prepare_stage2_v5_1.py` | `training_stage2_v5_1/` | 54% | 0.090 | 0.40 | 0.25 | empty-majority → under-calls |
| v5.2 | `prepare_stage2_v5_2.py` | `training_stage2_v5_2/` | 47% | 0.000 | 1.00 | 0 | midpoint → **total collapse** (non-monotonic!) |
| v5.3 | `prepare_stage2_v5_3.py` | `training_stage2_v5_3/` | 44% | _pending_ | | | single-var: skill_check 30%→21.6% (test dominance hypothesis) |

**Finding:** (1) length-/register-matched synthetic data **does** give positive
feedback in a from-scratch, real-data-dominated SFT — correcting the v3.3 failure.
(2) Proactive-tool-calling SFT is **chaotic in its data balance**: three empty
ratios (44/47/54%) gave over-call / total-collapse / under-call *non-monotonically*
— good scores are fortunate landings, not tunable optima.

---

## Infrastructure notes

- **Offline DPO** (`precompute_ref_*.py` → `train_dpo_offline_*.py`): two 2B
  float32 models OOM on 16 GB. Phase 1 loads only the reference and caches
  `logP(chosen/rejected)` to disk; Phase 2 trains the policy alone (~10 GB peak).
- **MPS dtype patch**: `qwen35_mps_fix.py` (repo `model/`) — required for any
  Qwen3.5 train/infer on Apple Silicon.
- **Reproducibility**: greedy decoding → DEBench is bit-identical across reruns.
- **Val Acc/Loss are traps**: 0.8B DPO hit 98.6% then collapsed; v5.x hit
  0.89–0.95 with weak/collapsed generation. Only DEBench is ground truth.
