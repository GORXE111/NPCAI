"""
DPO data v2: Add 'missing tool' preference pairs to counter distributional collapse.

Discovery: v1 synthetic data was biased toward 'remove tool' perturbations
(F4 OOC + F6 extra + F7 reorder = 64% of pairs). DPO learned to be over-conservative
and dropped tool calls.

Fix: Add F0_missing_tool — chosen has tool, rejected has empty list.
This teaches "when context demands a tool, you MUST call it".
"""
import json, os, random, copy

INPUT = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2/kim_tool_train.jsonl"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_dpo"
random.seed(42)

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

samples = load_jsonl(INPUT)
print(f"Loaded {len(samples)} Stage 2 samples")

# Filter samples where ground truth HAS at least one tool_call
with_tools = []
for s in samples:
    target = json.loads(s["messages"][2]["content"])
    if target.get("tool_calls"):
        with_tools.append(s)

print(f"  {len(with_tools)} samples have ≥1 tool_call (will be augmented)")

# === F0_missing_tool: chosen=full target, rejected=empty tool_calls ===
new_pairs = []
for s in with_tools:
    chosen_target = json.loads(s["messages"][2]["content"])
    rejected_target = copy.deepcopy(chosen_target)
    rejected_target["tool_calls"] = []
    new_pairs.append({
        "system": s["messages"][0]["content"],
        "prompt": s["messages"][1]["content"],
        "chosen": json.dumps(chosen_target, ensure_ascii=False),
        "rejected": json.dumps(rejected_target, ensure_ascii=False),
        "perturbation": "F0_missing_tool",
    })

# === F0b: keep some tools, drop others (partial missing) ===
multi_tool_samples = [s for s in with_tools if len(json.loads(s["messages"][2]["content"])["tool_calls"]) >= 2]
print(f"  {len(multi_tool_samples)} samples have ≥2 tool_calls (partial-drop pairs)")
for s in multi_tool_samples:
    chosen_target = json.loads(s["messages"][2]["content"])
    rejected_target = copy.deepcopy(chosen_target)
    # Drop a random tool_call (not all)
    drop_idx = random.randint(0, len(rejected_target["tool_calls"]) - 1)
    del rejected_target["tool_calls"][drop_idx]
    new_pairs.append({
        "system": s["messages"][0]["content"],
        "prompt": s["messages"][1]["content"],
        "chosen": json.dumps(chosen_target, ensure_ascii=False),
        "rejected": json.dumps(rejected_target, ensure_ascii=False),
        "perturbation": "F0b_partial_drop",
    })

print(f"\nGenerated {len(new_pairs)} balancing pairs")
print(f"  F0_missing_tool: {sum(1 for p in new_pairs if p['perturbation']=='F0_missing_tool')}")
print(f"  F0b_partial_drop: {sum(1 for p in new_pairs if p['perturbation']=='F0b_partial_drop')}")

# Merge with existing synthetic data
existing_train = load_jsonl(os.path.join(OUTPUT_DIR, "synthetic_train.jsonl"))
existing_val = load_jsonl(os.path.join(OUTPUT_DIR, "synthetic_valid.jsonl"))

all_pairs = list(existing_train) + list(existing_val) + new_pairs
random.shuffle(all_pairs)

split = int(len(all_pairs) * 0.95)
train = all_pairs[:split]
val = all_pairs[split:]

with open(os.path.join(OUTPUT_DIR, "balanced_train.jsonl"), "w", encoding="utf-8") as f:
    for s in train: f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "balanced_valid.jsonl"), "w", encoding="utf-8") as f:
    for s in val: f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nFinal train: {len(train)} | val: {len(val)}")

# Distribution check
from collections import Counter
dist = Counter(p["perturbation"] for p in all_pairs)
print("\nFinal perturbation distribution:")
for k, v in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {k:25s} {v:>4} ({v/len(all_pairs):.1%})")
