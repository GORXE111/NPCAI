"""
Stage 2 v3.3 — keep v3.1 baseline intact, add ONLY branch + emotion positives.

v3.2 mistake: added 115 smalltalk negatives (incl 80 duplicates) → over-suppression collapse.
v3.3 fix: keep v3.1 exactly as-is (677 samples), just append 54 branch + 40 emotion.
No smalltalk additions, no duplicates.

Expected: maintain v3.1's F1 0.639 + non-zero branch/emotion F1.
"""
import json, os, random

PRIOR_V3_1 = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_1/kim_tool_train.jsonl"
PRIOR_V3_1_VAL = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_1/kim_tool_valid.jsonl"
V3_2_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_2"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# Load v3.1 baseline (intact)
v3_1_train = load_jsonl(PRIOR_V3_1)
v3_1_val = load_jsonl(PRIOR_V3_1_VAL)
print(f"v3.1 baseline: {len(v3_1_train)} train + {len(v3_1_val)} val")

# Load v3.2 synthesized samples — extract ONLY branch + emotion (skip smalltalk negatives)
v3_2_train = load_jsonl(os.path.join(V3_2_DIR, "kim_tool_train.jsonl"))
v3_2_val = load_jsonl(os.path.join(V3_2_DIR, "kim_tool_valid.jsonl"))

branch_emotion = []
for s in v3_2_train + v3_2_val:
    tc = json.loads(s["messages"][2]["content"]).get("tool_calls", [])
    if tc and tc[0]["name"] in ("present_choices", "set_expression"):
        branch_emotion.append(s)

# Dedup by target hash
seen = set()
unique_be = []
for s in branch_emotion:
    h = hash(s["messages"][2]["content"])
    if h in seen: continue
    seen.add(h)
    unique_be.append(s)

print(f"\nUnique branch+emotion from v3.2: {len(unique_be)}")
from collections import Counter
cnt = Counter(json.loads(s["messages"][2]["content"])["tool_calls"][0]["name"] for s in unique_be)
print(f"  Distribution: {dict(cnt)}")

# Combine: v3.1 train + unique branch+emotion → new train
# v3.1 val stays as val (no need to split branch+emotion)
combined_train = list(v3_1_train) + unique_be
random.shuffle(combined_train)

# Save
with open(os.path.join(OUTPUT_DIR, "kim_tool_train.jsonl"), "w", encoding="utf-8") as f:
    for s in combined_train: f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "kim_tool_valid.jsonl"), "w", encoding="utf-8") as f:
    for s in v3_1_val: f.write(json.dumps(s, ensure_ascii=False) + "\n")

# Stats
tool_counts = Counter()
for s in combined_train:
    tc = json.loads(s["messages"][2]["content"]).get("tool_calls", [])
    if not tc: tool_counts["[empty]"] += 1
    for t in tc: tool_counts[t["name"]] += 1
print(f"\nFinal v3.3 train: {len(combined_train)} | val: {len(v3_1_val)}")
print("Tool distribution:")
for k, v in tool_counts.most_common():
    print(f"  {k:20s} {v} ({v/len(combined_train):.1%})")
