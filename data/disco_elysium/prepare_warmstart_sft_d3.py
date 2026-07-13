"""
Stage 3a (D3 warm-start SFT data) — extract POSITIVE tool-call targets from the
D2 preference pairs to lift set_expression / present_choices probability from ~0
to non-zero, so the subsequent DPO (ref=warm-start) can finally reweight them.

Why this is safe (vs the v3.1.1 SFT-continuation collapse):
- v3.1.1 collapsed because of 100 suppression NEGATIVES that dominated the loss.
- Here we use ONLY positive new-tool targets + a few anchors. NO suppress negatives.
- Goal is not "learn well" — just "make the new tools non-zero probability."
  DPO does the real alignment afterward.

Composition (170 SFT samples, all chosen-side of D2 pairs):
   70 branch   (present_choices) — chosen from D2 branch pairs
   60 emotion  (set_expression)  — chosen from D2 emotion pairs
   40 anchor   (skill_check/show_character) — prevent total drift
   0  suppress — DELIBERATELY NONE
"""
import json, os, random

D2_TRAIN = "D:/AIproject/NPCAI/data/disco_elysium/training_dpo_v3_1_D2/dpo_train.jsonl"
D2_VALID = "D:/AIproject/NPCAI/data/disco_elysium/training_dpo_v3_1_D2/dpo_valid.jsonl"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_warmstart_d3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(45)

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def pair_to_sft(pair):
    """Convert a D2 preference pair -> SFT sample using the CHOSEN response."""
    return {
        "messages": [
            {"role": "system", "content": pair["system"]},
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["chosen"]},
        ]
    }

def tool_of(chosen_str):
    for t in ["present_choices", "set_expression", "skill_check", "show_character"]:
        if f'"{t}"' in chosen_str: return t
    return "empty"

all_pairs = load_jsonl(D2_TRAIN) + load_jsonl(D2_VALID)
print(f"D2 pairs loaded: {len(all_pairs)}")

# Bucket by the chosen tool
buckets = {"present_choices": [], "set_expression": [], "skill_check": [], "show_character": [], "empty": []}
for p in all_pairs:
    buckets[tool_of(p["chosen"])].append(p)

for k, v in buckets.items():
    print(f"  {k:20s} {len(v)}")

# Build warm-start SFT set: ALL new-tool positives + capped anchors, NO empty
random.shuffle(buckets["present_choices"])
random.shuffle(buckets["set_expression"])
random.shuffle(buckets["skill_check"])
random.shuffle(buckets["show_character"])

sft = []
sft += [pair_to_sft(p) for p in buckets["present_choices"][:70]]   # branch
sft += [pair_to_sft(p) for p in buckets["set_expression"][:60]]    # emotion
sft += [pair_to_sft(p) for p in buckets["skill_check"][:25]]       # anchor
sft += [pair_to_sft(p) for p in buckets["show_character"][:15]]    # anchor
# NO empty/suppress samples — this is the key difference from v3.1.1

random.shuffle(sft)
print(f"\nWarm-start SFT total: {len(sft)} (NO suppress negatives)")

# Dedup by assistant content
seen = set(); dedup = []
for s in sft:
    key = s["messages"][1]["content"][:80] + s["messages"][2]["content"][:60]
    if key in seen: continue
    seen.add(key); dedup.append(s)
print(f"After dedup: {len(dedup)}")

# Tool distribution sanity
from collections import Counter
dist = Counter(tool_of(s["messages"][2]["content"]) for s in dedup)
print(f"Tool distribution: {dict(dist)}")

# Split 90/10
random.shuffle(dedup)
n_val = max(8, len(dedup) // 10)
val = dedup[:n_val]
train = dedup[n_val:]

with open(os.path.join(OUTPUT_DIR, "kim_tool_train.jsonl"), "w", encoding="utf-8") as f:
    for s in train: f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "kim_tool_valid.jsonl"), "w", encoding="utf-8") as f:
    for s in val: f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nFinal: Train {len(train)}  Val {len(val)}")
print(f"Output: {OUTPUT_DIR}")
