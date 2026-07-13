"""
Stage 3 v3.1.D7 — DPO refine on top of v5.0 (the comprehensive all-tool SFT).

Why D7 can finally break through where D2-D5 couldn't:
- v5.0 made emotion NON-ZERO (0.20) in the base distribution. DPO's log(pi/pi_ref)
  can now reweight it UP (the zero-prob deadlock that killed D2-D5 is gone).
- v5.0 is a COMPLETE balanced base (all tools present), unlike D3/D4's pure-positive
  warm-start that collapsed to "always call". So DPO can refine, not rescue.

Target v5.0's profile:
  STRONG (preserve): evidence 1.0/1.0, social 0.8/0.8, filler 1.0/1.0, emotion 0.20
  WEAK (fix):        scene 0, combined 0.17, branch 0, Suppress 0.10 (over-calls)

Pairs (205):
   70 suppress     chosen=empty, rejected=wrong tool      -> fix Suppress 0.10
   40 scene-recall chosen=show_character, rejected=empty  -> recover scene
   40 emotion      chosen=set_expression, rejected=wrong  -> boost emotion (ref nonzero!)
   30 branch       chosen=present_choices, rejected=show  -> try branch
   25 evidence-anc chosen=skill_check, rejected=empty     -> preserve evidence/social

ref = init = v5.0. beta=0.1.
"""
import json, os, random
from collections import Counter

V50_TRAIN = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v5_0/kim_tool_train.jsonl"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_dpo_d7"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(48)

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def classify(s):
    c = s["messages"][2]["content"]
    if '"skill_check"' in c: return "skill_check"
    if '"show_character"' in c: return "show_character"
    if '"present_choices"' in c: return "present_choices"
    if '"set_expression"' in c: return "set_expression"
    return "empty"

v50 = load_jsonl(V50_TRAIN)
print(f"v5.0 samples: {len(v50)}")
buckets = {"skill_check": [], "show_character": [], "present_choices": [], "set_expression": [], "empty": []}
for s in v50:
    t = classify(s)
    if t in buckets: buckets[t].append(s)
print("v5.0 by tool:", {k: len(v) for k, v in buckets.items()})
for k in buckets: random.shuffle(buckets[k])

SYSTEM = v50[0]["messages"][0]["content"]

def mk(prompt, chosen, rejected, ptype):
    return {"system": SYSTEM, "prompt": prompt, "chosen": chosen, "rejected": rejected, "type": ptype}

def empty_resp(chosen_str):
    cd = json.loads(chosen_str)
    return json.dumps({"dialogue": cd.get("dialogue", "..."), "tool_calls": []}, ensure_ascii=False)

def wrong_tool_resp(chosen_str, wrong):
    cd = json.loads(chosen_str)
    return json.dumps({"dialogue": cd.get("dialogue", "..."), "tool_calls": [wrong]}, ensure_ascii=False)

pairs = []
WRONG_POOL = [
    {"name": "skill_check", "args": {"skill": "Logic", "message": "Something unspecified."}},
    {"name": "show_character", "args": {"actor": "Stranger", "slot": "right"}},
    {"name": "present_choices", "args": {"options": ["Option A.", "Option B."]}},
    {"name": "play_bgm", "args": {"track": "ambient_bg"}},
    {"name": "narrate", "args": {"text": "The scene continues without incident."}},
]

# 70 suppress (chosen=empty, rejected=wrong tool)
for s in buckets["empty"][:70]:
    m = s["messages"]
    try:
        pairs.append(mk(m[1]["content"], m[2]["content"], wrong_tool_resp(m[2]["content"], random.choice(WRONG_POOL)), "suppress"))
    except: continue

# 40 scene-recall (chosen=show_character, rejected=empty)
for s in buckets["show_character"][:40]:
    m = s["messages"]
    try:
        pairs.append(mk(m[1]["content"], m[2]["content"], empty_resp(m[2]["content"]), "scene_recall"))
    except: continue

# 40 emotion boost (chosen=set_expression, rejected=wrong tool: skill_check/show alternating)
for i, s in enumerate(buckets["set_expression"][:40]):
    m = s["messages"]
    wrong = {"name": "skill_check", "args": {"skill": "Empathy", "message": "Unspecified."}} if i % 2 == 0 \
            else {"name": "show_character", "args": {"actor": "Detective", "slot": "left"}}
    try:
        pairs.append(mk(m[1]["content"], m[2]["content"], wrong_tool_resp(m[2]["content"], wrong), "emotion"))
    except: continue

# 30 branch (chosen=present_choices, rejected=show_character)
for s in buckets["present_choices"][:30]:
    m = s["messages"]
    try:
        pairs.append(mk(m[1]["content"], m[2]["content"],
                        wrong_tool_resp(m[2]["content"], {"name": "show_character", "args": {"actor": "Detective", "slot": "right"}}),
                        "branch"))
    except: continue

# 25 evidence anchor (chosen=skill_check, rejected=empty)
for s in buckets["skill_check"][:25]:
    m = s["messages"]
    try:
        pairs.append(mk(m[1]["content"], m[2]["content"], empty_resp(m[2]["content"]), "evidence_anchor"))
    except: continue

print(f"\nTotal: {len(pairs)}")
for k, v in Counter(p["type"] for p in pairs).most_common(): print(f"  {k:18s} {v}")
ct = sum(1 for p in pairs if '"tool_calls": []' not in p["chosen"])
rt = sum(1 for p in pairs if '"tool_calls": []' not in p["rejected"])
print(f"chosen:   with_tool={ct} empty={len(pairs)-ct}")
print(f"rejected: with_tool={rt} empty={len(pairs)-rt}")

random.shuffle(pairs)
n_val = max(10, len(pairs)//10)
val, train = pairs[:n_val], pairs[n_val:]
def strip(p): return {k: v for k, v in p.items() if k != "type"}
with open(os.path.join(OUTPUT_DIR, "dpo_train.jsonl"), "w", encoding="utf-8") as f:
    for p in train: f.write(json.dumps(strip(p), ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "dpo_valid.jsonl"), "w", encoding="utf-8") as f:
    for p in val: f.write(json.dumps(strip(p), ensure_ascii=False) + "\n")
print(f"\nFinal: Train {len(train)}  Val {len(val)}")
print(f"Output: {OUTPUT_DIR}")
