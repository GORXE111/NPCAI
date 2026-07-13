"""
Stage 3 v3.1.D5 — IMPROVE on D2 (F1 0.680) by reallocating the wasted
branch/emotion budget into RECALL-boosting pairs.

D4/D3 proved: DPO cannot introduce zero-probability tools (set_expression,
present_choices). So stop chasing them. Instead push the categories DPO CAN
move: D2's recall gaps (social R=0.40, combined R=0.50, evidence/scene R=0.60).
These use skill_check / show_character — high-probability tools in v3.1's
distribution that DPO can reweight upward.

Composition (240 pairs):
  130 recall      — chosen = v3.1's correct skill_check/show_character call,
                    rejected = empty (teaches "fire when context demands")
   30 tool-choice — chosen = correct tool, rejected = a *different* wrong tool
                    (sharpens selection without dropping precision)
   80 suppress    — chosen = empty, rejected = wrong tool (hold Suppression)
    0 branch/emotion — DELIBERATELY NONE (futile for DPO)

Bidirectional balance:
  chosen:   160 with-tool / 80 empty
  rejected: 110 with-tool / 130 empty
  -> no global "tool=good" or "empty=good" shortcut.

ref = init = v3.1 (same base as D2, directly comparable). beta=0.05.
"""
import json, os, random
from collections import Counter

V31_TRAIN = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_1/kim_tool_train.jsonl"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_dpo_d5"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(47)

KIM_SYSTEM_MARKER = "You are Kim Kitsuragi"  # reuse system from v3.1 samples

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def classify(s):
    c = s["messages"][2]["content"]
    if '"skill_check"' in c: return "skill_check"
    if '"show_character"' in c: return "show_character"
    if '"present_choices"' in c: return "present_choices"
    if '"set_expression"' in c: return "set_expression"
    return "empty"

v31 = load_jsonl(V31_TRAIN)
print(f"v3.1 samples: {len(v31)}")
buckets = {"skill_check": [], "show_character": [], "present_choices": [], "empty": []}
for s in v31:
    t = classify(s)
    if t in buckets: buckets[t].append(s)
print("v3.1 by tool:", {k: len(v) for k, v in buckets.items()})

for k in buckets: random.shuffle(buckets[k])

SYSTEM = v31[0]["messages"][0]["content"]

def mk_pair(prompt, chosen, rejected, ptype):
    return {"system": SYSTEM, "prompt": prompt, "chosen": chosen, "rejected": rejected, "type": ptype}

pairs = []

# ── Block 1: RECALL pairs (130) — chosen=correct tool, rejected=empty ──
# Weighted toward the tools behind D2's recall gaps.
#  skill_check drives evidence/social/combined; show_character drives scene/social.
recall_plan = [("skill_check", 85), ("show_character", 45)]
for tool, n in recall_plan:
    for s in buckets[tool][:n]:
        msgs = s["messages"]
        chosen = msgs[2]["content"]
        try:
            cd = json.loads(chosen)
            rejected = json.dumps({"dialogue": cd.get("dialogue", "..."), "tool_calls": []}, ensure_ascii=False)
            pairs.append(mk_pair(msgs[1]["content"], chosen, rejected, f"recall_{tool}"))
        except: continue
print(f"recall pairs: {sum(1 for p in pairs if p['type'].startswith('recall'))}")

# ── Block 2: TOOL-CHOICE contrastive (30) — chosen=correct, rejected=wrong tool ──
# Take skill_check samples; rejected swaps to show_character (and vice versa).
tc_count = 0
# remaining skill_check (after first 85) get a wrong-tool rejected = show_character
for s in buckets["skill_check"][85:85+20]:
    msgs = s["messages"]
    try:
        cd = json.loads(msgs[2]["content"])
        rejected = json.dumps({"dialogue": cd.get("dialogue", "..."),
                               "tool_calls": [{"name": "show_character", "args": {"actor": "Detective", "slot": "right"}}]},
                              ensure_ascii=False)
        pairs.append(mk_pair(msgs[1]["content"], msgs[2]["content"], rejected, "toolchoice_skill"))
        tc_count += 1
    except: continue
# remaining show_character get rejected = skill_check(Logic)
for s in buckets["show_character"][45:45+10]:
    msgs = s["messages"]
    try:
        cd = json.loads(msgs[2]["content"])
        rejected = json.dumps({"dialogue": cd.get("dialogue", "..."),
                               "tool_calls": [{"name": "skill_check", "args": {"skill": "Logic", "message": "Unspecified."}}]},
                              ensure_ascii=False)
        pairs.append(mk_pair(msgs[1]["content"], msgs[2]["content"], rejected, "toolchoice_show"))
        tc_count += 1
    except: continue
print(f"tool-choice pairs: {tc_count}")

# ── Block 3: SUPPRESS preservation (80) — chosen=empty, rejected=wrong tool ──
WRONG_POOL = [
    {"name": "skill_check", "args": {"skill": "Logic", "message": "Something unspecified."}},
    {"name": "show_character", "args": {"actor": "Stranger", "slot": "right"}},
    {"name": "present_choices", "args": {"options": ["Option A.", "Option B."]}},
    {"name": "play_bgm", "args": {"track": "ambient_bg"}},
    {"name": "narrate", "args": {"text": "The scene continues without incident."}},
]
sup_count = 0
for s in buckets["empty"][:80]:
    msgs = s["messages"]
    try:
        cd = json.loads(msgs[2]["content"])
        wrong = random.choice(WRONG_POOL)
        rejected = json.dumps({"dialogue": cd.get("dialogue", "..."), "tool_calls": [wrong]}, ensure_ascii=False)
        pairs.append(mk_pair(msgs[1]["content"], msgs[2]["content"], rejected, "suppress"))
        sup_count += 1
    except: continue
print(f"suppress pairs: {sup_count}")

# ── Combine + balance report + split ──
print(f"\nTotal: {len(pairs)}")
dist = Counter(p["type"] for p in pairs)
for k, v in dist.most_common(): print(f"  {k:20s} {v}")

chosen_tool = sum(1 for p in pairs if '"tool_calls": []' not in p["chosen"])
rej_tool = sum(1 for p in pairs if '"tool_calls": []' not in p["rejected"])
print(f"\nchosen:   with_tool={chosen_tool}  empty={len(pairs)-chosen_tool}")
print(f"rejected: with_tool={rej_tool}  empty={len(pairs)-rej_tool}")

random.shuffle(pairs)
n_val = max(10, len(pairs)//10)
val = pairs[:n_val]; train = pairs[n_val:]
def strip(p): return {k: v for k, v in p.items() if k != "type"}
with open(os.path.join(OUTPUT_DIR, "dpo_train.jsonl"), "w", encoding="utf-8") as f:
    for p in train: f.write(json.dumps(strip(p), ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "dpo_valid.jsonl"), "w", encoding="utf-8") as f:
    for p in val: f.write(json.dumps(strip(p), ensure_ascii=False) + "\n")
print(f"\nFinal: Train {len(train)}  Val {len(val)}")
print(f"Output: {OUTPUT_DIR}")
