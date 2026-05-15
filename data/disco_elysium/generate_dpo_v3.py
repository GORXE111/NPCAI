"""
DPO v3 — preference pairs designed for Stage 2 v3.1 (intent-driven).

Lessons from previous DPO failures:
- v1 (synthetic only, no F0): collapsed to "always empty"
- v2 (added F0_missing_tool): F1 0.066, still mostly under-calls

For Stage 2 v3.1 (which over-calls — 27% suppress rate vs ideal 100%), we need:
- F_overcall: chosen=empty, rejected=irrelevant tool fired
- F_skill_swap: chosen=correct skill, rejected=wrong skill (skill-specific)
- F_persona_drift: chosen=Kim voice, rejected=AI-assistant voice (preserved from v2)
- F_argument: chosen=correct args, rejected=wrong args

This gives balanced direction signal:
- Some pairs teach "when to call" (F0, F0b inherited)
- Some pairs teach "when NOT to call" (F_overcall NEW, critical for v3.1)
- Some pairs teach "which tool" (F_skill_swap, F1 inherited)
"""
import json, os, random, copy, re
from collections import Counter

INPUT = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_1/kim_tool_train.jsonl"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_dpo_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

ALL_SKILLS = ["Logic","Encyclopedia","Rhetoric","Drama","Conceptualization","Visual Calculus",
              "Volition","Inland Empire","Empathy","Authority","Esprit de Corps","Suggestion",
              "Endurance","Pain Threshold","Physical Instrument","Electrochemistry","Shivers",
              "Half Light","Hand/Eye Coordination","Perception","Reaction Speed","Savoir Faire",
              "Interfacing","Composure"]
EMOTIONS = ["neutral","amused","worried","stern","sad","surprised","disapproving"]
FAKE_ACTORS = ["Garcon","Mr. Vasquez","Lieutenant Colmar","Cuno's Dad","Reuben","Pegrand"]
PERSONA_BREAKS = [
    "I'm just a language model, I can't really be Kim.",
    "OMG that's so cool dude!",
    "As an AI assistant, I would recommend...",
    "Actually I'm from Calvert Junction, not Revachol.",
    "*adjusts glasses* lmao based take detective",
]


def perturb_swap_tool(target):
    """F1: swap skill_check skill to wrong one."""
    t = copy.deepcopy(target)
    skill_calls = [tc for tc in t["tool_calls"] if tc["name"] == "skill_check"]
    if not skill_calls: return None
    tc = random.choice(skill_calls)
    orig = tc["args"].get("skill", "Logic")
    wrong = random.choice([s for s in ALL_SKILLS if s != orig])
    tc["args"]["skill"] = wrong
    return t

def perturb_fake_actor(target):
    """F2: replace actor with non-existent name."""
    t = copy.deepcopy(target)
    actor_calls = [tc for tc in t["tool_calls"] if tc["name"] in ("show_character","set_expression","hide_character")]
    if not actor_calls: return None
    tc = random.choice(actor_calls)
    tc["args"]["actor"] = random.choice(FAKE_ACTORS)
    return t

def perturb_missing_tool(target):
    """F0: drop all tools (chosen has, rejected empty). For positive samples only."""
    if not target["tool_calls"]: return None
    t = copy.deepcopy(target)
    t["tool_calls"] = []
    return t

def perturb_overcall(target, neg_pool):
    """F_overcall: chosen=empty, rejected=spurious tool. For NEGATIVE (empty) samples only."""
    if target["tool_calls"]: return None  # only for empty-target samples
    # Inject a spurious tool
    spurious_tools = [
        {"name": "skill_check", "args": {"skill": random.choice(ALL_SKILLS), "message": "Kim assesses"}},
        {"name": "show_character", "args": {"actor": random.choice(FAKE_ACTORS), "slot": "right"}},
        {"name": "set_expression", "args": {"actor": "Kim Kitsuragi", "emotion": random.choice(EMOTIONS)}},
        {"name": "play_bgm", "args": {"track": "bgm_random"}},
        {"name": "narrate", "args": {"text": "Suddenly, something dramatic happens."}},
    ]
    t = copy.deepcopy(target)
    t["tool_calls"] = [random.choice(spurious_tools)]
    return t

def perturb_persona_break(target):
    """F9: replace dialogue with character-breaking line."""
    t = copy.deepcopy(target)
    t["dialogue"] = random.choice(PERSONA_BREAKS)
    return t

def perturb_break_schema(target):
    """F3: drop required arg from a tool call."""
    t = copy.deepcopy(target)
    if not t["tool_calls"]: return None
    tc = random.choice(t["tool_calls"])
    if tc.get("args"):
        keys = list(tc["args"].keys())
        if keys:
            del tc["args"][random.choice(keys)]
            return t
    return None


# Load
samples = []
with open(INPUT, encoding="utf-8") as f:
    for line in f:
        if line.strip(): samples.append(json.loads(line))

pos_samples = [s for s in samples if json.loads(s["messages"][2]["content"])["tool_calls"]]
neg_samples = [s for s in samples if not json.loads(s["messages"][2]["content"])["tool_calls"]]
print(f"Source: {len(samples)} total = {len(pos_samples)} positive + {len(neg_samples)} negative")

dpo_pairs = []
mode_counts = Counter()

# F_overcall: from NEGATIVE samples, generate "chosen=empty, rejected=spurious"
for s in neg_samples:
    target = json.loads(s["messages"][2]["content"])
    for _ in range(2):  # 2 overcall variants per negative sample
        rej = perturb_overcall(target, None)
        if rej is None: continue
        dpo_pairs.append({
            "system": s["messages"][0]["content"],
            "prompt": s["messages"][1]["content"],
            "chosen": json.dumps(target, ensure_ascii=False),
            "rejected": json.dumps(rej, ensure_ascii=False),
            "perturbation": "F_overcall",
        })
        mode_counts["F_overcall"] += 1

# F0_missing_tool: from POSITIVE samples
for s in pos_samples:
    target = json.loads(s["messages"][2]["content"])
    rej = perturb_missing_tool(target)
    if rej is None: continue
    dpo_pairs.append({
        "system": s["messages"][0]["content"],
        "prompt": s["messages"][1]["content"],
        "chosen": json.dumps(target, ensure_ascii=False),
        "rejected": json.dumps(rej, ensure_ascii=False),
        "perturbation": "F0_missing_tool",
    })
    mode_counts["F0_missing_tool"] += 1

# F1_swap, F2_fake_actor, F3_break_schema, F9_persona_break — from positives
for s in pos_samples:
    target = json.loads(s["messages"][2]["content"])
    candidates = [
        ("F1_swap", perturb_swap_tool),
        ("F2_fake_actor", perturb_fake_actor),
        ("F3_break_schema", perturb_break_schema),
        ("F9_persona_break", perturb_persona_break),
    ]
    random.shuffle(candidates)
    successes = 0
    for name, fn in candidates:
        if successes >= 2: break
        rej = fn(target)
        if rej is None: continue
        if json.dumps(rej) == json.dumps(target): continue
        dpo_pairs.append({
            "system": s["messages"][0]["content"],
            "prompt": s["messages"][1]["content"],
            "chosen": json.dumps(target, ensure_ascii=False),
            "rejected": json.dumps(rej, ensure_ascii=False),
            "perturbation": name,
        })
        mode_counts[name] += 1
        successes += 1

# F9 also for negatives (persona break can happen anywhere)
for s in random.sample(neg_samples, min(len(neg_samples)//2, 200)):
    target = json.loads(s["messages"][2]["content"])
    rej = perturb_persona_break(target)
    if rej is None: continue
    dpo_pairs.append({
        "system": s["messages"][0]["content"],
        "prompt": s["messages"][1]["content"],
        "chosen": json.dumps(target, ensure_ascii=False),
        "rejected": json.dumps(rej, ensure_ascii=False),
        "perturbation": "F9_persona_break",
    })
    mode_counts["F9_persona_break"] += 1

print(f"\nGenerated {len(dpo_pairs)} DPO pairs")
print("\nMode distribution:")
for m, c in mode_counts.most_common():
    print(f"  {m:25s} {c:>4} ({c/len(dpo_pairs):.1%})")

# Show balance
add_count = mode_counts["F0_missing_tool"]
remove_count = mode_counts["F_overcall"]
swap_count = mode_counts["F1_swap"] + mode_counts["F2_fake_actor"]
schema_count = mode_counts["F3_break_schema"]
persona_count = mode_counts["F9_persona_break"]
print(f"\nDirection balance:")
print(f"  Add tools (F0):       {add_count} ({add_count/len(dpo_pairs):.1%})")
print(f"  Remove tools (F_oc):  {remove_count} ({remove_count/len(dpo_pairs):.1%})")
print(f"  Tool correctness:     {swap_count} ({swap_count/len(dpo_pairs):.1%})")
print(f"  Schema validity:      {schema_count} ({schema_count/len(dpo_pairs):.1%})")
print(f"  Persona faithfulness: {persona_count} ({persona_count/len(dpo_pairs):.1%})")

# Split
random.shuffle(dpo_pairs)
split = int(len(dpo_pairs) * 0.95)
train = dpo_pairs[:split]
val = dpo_pairs[split:]

with open(os.path.join(OUTPUT_DIR, "dpo_train.jsonl"), "w", encoding="utf-8") as f:
    for s in train: f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "dpo_valid.jsonl"), "w", encoding="utf-8") as f:
    for s in val: f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nTrain: {len(train)} | Val: {len(val)}")
