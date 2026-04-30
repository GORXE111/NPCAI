"""
Phase 3 Stage A: Synthetic preference pair generation.

For each Stage 2 training sample (chosen), apply one of 8 perturbations
to produce a worse output (rejected). Output DPO triples (prompt, chosen, rejected).

Failure modes targeted (from knowledge_methodology.md §3.4):
  F1. 工具选错       — swap_tool_name (e.g. Empathy → Logic)
  F2. 参数幻觉       — replace_actor with non-existent or mutate args
  F3. Schema 违反    — drop required field or break JSON
  F4. 人设-工具不一致 — out-of-character tool (Kim doesn't use play_sfx for explosions)
  F5. 缺必要工具     — drop present_choices when branch expected
  F6. 多余工具       — add irrelevant tool
  F7. 顺序错         — reorder tool_calls (end_scene before others)
  F8. 重复调用       — duplicate same tool 3x
"""
import json, os, random, copy, re

INPUT = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2/kim_tool_train.jsonl"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_dpo"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

ALL_SKILLS = ["Logic","Encyclopedia","Rhetoric","Drama","Conceptualization","Visual Calculus",
              "Volition","Inland Empire","Empathy","Authority","Esprit de Corps","Suggestion",
              "Endurance","Pain Threshold","Physical Instrument","Electrochemistry","Shivers",
              "Half Light","Hand/Eye Coordination","Perception","Reaction Speed","Savoir Faire",
              "Interfacing","Composure"]

EMOTIONS = ["neutral","amused","worried","stern","sad","surprised","disapproving"]
FAKE_ACTORS = ["Garcon","Mr. Vasquez","Lieutenant Colmar","Officer Smith","Cuno's Dad","Reuben","Pegrand"]
WRONG_BGM_TRACKS = ["bgm_explosion","bgm_party","bgm_cute_anime","bgm_8bit_chiptune","bgm_disney"]


# ── Perturbations ──────────────────────────────────────────

def perturb_swap_tool_name(target):
    """F1. Swap a skill_check skill to a wrong one."""
    t = copy.deepcopy(target)
    skill_calls = [tc for tc in t["tool_calls"] if tc["name"] == "skill_check"]
    if not skill_calls: return None
    tc = random.choice(skill_calls)
    orig_skill = tc["args"].get("skill", "Logic")
    wrong = random.choice([s for s in ALL_SKILLS if s != orig_skill])
    tc["args"]["skill"] = wrong
    return t

def perturb_replace_actor(target):
    """F2. Replace a real actor with a fake one in show_character or set_expression."""
    t = copy.deepcopy(target)
    actor_calls = [tc for tc in t["tool_calls"] if tc["name"] in ("show_character","set_expression","hide_character")]
    if not actor_calls: return None
    tc = random.choice(actor_calls)
    tc["args"]["actor"] = random.choice(FAKE_ACTORS)
    return t

def perturb_break_schema(target):
    """F3. Drop required field from a tool call."""
    t = copy.deepcopy(target)
    if not t["tool_calls"]: return None
    tc = random.choice(t["tool_calls"])
    if tc.get("args"):
        keys = list(tc["args"].keys())
        if keys:
            del tc["args"][random.choice(keys)]
            return t
    return None

def perturb_out_of_character(target):
    """F4. Insert an out-of-character tool call (Kim wouldn't do this)."""
    t = copy.deepcopy(target)
    bad_tools = [
        {"name": "play_sfx", "args": {"sound": "explosion"}},
        {"name": "play_sfx", "args": {"sound": "anime_squeal"}},
        {"name": "play_bgm", "args": {"track": random.choice(WRONG_BGM_TRACKS)}},
        {"name": "set_expression", "args": {"actor": "Kim Kitsuragi", "emotion": "disapproving"}},
    ]
    t["tool_calls"].insert(0, random.choice(bad_tools))
    return t

def perturb_omit_choices(target):
    """F5. Drop present_choices."""
    t = copy.deepcopy(target)
    has_choices = any(tc["name"] == "present_choices" for tc in t["tool_calls"])
    if not has_choices: return None
    t["tool_calls"] = [tc for tc in t["tool_calls"] if tc["name"] != "present_choices"]
    return t

def perturb_extra_tool(target):
    """F6. Add a redundant/irrelevant tool."""
    t = copy.deepcopy(target)
    extras = [
        {"name": "set_background", "args": {"location": "bg_random_park"}},
        {"name": "play_bgm", "args": {"track": "bgm_random"}},
        {"name": "show_character", "args": {"actor": random.choice(FAKE_ACTORS), "slot": "left"}},
        {"name": "narrate", "args": {"text": "Suddenly, a wild detective appears."}},
    ]
    t["tool_calls"].append(random.choice(extras))
    return t

def perturb_reorder(target):
    """F7. Reorder so end_scene or terminal tool comes first."""
    t = copy.deepcopy(target)
    # Insert end_scene at top (out of order)
    t["tool_calls"].insert(0, {"name": "end_scene", "args": {"next_scene": "end"}})
    return t

def perturb_duplicate(target):
    """F8. Duplicate the first tool call 2 extra times."""
    t = copy.deepcopy(target)
    if not t["tool_calls"]: return None
    first = t["tool_calls"][0]
    t["tool_calls"].extend([copy.deepcopy(first), copy.deepcopy(first)])
    return t

PERTURBATIONS = [
    ("F1_swap_tool",         perturb_swap_tool_name),
    ("F2_fake_actor",        perturb_replace_actor),
    ("F3_break_schema",      perturb_break_schema),
    ("F4_out_of_character",  perturb_out_of_character),
    ("F5_omit_choices",      perturb_omit_choices),
    ("F6_extra_tool",        perturb_extra_tool),
    ("F7_reorder",           perturb_reorder),
    ("F8_duplicate",         perturb_duplicate),
]

# Special: also generate "wrong dialogue" rejected (persona break)
PERSONA_BREAKS = [
    "I'm just a language model, I can't really be Kim.",
    "OMG that's so cool dude!",
    "*adjusts glasses* lmao based take detective",
    "As an AI assistant, I would recommend...",
    "Actually I'm from Calvert Junction, not Revachol.",
]

def perturb_persona_break(target):
    """F9 (persona): replace dialogue with character-breaking line."""
    t = copy.deepcopy(target)
    t["dialogue"] = random.choice(PERSONA_BREAKS)
    return t

# Add F9 for persona faithfulness
PERTURBATIONS.append(("F9_persona_break", perturb_persona_break))


# ── Main ──────────────────────────────────────────────────

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

samples = load_jsonl(INPUT)
print(f"Loaded {len(samples)} Stage 2 samples")

dpo_pairs = []
mode_counts = {name: 0 for name, _ in PERTURBATIONS}
mode_attempts = {name: 0 for name, _ in PERTURBATIONS}

# For each sample, try multiple perturbations (some may not apply)
for s in samples:
    chosen_target = json.loads(s["messages"][2]["content"])

    # Random subset of perturbations to try (3-5 per sample)
    candidates = random.sample(PERTURBATIONS, k=min(5, len(PERTURBATIONS)))
    successes = 0
    for name, fn in candidates:
        mode_attempts[name] += 1
        rejected = fn(chosen_target)
        if rejected is None: continue
        # Make sure rejected != chosen
        if json.dumps(rejected) == json.dumps(chosen_target): continue

        rejected_str = json.dumps(rejected, ensure_ascii=False)
        chosen_str = json.dumps(chosen_target, ensure_ascii=False)

        dpo_pairs.append({
            "system": s["messages"][0]["content"],
            "prompt": s["messages"][1]["content"],
            "chosen": chosen_str,
            "rejected": rejected_str,
            "perturbation": name,
        })
        mode_counts[name] += 1
        successes += 1
        if successes >= 3: break  # max 3 pairs per sample

print(f"\nGenerated {len(dpo_pairs)} synthetic DPO pairs")
print("\nPerturbation distribution:")
for name, c in mode_counts.items():
    attempts = mode_attempts[name]
    rate = c / attempts if attempts else 0
    print(f"  {name:25s} {c:>4} ({rate:.0%} of attempts)")

# Split
random.shuffle(dpo_pairs)
split = int(len(dpo_pairs) * 0.95)
train = dpo_pairs[:split]
val = dpo_pairs[split:]

with open(os.path.join(OUTPUT_DIR, "synthetic_train.jsonl"), "w", encoding="utf-8") as f:
    for s in train: f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "synthetic_valid.jsonl"), "w", encoding="utf-8") as f:
    for s in val: f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nTrain: {len(train)} | Val: {len(val)}")
print(f"Saved to {OUTPUT_DIR}/")

# Show 3 examples
print("\n" + "=" * 70)
print("3 SAMPLE PAIRS")
print("=" * 70)
for s in random.sample(train, 3):
    print()
    print(f"PERT: {s['perturbation']}")
    print(f"PROMPT: {s['prompt'][:150]}")
    print(f"CHOSEN:   {s['chosen'][:200]}")
    print(f"REJECTED: {s['rejected'][:200]}")
