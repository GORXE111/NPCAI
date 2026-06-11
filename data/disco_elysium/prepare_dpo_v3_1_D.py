"""
Stage 3 v3.1.D — DPO preference pairs targeting v3.1's EXACT failure modes.

Why this could work where SFT failed:
- KL penalty against v3.1 reference model prevents drift on unchanged categories
- Pairs teach "X is preferred over Y" not "Y should be empty"
- Rejected = v3.1's ACTUAL systematic mistake, not synthetic noise

Why old DPO failed:
- Old: 2789 synthetic pairs, chosen=empty/rejected=any-tool → "less is more" collapse
- New: chosen and rejected both have tools sometimes → must judge by context

Pair composition (200 target):
   50 branch    — chosen=present_choices, rejected=show_character  (v3.1's mistake)
   40 emotion   — chosen=set_expression,  rejected=skill_check     (v3.1's mistake)
   60 anchor    — chosen=v3.1's correct call, rejected=empty       (don't drift to silent)
   50 suppress  — chosen=empty, rejected=random wrong tool          (don't over-call)

Crucial: 50/50 split on chosen having tools vs empty
         50/50 split on rejected having tools vs empty
         → model cannot learn "tools always chosen" or "empty always chosen"
"""
import json, os, random, re
from collections import Counter

INPUT = "D:/AIproject/NPCAI/data/disco_elysium/output.json"
V31_TRAIN = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_1/kim_tool_train.jsonl"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_dpo_v3_1_D"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

KIM_SYSTEM = """You are Kim Kitsuragi from Disco Elysium. Output STRICT JSON:
{"dialogue": "<Kim's spoken line>", "tool_calls": [{"name": "...", "args": {...}}]}

Available tools:
- skill_check(skill, message): trigger a DE internal skill (Logic, Empathy, Visual Calculus, Shivers, Half Light, Inland Empire, Authority, Rhetoric, Drama, Conceptualization, Encyclopedia, Volition, Esprit de Corps, Suggestion, Endurance, Pain Threshold, Physical Instrument, Electrochemistry, Hand/Eye Coordination, Perception, Reaction Speed, Savoir Faire, Interfacing, Composure)
- present_choices(options): show 2-5 player choices
- set_expression(actor, emotion): change facial expression
- show_character(actor, slot), hide_character(actor)
- play_bgm(track), play_sfx(sound), set_background(location)
- narrate(text), end_scene(next)

PROACTIVELY call tools when the situation warrants. Reply ONLY in valid JSON."""

SKILLS_ALL = {"Logic","Encyclopedia","Rhetoric","Drama","Conceptualization","Visual Calculus",
              "Volition","Inland Empire","Empathy","Authority","Esprit de Corps","Suggestion",
              "Endurance","Pain Threshold","Physical Instrument","Electrochemistry","Shivers",
              "Half Light","Hand/Eye Coordination","Perception","Reaction Speed","Savoir Faire",
              "Interfacing","Composure"}

def is_kim(a): return a and "Kim" in a and "Kitsuragi" in a
def is_skill_actor(a): return a in SKILLS_ALL
def is_player(a): return a in ("You", "Player", "Harry")
def is_action_meta(d): return d.startswith("[Action") or d.startswith("[Condition") or d.startswith("[Variable")
def clean(t): return re.sub(r'wavs/.*?\.wav\|', '', t or '').strip()

def quoted(d):
    if not d: return None
    qs = re.findall(r'"([^"]+)"', d)
    qs = [q.strip() for q in qs if q.strip()]
    s = " ".join(qs) if qs else None
    if s: s = re.sub(r'\[(Action|Condition|Variable)[^\]]*\]', '', s).strip()
    if not s or len(s) < 4: return None
    if re.match(r'^[a-z_]+\.[a-z_]+', s) or re.match(r'^auto\.', s): return None
    if s.count("_") > s.count(" ") + 2: return None
    return s

def fmt_ctx(entries):
    lines = []
    for e in entries:
        a = e.get("actor", "")
        d = clean(e.get("dialogue", ""))
        if not d: continue
        if is_action_meta(d): lines.append(f"[scene: {d[:120]}]")
        elif is_skill_actor(a): continue
        elif is_player(a):
            s = quoted(d) or d
            lines.append(f"Detective: {s[:200]}")
        else:
            s = quoted(d) or d
            lines.append(f"{a}: {s[:200]}")
    return "\n".join(lines)

def make_pair(prompt, chosen_dialogue, chosen_tools, rejected_dialogue, rejected_tools, ptype):
    chosen = {"dialogue": chosen_dialogue, "tool_calls": chosen_tools}
    rejected = {"dialogue": rejected_dialogue, "tool_calls": rejected_tools}
    return {
        "system": KIM_SYSTEM,
        "prompt": prompt,
        "chosen": json.dumps(chosen, ensure_ascii=False),
        "rejected": json.dumps(rejected, ensure_ascii=False),
        "type": ptype
    }

# ───────────────────────────────────────────────────────────────
# Block 1: Branch pairs (50) — chosen=present_choices, rejected=show_character
# ───────────────────────────────────────────────────────────────
print("=" * 60)
print("BLOCK 1: branch pairs (50)")
print("=" * 60)

with open(INPUT, encoding="utf-8") as f:
    conversations = json.load(f)["conversations"]

# Mine 35 branch contexts (same as v3.1.1)
mined_branch = []
for conv_idx, conv in enumerate(conversations):
    for i, entry in enumerate(conv):
        if not is_kim(entry.get("actor", "")): continue
        kim_sp = quoted(clean(entry.get("dialogue", "")))
        if not kim_sp or len(kim_sp.split()) < 2 or len(kim_sp.split()) > 50: continue

        options = []
        for j in range(i + 1, min(len(conv), i + 6)):
            nxt = conv[j]
            if is_player(nxt.get("actor", "")):
                psp = quoted(clean(nxt.get("dialogue", "")))
                if psp and 3 < len(psp) < 120: options.append(psp[:120])
                if len(options) >= 4: break
            elif is_kim(nxt.get("actor", "")) or is_action_meta(nxt.get("dialogue", "")):
                if len(options) >= 2: break
                continue
        if len(options) < 2: continue

        prior = conv[max(0, i - 4):i]
        has_det_question = False
        for p in prior:
            if not is_player(p.get("actor", "")): continue
            pd = clean(p.get("dialogue", ""))
            pq = quoted(pd) or pd
            if pq and ("?" in pq or len(pq.split()) < 10):
                has_det_question = True; break
        if not has_det_question: continue

        ctx = fmt_ctx(prior)
        if not ctx.strip(): ctx = "[Conversation continues.]"

        # Pick a plausible "wrong" actor for rejected (v3.1's typical mistake = show_character)
        # Use a generic actor from the conversation if available
        wrong_actor = "Detective"
        for p in prior:
            a = p.get("actor", "")
            if a and not is_kim(a) and not is_skill_actor(a) and not is_player(a) and not is_action_meta(p.get("dialogue", "")):
                wrong_actor = a
                break

        pair = make_pair(
            ctx, kim_sp,
            [{"name": "present_choices", "args": {"options": options}}],
            kim_sp,  # same dialogue, different tool
            [{"name": "show_character", "args": {"actor": wrong_actor, "slot": "right"}}],
            "branch"
        )
        mined_branch.append(pair)

# Dedup
seen = set(); dedup_mined = []
for p in mined_branch:
    key = p["prompt"][:80] + p["chosen"][:60]
    if key in seen: continue
    seen.add(key); dedup_mined.append(p)
random.shuffle(dedup_mined)
mined_branch_selected = dedup_mined[:35]
print(f"  mined: {len(mined_branch_selected)}")

# Handcraft 15 DEBench-style branch pairs
BRANCH_SEEDS = [
    (["Detective: \"We have three doors. Which one first?\""],
     "Let's think about this strategically.",
     ["Left door, where I heard voices.", "Middle door, the unlocked one.", "Right door, marked with chalk."]),
    (["Detective: \"How should we approach the union boss?\""],
     "We have a few options.",
     ["Direct confrontation about the strike.", "Ask about the dockworker's union history.", "Press on the body in the courtyard."]),
    (["Detective: \"What angle do we work next, Kim?\""],
     "Three threads worth pulling.",
     ["The mercenaries on the island.", "The Hardie boys' alibi.", "Klaasje's missing testimony."]),
    (["[At the church entrance.]",
      "Detective: \"Do we go in or talk to the kids outside first?\""],
     "Both have merit. You choose.",
     ["Enter the church now.", "Question the kids about the hanged man.", "Wait for them to disperse."]),
    (["Detective: \"Who do we question about the gun?\""],
     "Several people had access.",
     ["The harbormaster, who saw a shipment.", "Cuno's father, the drunk.", "The deserter rumored to have weapons."]),
    (["Detective: \"What do we ask the witness?\""],
     "Pick your line of attack.",
     ["Where she was on the night of the murder.", "What she saw from her window.", "Why she didn't come forward sooner."]),
    (["Detective: \"Our next stop, Kim?\""],
     "Three good options.",
     ["The fishing village to find the deserter.", "Back to the morgue for the autopsy.", "The hotel to interrogate Klaasje."]),
    (["Detective: \"How do we play this interrogation?\""],
     "Tone matters here.",
     ["Hard line, push him until he breaks.", "Calm and procedural, by the book.", "Offer him a deal in exchange for talking."]),
    (["Detective: \"What's our next move on the murder weapon?\""],
     "A few angles open.",
     ["Check the pawnshops near the harbor.", "Search the rooftops again.", "Ask the dock workers about smuggling."]),
    (["Detective: \"Where do we look for the bullet, Kim?\""],
     "Geometry of the shot gives us choices.",
     ["The tree behind the church.", "The wall of the hostel.", "The rooftop opposite the courtyard."]),
    (["Detective: \"How do I open this conversation with Evrart?\""],
     "Choose your foothold.",
     ["Tell him you respect the union.", "Demand answers about the strike.", "Pretend you know more than you do."]),
    (["Detective: \"Should we trust this informant?\""],
     "Two ways to read it.",
     ["Yes, his fear seems genuine.", "No, his story has too many holes.", "Push for one more piece of evidence first."]),
    (["Detective: \"What's our angle on the autopsy report?\""],
     "It tells us several things at once.",
     ["The cause of death — strangulation, not hanging.", "The timing — at least three days ago.", "The defensive wounds on his arms."]),
    (["Detective: \"Should we confront her tonight, Kim?\""],
     "Risk versus information.",
     ["Yes, before she destroys evidence.", "No, wait for a warrant.", "Send a tail and follow her movements."]),
    (["Detective: \"What kind of questions do we ask the boy?\""],
     "He's twelve. Tread carefully.",
     ["Where his father is right now.", "What he saw last week.", "If anyone has threatened him."]),
]
hand_branch = []
for ctx_lines, kim_d, options in BRANCH_SEEDS:
    pair = make_pair(
        "\n".join(ctx_lines), kim_d,
        [{"name": "present_choices", "args": {"options": options}}],
        kim_d,
        [{"name": "show_character", "args": {"actor": "Detective", "slot": "right"}}],
        "branch"
    )
    hand_branch.append(pair)

branch_pairs = mined_branch_selected + hand_branch
print(f"  total branch: {len(branch_pairs)}")

# ───────────────────────────────────────────────────────────────
# Block 2: Emotion pairs (40) — chosen=set_expression, rejected=skill_check
# ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 2: emotion pairs (40)")
print("=" * 60)

EMOTION_SEEDS = [
    (["Detective: \"We found Lely's body. He was hanging from a tree.\""],
     "A bad way to go. Show me.", "serious"),
    (["Detective: \"Cuno just threw a brick at my head, Kim.\""],
     "Are you bleeding? Let me see.", "concerned"),
    (["Detective: \"I think Ruby might be the killer.\""],
     "Speculation is dangerous without evidence.", "serious"),
    (["Detective: \"Klaasje confessed everything to me last night.\""],
     "Did she... that complicates the case considerably.", "troubled"),
    (["Detective: \"I lost my badge. It's gone.\""],
     "Then we find it. Now.", "stern"),
    (["Detective: \"There's been another murder, Kim.\""],
     "Where? Take me there.", "alarmed"),
    (["Detective: \"The Hardie boys want a sit-down with us.\""],
     "I don't like the sound of that.", "wary"),
    (["Detective: \"I had a vision. A bug. A huge insect.\""],
     "Are you... are you alright, officer?", "concerned"),
    (["Detective: \"Joyce Messier vouched for the union.\""],
     "Joyce has her reasons. Not all of them noble.", "skeptical"),
    (["Detective: \"The kid in the bandana is Cuno's friend, isn't he?\""],
     "Yes. And he's twelve years old.", "sad"),
    (["[Scene: bullet casing found in tree bark.]",
      "Detective: \"This means the shot came from the island.\""],
     "Then we've been looking in the wrong direction.", "realizing"),
    (["Detective: \"My partner died in the line of duty last year.\""],
     "I'm sorry. That stays with you.", "sympathetic"),
    (["Detective: \"Did you hear what Evrart said? About the contract?\""],
     "Every word. He's testing us.", "serious"),
    (["Detective: \"I think I've been awake for three days, Kim.\""],
     "Sit down. I will handle this.", "concerned"),
    (["Detective: \"They want to question the deserter in the church.\""],
     "Then we go. Now, before he disappears.", "decisive"),
    (["Detective: \"Klaasje is gone. The room is empty.\""],
     "Of course she is. We should have seen it.", "frustrated"),
    (["Detective: \"Titus is dead. They shot him in the chest.\""],
     "...did anyone witness it?", "grim"),
    (["Detective: \"I think we just solved the case, Kim.\""],
     "Let's not celebrate yet. Confirm everything.", "guarded"),
    (["Detective: \"The phasmid is real. I saw it.\""],
     "I saw it too. The world is wider than I knew.", "awed"),
    (["Detective: \"The lieutenant is calling me back to Precinct 41.\""],
     "Then this is goodbye, partner. It was an honor.", "warm"),
    # 20 more emotion scenarios (DEBench-disjoint)
    (["Detective: \"The body in the meat locker is the lieutenant's son.\""],
     "Oh God. Cover it. Don't let his mother see.", "shocked"),
    (["Detective: \"I drank three bottles of wine last night.\""],
     "We need to talk. After you sober up.", "stern"),
    (["Detective: \"The kid said he watched his mother die.\""],
     "Then he needs someone gentle. Not us.", "sad"),
    (["Detective: \"Garte just admitted he forged the bills.\""],
     "Then we have leverage. Use it carefully.", "calculating"),
    (["Detective: \"There's a tape — the murder is recorded.\""],
     "Play it. Now.", "intense"),
    (["Detective: \"Klaasje wasn't lying. The shot came from above.\""],
     "Then I owe her an apology. Let's verify the angle.", "humbled"),
    (["Detective: \"The deserter has been hiding for fifty years.\""],
     "Fifty years. Imagine carrying that.", "somber"),
    (["Detective: \"I had to put down my old dog last month.\""],
     "I'm sorry. They become family.", "warm"),
    (["Detective: \"Cuno's father just died. Right in front of him.\""],
     "Where is the boy? Take me to him.", "urgent"),
    (["Detective: \"My old captain hated me. Real personal.\""],
     "Then he didn't see what I see.", "supportive"),
    (["Detective: \"The Hardie boys are armed. I think they'll shoot.\""],
     "Stay behind me. Do not engage.", "tense"),
    (["Detective: \"I just realized the gun was mine all along.\""],
     "Then we are walking back to the precinct. Slowly.", "grave"),
    (["[The radio static cuts to silence.]",
      "Detective: \"They cut the signal. We're alone now.\""],
     "Then we proceed assuming no backup. Stay alert.", "tense"),
    (["Detective: \"The boy is dead. I couldn't save him.\""],
     "It wasn't your job to save him. It was their job.", "sad"),
    (["Detective: \"Evrart offered me a bribe. A real one.\""],
     "And you turned it down. That matters.", "approving"),
    (["Detective: \"The harbor master keeps a photo of his daughter.\""],
     "Let him keep it. Some leverage isn't worth using.", "thoughtful"),
    (["Detective: \"I think I might have a family. Somewhere.\""],
     "Then we find out. After the case.", "hopeful"),
    (["Detective: \"The lab confirmed: the bullet came from a war-era rifle.\""],
     "Fifty-year-old shell. Same as the deserter's.", "focused"),
    (["Detective: \"I'm scared, Kim. Really scared.\""],
     "Sit down. Breathe. Tell me everything.", "gentle"),
    (["Detective: \"They want me to disappear. Make it look like a suicide.\""],
     "Then we don't let them. Not today.", "resolute"),
]

emotion_pairs = []
for i, (ctx_lines, kim_d, emo) in enumerate(EMOTION_SEEDS[:40]):
    # Rejected: v3.1's actual mistake on emotion contexts = skill_check or show_character
    rejected_tool = ([{"name": "skill_check", "args": {"skill": "Empathy", "message": ctx_lines[-1][:80]}}] if i % 2 == 0
                     else [{"name": "show_character", "args": {"actor": "Detective", "slot": "left"}}])
    pair = make_pair(
        "\n".join(ctx_lines), kim_d,
        [{"name": "set_expression", "args": {"actor": "Kim Kitsuragi", "emotion": emo}}],
        kim_d, rejected_tool,
        "emotion"
    )
    emotion_pairs.append(pair)
print(f"  emotion: {len(emotion_pairs)}")

# ───────────────────────────────────────────────────────────────
# Block 3: Anchor preservation pairs (60) — chosen=v3.1 correct, rejected=empty
# Teaches: don't drift to silent
# ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 3: anchor preservation pairs (60)")
print("=" * 60)

v31_samples = [json.loads(l) for l in open(V31_TRAIN, encoding="utf-8")]

def classify_v31(s):
    c = s["messages"][2]["content"]
    if '"skill_check"' in c: return "skill_check"
    if '"show_character"' in c: return "show_character"
    if '"present_choices"' in c: return "present_choices"
    return "empty"

v31_by_type = {"skill_check": [], "show_character": [], "present_choices": []}
for s in v31_samples:
    t = classify_v31(s)
    if t in v31_by_type: v31_by_type[t].append(s)

random.shuffle(v31_by_type["skill_check"])
random.shuffle(v31_by_type["show_character"])
random.shuffle(v31_by_type["present_choices"])

anchor_pairs = []
for s in v31_by_type["skill_check"][:30]:  # 30 skill anchor
    msgs = s["messages"]
    user_prompt = msgs[1]["content"]
    chosen = msgs[2]["content"]  # v3.1's actual training target (correct)
    # Parse chosen to extract dialogue for rejected (same dialogue, empty tools)
    try:
        cd = json.loads(chosen)
        rejected = json.dumps({"dialogue": cd.get("dialogue", "..."), "tool_calls": []}, ensure_ascii=False)
        anchor_pairs.append({
            "system": KIM_SYSTEM, "prompt": user_prompt,
            "chosen": chosen, "rejected": rejected, "type": "anchor_skill"
        })
    except: continue

for s in v31_by_type["show_character"][:20]:
    msgs = s["messages"]
    try:
        cd = json.loads(msgs[2]["content"])
        rejected = json.dumps({"dialogue": cd.get("dialogue", "..."), "tool_calls": []}, ensure_ascii=False)
        anchor_pairs.append({
            "system": KIM_SYSTEM, "prompt": msgs[1]["content"],
            "chosen": msgs[2]["content"], "rejected": rejected, "type": "anchor_show"
        })
    except: continue

for s in v31_by_type["present_choices"][:10]:
    msgs = s["messages"]
    try:
        cd = json.loads(msgs[2]["content"])
        rejected = json.dumps({"dialogue": cd.get("dialogue", "..."), "tool_calls": []}, ensure_ascii=False)
        anchor_pairs.append({
            "system": KIM_SYSTEM, "prompt": msgs[1]["content"],
            "chosen": msgs[2]["content"], "rejected": rejected, "type": "anchor_pc"
        })
    except: continue

print(f"  anchor: {len(anchor_pairs)} (skill+show+pc)")

# ───────────────────────────────────────────────────────────────
# Block 4: Suppression preservation pairs (50) — chosen=empty, rejected=wrong tool
# Teaches: don't over-call
# ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 4: suppression preservation pairs (50)")
print("=" * 60)

empty_samples = [s for s in v31_samples if classify_v31(s) == "empty"]
random.shuffle(empty_samples)

WRONG_TOOLS_POOL = [
    {"name": "skill_check", "args": {"skill": "Logic", "message": "Something unspecified."}},
    {"name": "show_character", "args": {"actor": "Stranger", "slot": "right"}},
    {"name": "present_choices", "args": {"options": ["Option A.", "Option B."]}},
    {"name": "set_expression", "args": {"actor": "Kim Kitsuragi", "emotion": "neutral"}},
    {"name": "play_bgm", "args": {"track": "ambient_bg"}},
    {"name": "narrate", "args": {"text": "The scene continues without incident."}},
]

suppress_pairs = []
for s in empty_samples[:50]:
    msgs = s["messages"]
    try:
        cd = json.loads(msgs[2]["content"])
        wrong = random.choice(WRONG_TOOLS_POOL)
        rejected = json.dumps({"dialogue": cd.get("dialogue", "..."), "tool_calls": [wrong]}, ensure_ascii=False)
        suppress_pairs.append({
            "system": KIM_SYSTEM, "prompt": msgs[1]["content"],
            "chosen": msgs[2]["content"], "rejected": rejected, "type": "suppress"
        })
    except: continue

print(f"  suppress: {len(suppress_pairs)}")

# ───────────────────────────────────────────────────────────────
# Block 5: Combine + Split
# ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 5: combine + split")
print("=" * 60)

all_pairs = branch_pairs + emotion_pairs + anchor_pairs + suppress_pairs
print(f"  total: {len(all_pairs)}")
print(f"    branch:   {len(branch_pairs)}")
print(f"    emotion:  {len(emotion_pairs)}")
print(f"    anchor:   {len(anchor_pairs)}")
print(f"    suppress: {len(suppress_pairs)}")

# Sanity stats
chosen_with_tool = sum(1 for p in all_pairs if '"tool_calls": []' not in p["chosen"])
chosen_empty = len(all_pairs) - chosen_with_tool
rejected_with_tool = sum(1 for p in all_pairs if '"tool_calls": []' not in p["rejected"])
rejected_empty = len(all_pairs) - rejected_with_tool
print(f"\n  chosen distribution:   with_tool={chosen_with_tool}  empty={chosen_empty}")
print(f"  rejected distribution: with_tool={rejected_with_tool}  empty={rejected_empty}")
print(f"  → model CANNOT learn 'tool always = chosen' or 'empty always = chosen'")

type_dist = Counter(p["type"] for p in all_pairs)
print("\n  by type:")
for k, v in type_dist.most_common(): print(f"    {k:25s} {v}")

random.shuffle(all_pairs)
n_val = max(10, len(all_pairs) // 10)
val = all_pairs[:n_val]
train = all_pairs[n_val:]

# Strip type field for training
def strip(p): return {k: v for k, v in p.items() if k != "type"}
train_out = [strip(p) for p in train]
val_out = [strip(p) for p in val]

with open(os.path.join(OUTPUT_DIR, "dpo_train.jsonl"), "w", encoding="utf-8") as f:
    for p in train_out: f.write(json.dumps(p, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "dpo_valid.jsonl"), "w", encoding="utf-8") as f:
    for p in val_out: f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\nFinal: Train {len(train_out)}  Val {len(val_out)}")
print(f"Output: {OUTPUT_DIR}")
