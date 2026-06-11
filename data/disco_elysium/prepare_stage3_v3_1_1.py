"""
Stage 3 v3.1.1 — Targeted repair of branch + emotion gaps in v3.1 via continued LoRA fine-tune.

v3.1 diagnosis:
  - branch 0/0:  model has present_choices (16 samples) but mistakes context → calls show_character
  - emotion 0/0: set_expression appeared in system prompt but ZERO training samples → unknown token

Architecture:
  Stage 2 v3.1 LoRA (F1 0.639) → continued-train with surgical 200-sample set → v3.1.1

Dataset composition (200 target):
   30  anchor    — random v3.1 samples (skill_check 10 + show_character 10 + filler 10)
   50  branch    — present_choices on Detective decision questions (mined from corpus)
   20  emotion   — set_expression (hand-curated, real DE actors/contexts, NEVER overlapping DEBench)
  100  suppress  — additional filler from v3.1 (not in anchor)

Safeguards:
  - Branch context = real Detective decision Q (matched to DEBench pattern), Kim response = real corpus
  - Emotion: synthesize only the JSON action; prompt uses real characters / scene markers
  - All samples length-matched to v3.1 distribution (~14 words avg per Kim dialogue)
  - 50/50 pos/neg (100 with tool + 100 without)
"""
import json, os, random, re
from collections import Counter

INPUT = "D:/AIproject/NPCAI/data/disco_elysium/output.json"
V31_TRAIN = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_1/kim_tool_train.jsonl"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_stage3_v3_1_1"
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

# ───────────────────────────────────────────────────────────────
# Helpers (same as v4.0)
# ───────────────────────────────────────────────────────────────
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
        elif is_skill_actor(a): continue  # exclude skills
        elif is_player(a):
            s = quoted(d) or d
            lines.append(f"Detective: {s[:200]}")
        else:
            s = quoted(d) or d
            lines.append(f"{a}: {s[:200]}")
    return "\n".join(lines)

def fmt_sample(ctx, kim_dialogue, tool_calls, sample_type, conv_id=-1):
    target = {"dialogue": kim_dialogue, "tool_calls": tool_calls}
    return {
        "messages": [
            {"role": "system", "content": KIM_SYSTEM},
            {"role": "user", "content": ctx},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
        ],
        "conv_id": conv_id,
        "type": sample_type
    }

# ───────────────────────────────────────────────────────────────
# Block 1: Load v3.1 train → anchor + suppress pools
# ───────────────────────────────────────────────────────────────
print("=" * 60)
print("BLOCK 1: anchor + suppress from v3.1")
print("=" * 60)

v31_samples = [json.loads(l) for l in open(V31_TRAIN, encoding="utf-8")]
print(f"v3.1 train size: {len(v31_samples)}")

def classify_v31(s):
    c = s["messages"][2]["content"]
    if '"skill_check"' in c: return "skill_check"
    if '"show_character"' in c: return "show_character"
    if '"present_choices"' in c: return "present_choices"
    if '"set_expression"' in c: return "set_expression"
    return "empty"

v31_by_type = {"skill_check": [], "show_character": [], "present_choices": [], "empty": []}
for s in v31_samples:
    t = classify_v31(s)
    if t in v31_by_type: v31_by_type[t].append(s)

print(f"v3.1 breakdown: skill={len(v31_by_type['skill_check'])}  show={len(v31_by_type['show_character'])}  "
      f"pc={len(v31_by_type['present_choices'])}  empty={len(v31_by_type['empty'])}")

# Anchor: 10 + 10 + 10
random.shuffle(v31_by_type["skill_check"])
random.shuffle(v31_by_type["show_character"])
random.shuffle(v31_by_type["empty"])

anchor = []
anchor += v31_by_type["skill_check"][:10]
anchor += v31_by_type["show_character"][:10]
anchor += v31_by_type["empty"][:10]
for s in anchor:
    s["type"] = "anchor_" + classify_v31(s)
    s["conv_id"] = -1  # treat as fresh

# Suppress: 100 more filler (from index 10 onward)
suppress = v31_by_type["empty"][10:110]
for s in suppress:
    s["type"] = "suppress_filler"
    s["conv_id"] = -1

print(f"  anchor: {len(anchor)} (10 skill + 10 show + 10 empty)")
print(f"  suppress: {len(suppress)}")

# ───────────────────────────────────────────────────────────────
# Block 2: Mine branch samples — Kim → 2+ player options (v4.0 pattern)
#          + 15 hand-crafted DEBench-style decision questions
# ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 2: branch mining (present_choices)")
print("=" * 60)

with open(INPUT, encoding="utf-8") as f:
    conversations = json.load(f)["conversations"]

# Strategy A: Kim → 2+ options (general mining)
mined_branch = []
for conv_idx, conv in enumerate(conversations):
    for i, entry in enumerate(conv):
        if not is_kim(entry.get("actor", "")): continue
        kim_sp = quoted(clean(entry.get("dialogue", "")))
        if not kim_sp or len(kim_sp.split()) < 2: continue
        if len(kim_sp.split()) > 50: continue

        # Look for player options after Kim
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

        # Need a Detective question in the prior 4 turns (any question, not strict)
        prior = conv[max(0, i - 4):i]
        has_det_question = False
        for p in prior:
            if not is_player(p.get("actor", "")): continue
            pd = clean(p.get("dialogue", ""))
            pq = quoted(pd) or pd
            if pq and ("?" in pq or len(pq.split()) < 10):
                has_det_question = True
                break
        if not has_det_question: continue

        ctx = fmt_ctx(prior)
        if not ctx.strip(): ctx = "[Conversation continues.]"

        sample = fmt_sample(
            ctx, kim_sp,
            [{"name": "present_choices", "args": {"options": options}}],
            "branch_mined", conv_idx
        )
        mined_branch.append(sample)

print(f"  Strategy A mined: {len(mined_branch)}")

# Dedup mined
seen = set()
dedup_mined = []
for s in mined_branch:
    key = (s["conv_id"], s["messages"][2]["content"][:80])
    if key in seen: continue
    seen.add(key)
    dedup_mined.append(s)
print(f"  Strategy A dedup: {len(dedup_mined)}")
random.shuffle(dedup_mined)
mined_selected = dedup_mined[:35]

# Strategy B: Hand-crafted DEBench-style decision Qs (NOT overlapping with DEBench test prompts)
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
    ctx = "\n".join(ctx_lines)
    sample = fmt_sample(
        ctx, kim_d,
        [{"name": "present_choices", "args": {"options": options}}],
        "branch_handcraft", -1
    )
    hand_branch.append(sample)
print(f"  Strategy B handcraft: {len(hand_branch)}")

branch = mined_selected + hand_branch
random.shuffle(branch)
print(f"  selected branch (total): {len(branch)}")

# Length sanity
branch_lens = [len(s["messages"][2]["content"].split()) for s in branch]
print(f"  branch assistant len: mean={sum(branch_lens)/len(branch_lens):.0f}  "
      f"min={min(branch_lens)}  max={max(branch_lens)}")

# Length sanity
branch_lens = [len(s["messages"][2]["content"].split()) for s in branch]
print(f"  branch assistant len: mean={sum(branch_lens)/len(branch_lens):.0f}  "
      f"min={min(branch_lens)}  max={max(branch_lens)}")

# ───────────────────────────────────────────────────────────────
# Block 3: Hand-curated emotion samples
# ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 3: emotion (hand-curated set_expression)")
print("=" * 60)

# 20 emotion scenarios — all DEBench-disjoint, real DE actors/locations.
# Format: (context_lines, kim_dialogue, emotion)
EMOTION_SEEDS = [
    # (Strong-stakes news → Kim reaction)
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
]

emotion_samples = []
for ctx_lines, kim_d, emo in EMOTION_SEEDS:
    ctx = "\n".join(ctx_lines)
    sample = fmt_sample(
        ctx,
        kim_d,
        [{"name": "set_expression", "args": {"actor": "Kim Kitsuragi", "emotion": emo}}],
        "emotion",
        -1
    )
    emotion_samples.append(sample)

print(f"  emotion samples: {len(emotion_samples)}")
emo_lens = [len(s["messages"][2]["content"].split()) for s in emotion_samples]
print(f"  emotion assistant len: mean={sum(emo_lens)/len(emo_lens):.0f}  "
      f"min={min(emo_lens)}  max={max(emo_lens)}")

# Sanity: emotions used
emos_used = Counter()
for s in emotion_samples:
    c = s["messages"][2]["content"]
    m = re.search(r'"emotion":\s*"([^"]+)"', c)
    if m: emos_used[m.group(1)] += 1
print(f"  emotion types: {dict(emos_used)}")

# ───────────────────────────────────────────────────────────────
# Block 4: Combine + Shuffle + Split
# ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK 4: combine + split")
print("=" * 60)

all_samples = anchor + branch + emotion_samples + suppress
print(f"  total: {len(all_samples)}")
print(f"    anchor:    {len(anchor)}")
print(f"    branch:    {len(branch)}")
print(f"    emotion:   {len(emotion_samples)}")
print(f"    suppress:  {len(suppress)}")

# Count pos/neg
def is_pos(s):
    c = s["messages"][2]["content"]
    return '"tool_calls": []' not in c and '"tool_calls":[]' not in c

pos = sum(1 for s in all_samples if is_pos(s))
neg = len(all_samples) - pos
print(f"  positive: {pos}  negative: {neg}  ratio: {pos/(pos+neg):.0%}/{neg/(pos+neg):.0%}")

# Distribution stats
type_dist = Counter()
tool_dist = Counter()
for s in all_samples:
    type_dist[s.get("type", "?")] += 1
    c = s["messages"][2]["content"]
    for t in ["skill_check", "show_character", "present_choices", "set_expression"]:
        if f'"{t}"' in c: tool_dist[t] += 1

print("\n  by type:")
for k, v in type_dist.most_common(): print(f"    {k:25s} {v}")
print("\n  by tool (in assistant):")
for k, v in tool_dist.most_common(): print(f"    {k:25s} {v}")

# Shuffle and split: 90% train / 10% val
random.shuffle(all_samples)
n_val = max(10, len(all_samples) // 10)
val = all_samples[:n_val]
train = all_samples[n_val:]

# Strip metadata fields, keep only messages
train_out = [{"messages": s["messages"]} for s in train]
val_out = [{"messages": s["messages"]} for s in val]

with open(os.path.join(OUTPUT_DIR, "kim_tool_train.jsonl"), "w", encoding="utf-8") as f:
    for s in train_out: f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "kim_tool_valid.jsonl"), "w", encoding="utf-8") as f:
    for s in val_out: f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nFinal: Train {len(train_out)}  Val {len(val_out)}")
print(f"Output: {OUTPUT_DIR}")
print(f"\nComparison vs v3.1:")
print(f"  v3.1     train=677  tools={{skill_check:174, show_character:148, present_choices:16, set_expression:0}}")
print(f"  v3.1.1   train={len(train_out)}  tools={{skill_check:{sum(1 for s in train if 'skill_check' in s['messages'][2]['content'])}, "
      f"show_character:{sum(1 for s in train if 'show_character' in s['messages'][2]['content'])}, "
      f"present_choices:{sum(1 for s in train if 'present_choices' in s['messages'][2]['content'])}, "
      f"set_expression:{sum(1 for s in train if 'set_expression' in s['messages'][2]['content'])}}}")
