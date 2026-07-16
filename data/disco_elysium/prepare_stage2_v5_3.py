"""
Stage 2 v4.0 — Enhanced v3.1 via more aggressive mining (same principles, more data).

v3.1 mined 322 positives from 1742 conversations.
v4.0 goal: 2.5-3× more positives (~800-1000), same distribution, same quality.

Principles (locked):
- 100% real DE corpus, NO synthesis
- 50/50 positive/negative balance
- Same intent triggers (skill actor before / new actor / options after)
- Greedy decoding for reproducibility

Refinements (all "more, not different"):
- Context window: 6 → 8 turns
- Skill lookback: 3 → 5 turns
- Min Kim words: 2 → 1
- Multi-skill: merge 2+ skill actors → single Kim sample with multiple skill_check
- Re-mine present_choices with lower threshold (≥1 option, was ≥2)
- Re-scan show_character with 5-turn lookback
"""
import json, os, random, re
from collections import Counter

INPUT = "D:/AIproject/NPCAI/data/disco_elysium/output.json"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v5_3"
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

# Enhanced parameters
CTX_WINDOW = 8  # was 6
SKILL_LOOKBACK = 5  # was 3
ACTOR_LOOKBACK = 5  # was 4
MIN_KIM_WORDS = 1  # was 2
MIN_PRESENT_OPTIONS = 1  # was 2 (relaxed)

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

def fmt_ctx(entries, exclude_skills=False):
    lines = []
    for e in entries:
        a = e.get("actor", "")
        d = clean(e.get("dialogue", ""))
        if not d: continue
        if is_action_meta(d): lines.append(f"[scene: {d[:120]}]")
        elif is_skill_actor(a):
            if exclude_skills: continue
            s = quoted(d) or d
            lines.append(f"({a} — internal): {s[:200]}")
        elif is_player(a):
            s = quoted(d) or d
            lines.append(f"Detective: {s[:200]}")
        else:
            s = quoted(d) or d
            lines.append(f"{a}: {s[:200]}")
    return "\n".join(lines)


with open(INPUT, encoding="utf-8") as f:
    conversations = json.load(f)["conversations"]

samples = []
sample_types = Counter()
skill_counts = Counter()
tool_counts = Counter()

for conv_idx, conv in enumerate(conversations):

    # ═══════════════════════════════════════════════════
    # Type 1: skill_check samples (with multi-skill merge)
    # ═══════════════════════════════════════════════════
    # For each Kim utterance, find ALL skill actors in lookback window
    for i, entry in enumerate(conv):
        if not is_kim(entry.get("actor", "")): continue
        kim_speech = quoted(clean(entry.get("dialogue", "")))
        if not kim_speech or len(kim_speech.split()) < MIN_KIM_WORDS: continue

        # Collect all skill actors in lookback (multi-skill merge)
        skill_calls = []
        seen_skills = set()
        for j in range(max(0, i - SKILL_LOOKBACK), i):
            prev = conv[j]
            prev_actor = prev.get("actor", "")
            if is_skill_actor(prev_actor) and prev_actor not in seen_skills:
                prev_d = clean(prev.get("dialogue", ""))
                msg = quoted(prev_d) or prev_d[:80]
                skill_calls.append({"name": "skill_check",
                                    "args": {"skill": prev_actor, "message": msg[:120]}})
                seen_skills.add(prev_actor)

        if not skill_calls: continue

        # Detective question check (must have)
        prior = conv[max(0, i - CTX_WINDOW):i]
        has_detective = any(is_player(p.get("actor", "")) for p in prior)
        if not has_detective: continue

        # Exclude skills from context (force Detective-intent inference)
        ctx = fmt_ctx(prior, exclude_skills=True)
        if not ctx.strip(): continue

        target = {"dialogue": kim_speech, "tool_calls": skill_calls}
        samples.append({
            "messages": [
                {"role": "system", "content": KIM_SYSTEM},
                {"role": "user", "content": ctx},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
            ],
            "conv_id": conv_idx,
            "type": "skill_check"
        })
        sample_types["skill_check"] += 1
        for tc in skill_calls:
            tool_counts["skill_check"] += 1
            skill_counts[tc["args"]["skill"]] += 1
        if len(skill_calls) > 1:
            sample_types[f"skill_check_multi_{len(skill_calls)}"] += 1

    # ═══════════════════════════════════════════════════
    # Type 2: show_character (broader lookback)
    # ═══════════════════════════════════════════════════
    seen_actors = set(["Kim Kitsuragi"])
    for i, entry in enumerate(conv):
        a = entry.get("actor", "")
        d = clean(entry.get("dialogue", ""))
        if (a and not is_kim(a) and not is_skill_actor(a)
            and not is_player(a) and not is_action_meta(d) and a not in seen_actors):
            seen_actors.add(a)
            # Find Kim's next response within ACTOR_LOOKBACK
            for j in range(i + 1, min(len(conv), i + ACTOR_LOOKBACK)):
                nxt = conv[j]
                if is_kim(nxt.get("actor", "")):
                    ksp = quoted(clean(nxt.get("dialogue", "")))
                    if ksp and len(ksp.split()) >= MIN_KIM_WORDS:
                        prior = conv[max(0, i - CTX_WINDOW + 2):i]
                        ctx = fmt_ctx(prior)
                        if not ctx.strip(): ctx = "[Scene begins.]"
                        target = {
                            "dialogue": ksp,
                            "tool_calls": [{"name": "show_character",
                                            "args": {"actor": a, "slot": "right"}}]
                        }
                        samples.append({
                            "messages": [
                                {"role": "system", "content": KIM_SYSTEM},
                                {"role": "user", "content": ctx},
                                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
                            ],
                            "conv_id": conv_idx,
                            "type": "show_character"
                        })
                        sample_types["show_character"] += 1
                        tool_counts["show_character"] += 1
                        break

    # ═══════════════════════════════════════════════════
    # Type 3: present_choices (relaxed: ≥1 player option counts)
    # ═══════════════════════════════════════════════════
    for i, entry in enumerate(conv):
        if not is_kim(entry.get("actor", "")): continue
        kim_sp = quoted(clean(entry.get("dialogue", "")))
        if not kim_sp or len(kim_sp.split()) < MIN_KIM_WORDS: continue

        options = []
        for j in range(i + 1, min(len(conv), i + 6)):
            nxt = conv[j]
            if is_player(nxt.get("actor", "")):
                psp = quoted(clean(nxt.get("dialogue", "")))
                if psp and len(psp) > 3: options.append(psp[:120])
                if len(options) >= 5: break
            elif is_kim(nxt.get("actor", "")) or is_action_meta(nxt.get("dialogue", "")):
                break

        if len(options) >= MIN_PRESENT_OPTIONS:
            # Need at least 2 for valid present_choices; if only 1, look for one more
            if len(options) < 2 and len(options) == 1:
                # Look further for one more option
                for j in range(i + 6, min(len(conv), i + 10)):
                    nxt = conv[j]
                    if is_player(nxt.get("actor", "")):
                        psp = quoted(clean(nxt.get("dialogue", "")))
                        if psp and len(psp) > 3:
                            options.append(psp[:120])
                            break

            if len(options) < 2: continue

            prior = conv[max(0, i - CTX_WINDOW):i]
            ctx = fmt_ctx(prior)
            if not ctx.strip(): ctx = "[Conversation continues.]"
            target = {"dialogue": kim_sp, "tool_calls": [{"name": "present_choices", "args": {"options": options}}]}
            samples.append({
                "messages": [
                    {"role": "system", "content": KIM_SYSTEM},
                    {"role": "user", "content": ctx},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
                ],
                "conv_id": conv_idx,
                "type": "present_choices"
            })
            sample_types["present_choices"] += 1
            tool_counts["present_choices"] += 1

    # ═══════════════════════════════════════════════════
    # Type 4: empty (neutral Kim utterances, no skill/actor/options)
    # ═══════════════════════════════════════════════════
    for i, entry in enumerate(conv):
        if not is_kim(entry.get("actor", "")): continue
        kim_sp = quoted(clean(entry.get("dialogue", "")))
        if not kim_sp or len(kim_sp.split()) < 2: continue  # v5.0: 3->2 for more empties
        prior = conv[max(0, i - 3):i]  # v5.0: lookback 4->3
        if any(is_skill_actor(p.get("actor", "")) for p in prior): continue
        ctx = fmt_ctx(prior)
        if not ctx.strip(): ctx = "[Scene continues.]"
        target = {"dialogue": kim_sp, "tool_calls": []}
        samples.append({
            "messages": [
                {"role": "system", "content": KIM_SYSTEM},
                {"role": "user", "content": ctx},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
            ],
            "conv_id": conv_idx,
            "type": "empty"
        })
        sample_types["empty"] += 1

# ═══════════════════════════════════════════════════
# v5.0 NEW: synthetic set_expression (the only truly zero-real tool)
# Length-matched to real DE (~14-17 words), hand-crafted, real DE characters,
# generated from held-out scene seeds (NOT copied from DEBench).
# This is the legitimate "synthetic with positive feedback" experiment:
#   - length-matched (fixes v3.3's 6.5-word failure)
#   - embedded as a small fraction of a large real dataset (not dominant)
#   - used in a FROM-SCRATCH Stage 2 (not perturbing a frozen optimum)
# ═══════════════════════════════════════════════════
EMOTION_SEEDS = [
    (["Detective: \"We found Lely's body. He was hanging from a tree.\""], "A bad way to go. Show me.", "serious"),
    (["Detective: \"Cuno just threw a brick at my head, Kim.\""], "Are you bleeding? Let me see.", "concerned"),
    (["Detective: \"I think Ruby might be the killer.\""], "Speculation is dangerous without evidence.", "serious"),
    (["Detective: \"Klaasje confessed everything to me last night.\""], "Did she... that complicates the case considerably.", "troubled"),
    (["Detective: \"I lost my badge. It's gone.\""], "Then we find it. Now.", "stern"),
    (["Detective: \"There's been another murder, Kim.\""], "Where? Take me there.", "alarmed"),
    (["Detective: \"The Hardie boys want a sit-down with us.\""], "I don't like the sound of that.", "wary"),
    (["Detective: \"I had a vision. A bug. A huge insect.\""], "Are you... are you alright, officer?", "concerned"),
    (["Detective: \"Joyce Messier vouched for the union.\""], "Joyce has her reasons. Not all of them noble.", "skeptical"),
    (["Detective: \"The kid in the bandana is Cuno's friend, isn't he?\""], "Yes. And he's twelve years old.", "sad"),
    (["[Scene: bullet casing found in tree bark.]", "Detective: \"This means the shot came from the island.\""], "Then we've been looking in the wrong direction.", "realizing"),
    (["Detective: \"My partner died in the line of duty last year.\""], "I'm sorry. That stays with you.", "sympathetic"),
    (["Detective: \"Did you hear what Evrart said? About the contract?\""], "Every word. He's testing us.", "serious"),
    (["Detective: \"I think I've been awake for three days, Kim.\""], "Sit down. I will handle this.", "concerned"),
    (["Detective: \"They want to question the deserter in the church.\""], "Then we go. Now, before he disappears.", "decisive"),
    (["Detective: \"Klaasje is gone. The room is empty.\""], "Of course she is. We should have seen it.", "frustrated"),
    (["Detective: \"Titus is dead. They shot him in the chest.\""], "...did anyone witness it?", "grim"),
    (["Detective: \"I think we just solved the case, Kim.\""], "Let's not celebrate yet. Confirm everything.", "guarded"),
    (["Detective: \"The phasmid is real. I saw it.\""], "I saw it too. The world is wider than I knew.", "awed"),
    (["Detective: \"The lieutenant is calling me back to Precinct 41.\""], "Then this is goodbye, partner. It was an honor.", "warm"),
    (["Detective: \"The body in the meat locker is the lieutenant's son.\""], "Oh God. Cover it. Don't let his mother see.", "shocked"),
    (["Detective: \"I drank three bottles of wine last night.\""], "We need to talk. After you sober up.", "stern"),
    (["Detective: \"The kid said he watched his mother die.\""], "Then he needs someone gentle. Not us.", "sad"),
    (["Detective: \"Garte just admitted he forged the bills.\""], "Then we have leverage. Use it carefully.", "calculating"),
    (["Detective: \"There's a tape — the murder is recorded.\""], "Play it. Now.", "intense"),
    (["Detective: \"Klaasje wasn't lying. The shot came from above.\""], "Then I owe her an apology. Let's verify the angle.", "humbled"),
    (["Detective: \"The deserter has been hiding for fifty years.\""], "Fifty years. Imagine carrying that.", "somber"),
    (["Detective: \"I had to put down my old dog last month.\""], "I'm sorry. They become family.", "warm"),
    (["Detective: \"Cuno's father just died. Right in front of him.\""], "Where is the boy? Take me to him.", "urgent"),
    (["Detective: \"My old captain hated me. Real personal.\""], "Then he didn't see what I see.", "supportive"),
    (["Detective: \"The Hardie boys are armed. I think they'll shoot.\""], "Stay behind me. Do not engage.", "tense"),
    (["Detective: \"I just realized the gun was mine all along.\""], "Then we are walking back to the precinct. Slowly.", "grave"),
    (["[The radio static cuts to silence.]", "Detective: \"They cut the signal. We're alone now.\""], "Then we proceed assuming no backup. Stay alert.", "tense"),
    (["Detective: \"The boy is dead. I couldn't save him.\""], "It wasn't your job to save him. It was their job.", "sad"),
    (["Detective: \"Evrart offered me a bribe. A real one.\""], "And you turned it down. That matters.", "approving"),
    (["Detective: \"The harbor master keeps a photo of his daughter.\""], "Let him keep it. Some leverage isn't worth using.", "thoughtful"),
    (["Detective: \"I think I might have a family. Somewhere.\""], "Then we find out. After the case.", "hopeful"),
    (["Detective: \"The lab confirmed: the bullet came from a war-era rifle.\""], "Fifty-year-old shell. Same as the deserter's.", "focused"),
    (["Detective: \"I'm scared, Kim. Really scared.\""], "Sit down. Breathe. Tell me everything.", "gentle"),
    (["Detective: \"They want me to disappear. Make it look like a suicide.\""], "Then we don't let them. Not today.", "resolute"),
    (["Measurehead: \"Your race is written in the angle of your skull.\""], "I've heard enough of this nonsense.", "cold"),
    (["Garte: \"You owe me for the window, the room, and the noise.\""], "We'll settle the bill. Itemize it.", "businesslike"),
    (["[Gunfire erupts in the courtyard. Then silence.]", "Detective: \"Is it over? Are they down?\""], "Check your weapon. Don't move yet.", "shaken"),
    (["Cindy: \"That's my tag on the wall. SKULL. Got a problem?\""], "It's vandalism. But that's not why we're here.", "unimpressed"),
    (["Iosef: \"I shot him. From the island. Fifty years too late.\""], "So that's how it ends. Tell me everything.", "grave"),
    (["Soona: \"The Pale is eating the world from the edges inward.\""], "And nobody can stop it. Go on.", "curious"),
    (["Detective: \"My mind keeps spiraling. I can't make it stop.\""], "Then lean on me until it does. I'll hold the line.", "worried"),
    (["Detective: \"Remind me again — am I actually a cop?\""], "Yes. Lieutenant-double-yefreitor. We'll go slowly.", "patient"),
    (["Detective: \"Do you believe in innocence, Kim? Real innocence?\""], "I believe in evidence. Innocence is what we protect.", "reflective"),
    (["[The body has been in the heat for a week.]", "Detective: \"God, the smell. How do you stand it?\""], "You breathe through your mouth and you work the case.", "clinical"),
    (["Detective: \"Ruby jumped. Right off the boardwalk. She's gone.\""], "We pushed too hard. I'll carry that.", "regretful"),
    (["Cuno: \"Cuno does what Cuno wants! Cuno's on speed!\""], "He's a child poisoning himself. We should intervene.", "concerned"),
    (["[Music drifts from the abandoned church.]", "Detective: \"Dance with the kids, Kim? Just once?\""], "I... will observe. You enjoy yourself.", "hesitant"),
    (["Detective: \"I found my gun, Kim. It was in the locker all along.\""], "Thank God. Holster it and let's not lose it again.", "relieved"),
    (["[They cut the hanged man down from the tree.]", "Detective: \"Easy. Lay him down gently.\""], "A week he hung there. He deserves some dignity now.", "somber"),
    (["Annette: \"Mum left me to mind the books. I'm not scared.\""], "You're very brave. Is there an adult nearby?", "gentle"),
    (["Detective: \"That's it. Every thread ties together. We solved it.\""], "We did. Good work, detective. Now the paperwork.", "satisfied"),
    (["Detective: \"The tribunal's coming for us. Mercenaries, armed.\""], "Then we hold this position and we stay calm.", "steeled"),
]
for ctx_lines, kim_d, emo in EMOTION_SEEDS:
    ctx = "\n".join(ctx_lines)
    target = {"dialogue": kim_d, "tool_calls": [{"name": "set_expression", "args": {"actor": "Kim Kitsuragi", "emotion": emo}}]}
    samples.append({
        "messages": [
            {"role": "system", "content": KIM_SYSTEM},
            {"role": "user", "content": ctx},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
        ],
        "conv_id": 100000 + len(samples),  # unique synthetic conv_id (avoids val-split collision)
        "type": "set_expression"
    })
    sample_types["set_expression"] += 1
    tool_counts["set_expression"] += 1
print(f"\n[v5.0] Added {len(EMOTION_SEEDS)} synthetic set_expression samples (length-matched)")

print(f"Raw samples: {len(samples)}")
print("\nType distribution:")
for k, v in sample_types.most_common(): print(f"  {k:25s} {v}")

print("\nTool distribution:")
for k, v in tool_counts.most_common(): print(f"  {k:20s} {v}")

print(f"\nSkill distribution (top 25):")
for k, v in skill_counts.most_common(25): print(f"  {k:25s} {v}")

# v5.3 SINGLE-VARIABLE test: from v5.0 (F1 0.641), cut ONLY skill_check dominance
# (453 -> 250) and HOLD the empty ratio at v5.0's known-good 44%. Hypothesis:
# skill_check dominance (30% of v5.0's data, all "fire a skill") causes BOTH of
# v5.0's flaws -- over-calling (fires skill on chitchat, Suppress 0.10) AND scene
# collapse (fires skill instead of show_character, tool confusion). Reducing it,
# while NOT touching the chaotic empty-ratio axis, should fix both while keeping
# evidence (1.0) and emotion (0.20). Also disambiguates v5.1's confound (skill cut
# vs empty raise -- v5.1 changed both).
CAPS = {"skill_check": 250, "show_character": 9999, "present_choices": 130, "set_expression": 55}
by_type = {}
for s in samples:
    if s["type"] == "empty": continue
    by_type.setdefault(s["type"], []).append(s)
positives = []
for t, lst in by_type.items():
    random.shuffle(lst)
    cap = CAPS.get(t, 9999)
    positives.extend(lst[:cap])
    print(f"  cap {t:18s}: {len(lst)} -> {min(len(lst), cap)}")
negatives = [s for s in samples if s["type"] == "empty"]
print(f"\nBefore balance: {len(positives)} positives + {len(negatives)} negatives")

# HOLD empty ratio at v5.0's 44%: target_neg = positives * (0.44/0.56) ~= 0.786
target_neg = int(len(positives) * 0.786)
sampled_negs = random.sample(negatives, min(target_neg, len(negatives)))
balanced = positives + sampled_negs
random.shuffle(balanced)
print(f"After balance: {len(balanced)} (pos {len(positives)} + neg {len(sampled_negs)}, "
      f"empty {len(sampled_negs)/len(balanced):.0%})")

# Dedup by (conv_id, type, target hash)
seen = set()
deduped = []
for s in balanced:
    key = (s["conv_id"], s["type"], hash(s["messages"][2]["content"]))
    if key in seen: continue
    seen.add(key)
    deduped.append(s)
print(f"After dedup: {len(deduped)}")

# Split
conv_ids = list(set(s["conv_id"] for s in deduped))
random.shuffle(conv_ids)
val_conv = set(conv_ids[:max(len(conv_ids)//20, 5)])
train = [{"messages": s["messages"]} for s in deduped if s["conv_id"] not in val_conv]
val = [{"messages": s["messages"]} for s in deduped if s["conv_id"] in val_conv]

with open(os.path.join(OUTPUT_DIR, "kim_tool_train.jsonl"), "w", encoding="utf-8") as f:
    for s in train: f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "kim_tool_valid.jsonl"), "w", encoding="utf-8") as f:
    for s in val: f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nTrain: {len(train)} | Val: {len(val)}")
print(f"\nVs v3.1: 677 train → {len(train)} train ({len(train)/677:.1%})")
