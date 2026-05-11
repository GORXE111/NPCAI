"""
Stage 2 v3 — Mine skill actor utterances as ground-truth tool calls.

Key insight: DE corpus has thousands of "Skill: <internal monologue>" lines.
These are the EXACT moments when a skill check fires in the game.

Strategy:
For every Skill actor line, find the surrounding (Detective question, Kim response)
context and create a training sample where:
- INPUT: Detective's question + scene state (Skill actor speech REMOVED)
- OUTPUT: Kim's response + skill_check(skill=<that skill>)

This teaches Kim to PROACTIVELY fire the right skill_check when Detective asks
the kind of question that would trigger that skill in DE.

Bonus: generate present_choices, show_character samples similarly from corpus signal.
"""
import json, os, random, re
from collections import Counter

INPUT = "D:/AIproject/NPCAI/data/disco_elysium/output.json"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3"
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

def is_kim(actor): return actor and "Kim" in actor and "Kitsuragi" in actor
def is_skill_actor(actor): return actor in SKILLS_ALL
def is_player(actor): return actor in ("You", "Player", "Harry")
def is_action_meta(d): return d.startswith("[Action") or d.startswith("[Condition") or d.startswith("[Variable")

def clean_dialogue(text):
    return re.sub(r'wavs/.*?\.wav\|', '', text or "").strip()

def extract_quoted_speech(dialogue):
    if not dialogue: return None
    quotes = re.findall(r'"([^"]+)"', dialogue)
    quotes = [q.strip() for q in quotes if q.strip()]
    speech = " ".join(quotes) if quotes else None
    if speech:
        speech = re.sub(r'\[(Action|Condition|Variable)[^\]]*\]', '', speech).strip()
    if not speech or len(speech) < 4: return None
    # Reject pure state-marker garbage
    if re.match(r'^[a-z_]+\.[a-z_]+\s*(TASK\.|Variable\[)', speech): return None
    if re.match(r'^auto\.', speech): return None
    if re.match(r'^TASK\.', speech): return None
    if speech.count("_") > speech.count(" ") + 2: return None  # snake_case dominated
    return speech

def get_speech_or_narrative(dialogue):
    """Get either quoted speech or the narrative text itself (for skill descriptions)."""
    if not dialogue: return None
    spe = extract_quoted_speech(dialogue)
    if spe: return spe
    # No quotes — just use the narrative
    text = re.sub(r'\[(Action|Condition|Variable)[^\]]*\]', '', dialogue).strip()
    return text if text and len(text) >= 4 else None

def format_context(entries, exclude_skills=False):
    lines = []
    for e in entries:
        actor = e.get("actor", "Narrator")
        d = clean_dialogue(e.get("dialogue", ""))
        if not d: continue
        if is_action_meta(d):
            lines.append(f"[scene: {d[:120]}]")
        elif is_skill_actor(actor):
            if exclude_skills: continue  # Drop the skill actor — pretend it didn't speak
            speech = get_speech_or_narrative(d) or d
            lines.append(f"({actor} — internal): {speech[:200]}")
        elif is_player(actor):
            speech = extract_quoted_speech(d) or d
            lines.append(f"Detective: {speech[:200]}")
        else:
            speech = extract_quoted_speech(d) or d
            lines.append(f"{actor}: {speech[:200]}")
    return "\n".join(lines)


with open(INPUT, encoding="utf-8") as f:
    conversations = json.load(f)["conversations"]

samples = []
tool_counts = Counter()
skill_counts = Counter()
sample_types = Counter()

for conv_idx, conv in enumerate(conversations):
    # =====================================================
    # Type 1: Skill actor → Kim response (skill_check sample)
    # =====================================================
    seen_actors = set(["Kim Kitsuragi"])
    for i, entry in enumerate(conv):
        actor = entry.get("actor", "")
        if not is_skill_actor(actor): continue
        skill = actor

        # Find Kim's next response after this skill actor
        kim_response = None
        kim_idx = None
        for j in range(i + 1, min(len(conv), i + 4)):
            nxt = conv[j]
            if is_kim(nxt.get("actor", "")):
                ksp = extract_quoted_speech(clean_dialogue(nxt.get("dialogue", "")))
                if ksp and len(ksp.split()) >= 2:
                    kim_response = ksp
                    kim_idx = j
                    break

        if not kim_response: continue

        # Build context: prior 6 entries BEFORE the skill, EXCLUDING the skill itself
        # This simulates: "Detective asks, Kim responds with skill_check"
        prior = conv[max(0, i - 6):i]
        # CRITICAL: exclude skill actors from context so model learns to fire skill_check
        # from Detective's question alone, not by mimicking the skill actor's prior speech.
        context = format_context(prior, exclude_skills=True)
        # Add Detective's most recent question if available
        detective_q = None
        for p in reversed(prior):
            if is_player(p.get("actor", "")):
                d_sp = extract_quoted_speech(clean_dialogue(p.get("dialogue", "")))
                if d_sp:
                    detective_q = d_sp
                    break

        if not detective_q:
            # No Detective in context — skip (we want Detective-initiated turns)
            continue

        # The skill actor's content as message hint
        skill_d = clean_dialogue(entry.get("dialogue", ""))
        skill_msg = get_speech_or_narrative(skill_d) or "internal observation"

        target = {
            "dialogue": kim_response,
            "tool_calls": [{"name": "skill_check",
                            "args": {"skill": skill, "message": skill_msg[:120]}}]
        }
        samples.append({
            "messages": [
                {"role": "system", "content": KIM_SYSTEM},
                {"role": "user", "content": context},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
            ],
            "conv_id": conv_idx,
            "type": "skill_check",
        })
        sample_types["skill_check"] += 1
        skill_counts[skill] += 1
        tool_counts["skill_check"] += 1

    # =====================================================
    # Type 2: New actor appearance → Kim show_character
    # =====================================================
    seen = set(["Kim Kitsuragi"])
    for i, entry in enumerate(conv):
        actor = entry.get("actor", "")
        d = clean_dialogue(entry.get("dialogue", ""))
        if (actor and not is_kim(actor) and not is_skill_actor(actor)
            and not is_player(actor) and not is_action_meta(d) and actor not in seen):
            seen.add(actor)
            # Find Kim's next response
            for j in range(i + 1, min(len(conv), i + 4)):
                nxt = conv[j]
                if is_kim(nxt.get("actor", "")):
                    ksp = extract_quoted_speech(clean_dialogue(nxt.get("dialogue", "")))
                    if ksp and len(ksp.split()) >= 2:
                        prior = conv[max(0, i - 5):i]
                        context = format_context(prior)
                        if not context.strip():
                            context = "[Scene begins.]"
                        target = {
                            "dialogue": ksp,
                            "tool_calls": [{"name": "show_character",
                                            "args": {"actor": actor, "slot": "right"}}]
                        }
                        samples.append({
                            "messages": [
                                {"role": "system", "content": KIM_SYSTEM},
                                {"role": "user", "content": context},
                                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
                            ],
                            "conv_id": conv_idx,
                            "type": "show_character",
                        })
                        sample_types["show_character"] += 1
                        tool_counts["show_character"] += 1
                        break

    # =====================================================
    # Type 3: ≥2 Player options after Kim → present_choices
    # =====================================================
    for i, entry in enumerate(conv):
        if not is_kim(entry.get("actor", "")): continue
        kim_sp = extract_quoted_speech(clean_dialogue(entry.get("dialogue", "")))
        if not kim_sp: continue
        # Look ahead for player options
        options = []
        for j in range(i + 1, min(len(conv), i + 5)):
            nxt = conv[j]
            if is_player(nxt.get("actor", "")):
                p_sp = extract_quoted_speech(clean_dialogue(nxt.get("dialogue", "")))
                if p_sp and len(p_sp) > 3:
                    options.append(p_sp[:120])
                if len(options) >= 4: break
            elif is_kim(nxt.get("actor", "")) or is_action_meta(nxt.get("dialogue", "")):
                break
        if len(options) >= 2:
            prior = conv[max(0, i - 6):i]
            context = format_context(prior)
            if not context.strip():
                context = "[Conversation continues.]"
            target = {
                "dialogue": kim_sp,
                "tool_calls": [{"name": "present_choices",
                                "args": {"options": options}}]
            }
            samples.append({
                "messages": [
                    {"role": "system", "content": KIM_SYSTEM},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
                ],
                "conv_id": conv_idx,
                "type": "present_choices",
            })
            sample_types["present_choices"] += 1
            tool_counts["present_choices"] += 1


    # =====================================================
    # Type 4: Neutral Kim responses → empty tool_calls
    # (only when no other tool fired, but Kim still spoke)
    # =====================================================
    for i, entry in enumerate(conv):
        if not is_kim(entry.get("actor", "")): continue
        kim_sp = extract_quoted_speech(clean_dialogue(entry.get("dialogue", "")))
        if not kim_sp or len(kim_sp.split()) < 3: continue
        # Check no skill actor / new actor / options around — pure small talk
        prior = conv[max(0, i - 4):i]
        has_skill = any(is_skill_actor(p.get("actor","")) for p in prior)
        if has_skill: continue
        # Make negative sample
        context = format_context(prior)
        if not context.strip():
            context = "[Scene continues.]"
        target = {"dialogue": kim_sp, "tool_calls": []}
        samples.append({
            "messages": [
                {"role": "system", "content": KIM_SYSTEM},
                {"role": "user", "content": context},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
            ],
            "conv_id": conv_idx,
            "type": "empty",
        })
        sample_types["empty"] += 1

print(f"Total raw samples: {len(samples)}")
print("\nSample types:")
for k, v in sample_types.most_common(): print(f"  {k:18s} {v}")

print("\nTool distribution:")
for k, v in tool_counts.most_common(): print(f"  {k:20s} {v}")

print("\nSkill distribution (top 24):")
for k, v in skill_counts.most_common(24): print(f"  {k:25s} {v}")

# Balance: keep all positive samples, downsample negative
positives = [s for s in samples if s["type"] != "empty"]
negatives = [s for s in samples if s["type"] == "empty"]
# Target: 30% negative (so model still learns when NOT to call tools)
target_neg = int(len(positives) * 0.43)
sampled_negs = random.sample(negatives, min(target_neg, len(negatives)))
balanced = positives + sampled_negs
random.shuffle(balanced)

print(f"\nAfter balancing:")
print(f"  Positives kept: {len(positives)}")
print(f"  Negatives sampled: {len(sampled_negs)} (target ~43% of total)")
print(f"  Total: {len(balanced)}")

# Re-check distribution after balancing
final_types = Counter(s["type"] for s in balanced)
print("\nFinal sample type distribution:")
for k, v in final_types.most_common():
    print(f"  {k:18s} {v} ({v/len(balanced):.0%})")

# Dedup by (conv_id, type, target hash)
seen_keys = set()
deduped = []
for s in balanced:
    target_hash = hash(s["messages"][2]["content"])
    key = (s["conv_id"], s["type"], target_hash)
    if key in seen_keys: continue
    seen_keys.add(key)
    deduped.append(s)
print(f"\nAfter dedup: {len(deduped)}")

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

# Show samples per type
print("\n" + "=" * 60)
print("SAMPLE OUTPUTS (one per type)")
print("=" * 60)
def _filter_pool(tt):
    pool = []
    for s in train:
        tc = json.loads(s["messages"][2]["content"]).get("tool_calls", [])
        if tt == "empty":
            if not tc: pool.append(s)
        else:
            if tc and tc[0]["name"] == tt: pool.append(s)
    return pool

for ttype in ["skill_check", "show_character", "present_choices", "empty"]:
    pool = _filter_pool(ttype)
    if not pool: continue
    s = random.choice(pool)
    print(f"\n--- [{ttype}] ---")
    print(f"CTX: {s['messages'][1]['content'][:250].replace(chr(10), ' | ')}")
    print(f"TGT: {s['messages'][2]['content'][:350]}")
