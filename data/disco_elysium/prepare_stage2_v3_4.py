"""
Stage 2 v3.4 — FIX v3.3 root cause: synthetic samples too short, wrong style.

DIAGNOSIS:
- v3.1 (F1 0.639): real Kim responses avg 13.9 words
- v3.3 (F1 0.102): added 55 synthetic samples avg 6.5 words (47% length)
- Model learned: short clean prompt → brief reply, minimal tools
- DEBench prompts are short+clean → model under-calls on them

FIX (single-variable):
1. Mine MORE real present_choices from DE corpus (lower threshold)
2. Generate emotion samples that MATCH real DE Kim style:
   - 20-40 word responses (match real length)
   - Use Kim's signature phrases ("Let me make a note", "Detective, please")
   - Include scene context
3. Lock random seed
4. Only ADD 20-30 high-quality samples (not 55) — minimum perturbation
"""
import json, os, random, re
from collections import Counter

INPUT = "D:/AIproject/NPCAI/data/disco_elysium/output.json"
PRIOR_V3_1 = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_1/kim_tool_train.jsonl"
PRIOR_V3_1_VAL = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_1/kim_tool_valid.jsonl"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2_v3_4"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

KIM_SYSTEM = """You are Kim Kitsuragi from Disco Elysium. Output STRICT JSON:
{"dialogue": "<Kim's spoken line>", "tool_calls": [{"name": "...", "args": {...}}]}

Available tools:
- skill_check(skill, message): trigger a DE internal skill (Logic, Empathy, Visual Calculus, Shivers, Half Light, Inland Empire, Authority, Rhetoric, Drama, Conceptualization, Encyclopedia, Volition, Esprit de Corps, Suggestion, Endurance, Pain Threshold, Physical Instrument, Electrochemistry, Hand/Eye Coordination, Perception, Reaction Speed, Savoir Faire, Interfacing, Composure)
- present_choices(options): show 2-5 player choices
- set_expression(actor, emotion): change facial expression (neutral, amused, worried, stern, sad, surprised, disapproving)
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
def is_player(a): return a in ("You", "Player", "Harry")
def is_skill_actor(a): return a in SKILLS_ALL
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


# ════════════════════════════════════════════════════
# Load v3.1 baseline
# ════════════════════════════════════════════════════
def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
v3_1_train = load_jsonl(PRIOR_V3_1)
v3_1_val = load_jsonl(PRIOR_V3_1_VAL)
print(f"v3.1 baseline: {len(v3_1_train)} train + {len(v3_1_val)} val")

# Existing v3.1 samples — record their target hashes to avoid dupes later
v3_1_hashes = set(hash(s["messages"][2]["content"]) for s in v3_1_train)

# ════════════════════════════════════════════════════
# Mine MORE present_choices from DE corpus
# Relaxed criterion: Kim asks a question OR multiple player options follow
# ════════════════════════════════════════════════════
with open(INPUT, encoding="utf-8") as f:
    conversations = json.load(f)["conversations"]

mined_choices = []
for conv_idx, conv in enumerate(conversations):
    for i, entry in enumerate(conv):
        if not is_kim(entry.get("actor", "")): continue
        kim_speech = quoted(clean(entry.get("dialogue", "")))
        if not kim_speech or len(kim_speech.split()) < 5: continue  # min 5 words (longer than v3.1's 2)

        # Look ahead for player options
        options = []
        for j in range(i + 1, min(len(conv), i + 6)):
            nxt = conv[j]
            if is_player(nxt.get("actor", "")):
                sp = quoted(clean(nxt.get("dialogue", "")))
                if sp and len(sp) > 5: options.append(sp[:120])
                if len(options) >= 5: break
            elif is_kim(nxt.get("actor", "")) or is_action_meta(nxt.get("dialogue", "")):
                break
        # Relaxed: ≥1 option OR Kim asks a question (ends with ?)
        if len(options) >= 1 or kim_speech.rstrip().endswith("?"):
            # For Kim-asks-question case with no options, fabricate plausible options from context
            if len(options) < 2:
                # Use generic options informed by Kim's question
                generic = [
                    ["Yes.", "No.", "Tell me more."],
                    ["Agreed.", "Let me think.", "Push back."],
                    ["Now.", "Later.", "Never."],
                ]
                options = options + random.choice(generic)
                options = options[:3]

            if len(options) >= 2:
                prior = conv[max(0, i - 6):i]
                ctx = fmt_ctx(prior)
                if not ctx.strip(): ctx = "[Conversation continues.]"
                target = {"dialogue": kim_speech, "tool_calls": [{"name": "present_choices", "args": {"options": options}}]}
                target_str = json.dumps(target, ensure_ascii=False)
                if hash(target_str) in v3_1_hashes: continue  # already in v3.1
                mined_choices.append({
                    "messages": [
                        {"role": "system", "content": KIM_SYSTEM},
                        {"role": "user", "content": ctx},
                        {"role": "assistant", "content": target_str}
                    ]
                })

print(f"\nMined present_choices candidates: {len(mined_choices)}")
# Limit to 30 (minimum perturbation principle)
mined_choices_kept = random.sample(mined_choices, min(30, len(mined_choices)))
print(f"  Keeping: {len(mined_choices_kept)}")

# ════════════════════════════════════════════════════
# High-quality LONG emotion samples — match real DE style
# Use Kim's actual signature phrases, 20-40 word responses, scene context
# ════════════════════════════════════════════════════
LONG_EMOTION_SAMPLES = [
    # (full context with scene, Kim long response, emotion)
    ("[At the abandoned mill, light fading.] Detective: \"Kim, the suspect just escaped through the back. He's gone.\"",
     "Let me make a note of that, Detective. Suspect departed through the rear exit. We'll need to widen the perimeter and call for assistance from Precinct 41 — they should still have a unit on standby.",
     "stern"),
    ("[At the morgue.] Detective: \"We were wrong about him. He was innocent the whole time.\"",
     "Then we owe him an apology. And a formal correction to the record. The mistake is on us — I'll draft the memo myself once we're back at the station.",
     "sad"),
    ("[Outside the precinct, morning.] Detective: \"They're giving us a commendation, Kim. For closing the case.\"",
     "That's gratifying to hear. Though honestly, the work itself is the satisfaction — the paperwork just acknowledges it. We should both attend the ceremony, of course.",
     "amused"),
    ("[Tense moment, suspect with a weapon.] Detective: \"Kim, he has a gun. What do we do?\"",
     "Stay back, Detective. Stay back. Lower your weapon, sir — there's no need for this to escalate. Just put it down, slowly, and we can talk this through.",
     "worried"),
    ("[Reading a letter from headquarters.] Detective: \"Kim — your transfer came through. You're going back to Precinct 41 tomorrow.\"",
     "...I see. Well, this is unexpected timing. I'd appreciated working with you, Detective — more than I let on. Let's make sure this last day is in order before I go.",
     "surprised"),
    ("[In the suspect's apartment.] Detective: \"Take a look at this — written confession, signed and dated.\"",
     "That changes things considerably. Let me make a note of that — signed confession, present location, date of authorship. We'll need to authenticate the signature, but on its face this is significant.",
     "surprised"),
    ("[Witness sobbing.] Detective: \"Her testimony was a lie. The whole interview was a setup.\"",
     "I see. That's deeply unfortunate, but useful to know. We'll need to revisit every assertion she made and verify independently. Don't confront her about it yet — we lose the surveillance value.",
     "disapproving"),
    ("[Receiving a phone call.] Detective: \"Kim — your old partner. He's in the hospital. Critical.\"",
     "...When did this happen. Tell me. Which hospital. I need to make a call, Detective — give me a moment, please. I'll be back shortly.",
     "worried"),
    ("[The suspect smirks at Kim.] Suspect: \"Lieutenant, your sister was a real piece of work, wasn't she?\"\nDetective: \"...\"",
     "We are not engaging on personal terms. Stand down, sir. Any further provocations will be noted in the formal record. Detective, let's continue with the questions we came to ask.",
     "stern"),
    ("[Discovery of an old case file.] Detective: \"This is the woman from twenty years ago. Same MO.\"",
     "Then we may be dealing with a serial pattern across decades. That's significant, Detective. Let me make a note of that — pattern recognition, cold case linkage, requires cross-precinct coordination.",
     "stern"),
    ("[Crowded square, body in the center.] Detective: \"There's been another one. Same handprint.\"",
     "Another one. Same signature mark. That's a pattern now, Detective — not a coincidence. We need to brief the captain immediately and request additional resources for what may become a major case.",
     "stern"),
    ("[Kim returns to find Detective drunk.] Detective: \"Heyyy Kim. I had a few. Or several.\"",
     "Detective, please. We have an active investigation. I'd ask that you not pursue any further drinks tonight — we have witness interviews scheduled at dawn and you need to be presentable.",
     "disapproving"),
    ("[Witness reveals key fact.] Detective: \"He just told us the murder weapon was hidden in the well.\"",
     "Excellent. That is highly actionable information. Let me make a note of that — well location, possible disposal site, requires immediate retrieval before evidence degrades further.",
     "amused"),
    ("[Quiet moment in the precinct break room.] Detective: \"Are we still good, Kim? After what I said yesterday?\"",
     "We're professionals, Detective. What happens between us doesn't impede the work. I appreciate the question — but please don't worry about it. We have a case to close.",
     "neutral"),
    ("[Late night, alone with files.] Detective: \"I think we've been chasing the wrong man this entire week.\"",
     "Then we need to retrace every step methodically. Let's go back to the original crime scene tomorrow with fresh eyes. I'd rather lose a week than convict the wrong person, Detective.",
     "worried"),
    ("[Reading a chilling note.] Detective: \"Kim, look — the killer left a message. For you specifically.\"",
     "For me, by name? Let me see it. Don't touch anything else on the paper — fingerprints. This is concerning, Detective, but it's also useful: it means he knows we're close.",
     "stern"),
    ("[Family at the morgue identifies body.] Detective: \"It's their son. They wanted to know everything.\"",
     "I'll speak with them privately. They deserve clarity, even if not all the details. Detective, please call the chaplain — this is exactly the situation we have a chaplain for.",
     "sad"),
    ("[Captain calls them in.] Detective: \"We're being pulled off the case. Effective immediately.\"",
     "On what grounds? That doesn't follow standard protocol mid-investigation. Detective, I'd like to know who's reassigning us and why — there should be a written directive somewhere.",
     "surprised"),
    ("[After a successful arrest.] Detective: \"He confessed before we even started questioning.\"",
     "Then we have the case sewn up. Let me make a note of that — voluntary confession in front of two officers, prior to formal interrogation. That should hold up at trial.",
     "amused"),
    ("[Quiet evening.] Detective: \"I miss my partner from before. From before everything went wrong.\"",
     "...I understand. We don't choose what shapes us, only what we do after. The work continues, Detective — that's what's available to us now. Let's get some rest.",
     "sad"),
]

emotion_samples = []
for ctx, kim, emo in LONG_EMOTION_SAMPLES[:20]:
    target = {"dialogue": kim, "tool_calls": [{"name": "set_expression", "args": {"actor": "Kim Kitsuragi", "emotion": emo}}]}
    emotion_samples.append({
        "messages": [
            {"role": "system", "content": KIM_SYSTEM},
            {"role": "user", "content": ctx},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
        ]
    })

print(f"\nHigh-quality emotion samples: {len(emotion_samples)} (avg {sum(len(json.loads(s['messages'][2]['content'])['dialogue'].split()) for s in emotion_samples)/len(emotion_samples):.1f} words)")

# ════════════════════════════════════════════════════
# Combine
# ════════════════════════════════════════════════════
added = mined_choices_kept + emotion_samples
print(f"\nTotal additions: {len(added)} ({len(mined_choices_kept)} present_choices + {len(emotion_samples)} emotion)")
print(f"vs v3.3 which added 55 (almost 2× as many) — minimum perturbation principle")

# Length verification
added_lengths = [len(json.loads(s['messages'][2]['content'])['dialogue'].split()) for s in added]
import statistics
print(f"\nAdded sample length: avg {statistics.mean(added_lengths):.1f} words, median {statistics.median(added_lengths):.0f}")
print(f"vs v3.1 real positives: avg 13.9 words  — should now MATCH")

combined_train = list(v3_1_train) + added
random.shuffle(combined_train)

# Save
with open(os.path.join(OUTPUT_DIR, "kim_tool_train.jsonl"), "w", encoding="utf-8") as f:
    for s in combined_train: f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "kim_tool_valid.jsonl"), "w", encoding="utf-8") as f:
    for s in v3_1_val: f.write(json.dumps(s, ensure_ascii=False) + "\n")

# Final distribution
tool_counts = Counter()
for s in combined_train:
    tc = json.loads(s["messages"][2]["content"]).get("tool_calls", [])
    if not tc: tool_counts["[empty]"] += 1
    for t in tc: tool_counts[t["name"]] += 1

print(f"\nFinal v3.4: train {len(combined_train)} | val {len(v3_1_val)}")
print("Tool distribution:")
for k, v in tool_counts.most_common():
    print(f"  {k:20s} {v} ({v/len(combined_train):.1%})")
