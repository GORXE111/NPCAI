"""
v2: Extract ONLY Kim's quoted speech as the assistant target.
Strip third-person narration, [Action/Check] tags, and stage directions
from training output (but keep them in prior context for scene awareness).
"""
import json, os, random, re

INPUT = "D:/AIproject/NPCAI/data/disco_elysium/output.json"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KIM_SYSTEM = """You are Kim Kitsuragi, a 43-year-old lieutenant from RCM Precinct 41 in Revachol, partnered with a deeply troubled detective on a murder investigation.

Speech style: clinical, dry, morally anchored — measured pauses, careful word choice, quiet wit. You frequently say things like "Let me make a note of that", "Detective, please", "That is... an interesting approach". You protect your partner from his worst impulses without being preachy.

Reply ONLY with what Kim Kitsuragi *says* (his spoken words). Do NOT include:
- third-person narration like "He nods" or "the lieutenant wipes his brow"
- scene tags like [Action/Check: ...] or stage directions
- character name prefixes like "Kim Kitsuragi:"

Just the spoken line, in Kim's voice. 1-3 sentences."""

def is_kim(actor):
    return actor and "Kim" in actor and "Kitsuragi" in actor

def is_action_meta(entry):
    d = entry.get("dialogue", "").strip()
    return d.startswith("[Action") or d.startswith("[Condition") or d.startswith("[Variable")

def is_skill(actor):
    SKILLS = {"Logic","Encyclopedia","Rhetoric","Drama","Conceptualization","Visual Calculus",
              "Volition","Inland Empire","Empathy","Authority","Esprit de Corps","Suggestion",
              "Endurance","Pain Threshold","Physical Instrument","Electrochemistry","Shivers",
              "Half Light","Hand/Eye Coordination","Perception","Reaction Speed","Savoir Faire",
              "Interfacing","Composure"}
    return actor in SKILLS

def is_player(actor):
    return actor in ("You", "Player", "Harry")

def clean_dialogue(text):
    text = re.sub(r'wavs/.*?\.wav\|', '', text)
    return text.strip()

def extract_quoted_speech(dialogue):
    """Extract only the quoted-speech portion of a DE dialogue string.

    DE format: "actual quoted speech." The lieutenant nods. Maybe more "speech."
    We want to concatenate all quoted speech, dropping the narration in between.
    """
    if not dialogue: return None
    # Find all "..." content (using non-greedy, allowing escaped quotes)
    quotes = re.findall(r'"([^"]+)"', dialogue)
    # Filter out empty/whitespace quotes
    quotes = [q.strip() for q in quotes if q.strip()]
    if not quotes:
        return None
    # Concatenate with space
    speech = " ".join(quotes)
    # Strip stage tags inside quotes (rare but possible)
    speech = re.sub(r'\[(Action|Condition|Variable)[^\]]*\]', '', speech)
    speech = speech.strip()
    if len(speech) < 4: return None
    return speech

def format_context(entries):
    """Format prior turns. Keep [Action/Check] tags as scene metadata for context."""
    lines = []
    for e in entries:
        actor = e.get("actor", "Narrator")
        dialogue = clean_dialogue(e.get("dialogue", ""))
        if not dialogue: continue
        if is_action_meta(e):
            lines.append(f"[scene: {dialogue}]")
        elif is_skill(actor):
            lines.append(f"({actor} — internal): {dialogue}")
        elif is_player(actor):
            lines.append(f"Detective: {dialogue}")
        else:
            lines.append(f"{actor}: {dialogue}")
    return "\n".join(lines)


with open(INPUT, encoding="utf-8") as f:
    conversations = json.load(f)["conversations"]

samples = []
filtered_no_speech = 0
context_window = 6

for conv_idx, conv in enumerate(conversations):
    for i, entry in enumerate(conv):
        if not is_kim(entry.get("actor", "")): continue

        kim_line_raw = clean_dialogue(entry.get("dialogue", ""))
        kim_speech = extract_quoted_speech(kim_line_raw)

        if not kim_speech:
            filtered_no_speech += 1
            continue
        if len(kim_speech.split()) < 2:
            filtered_no_speech += 1
            continue

        prior = conv[max(0, i - context_window):i]
        context = format_context(prior)
        if not context.strip():
            context = "[Scene begins. Detective approaches Kim.]"

        samples.append({
            "messages": [
                {"role": "system", "content": KIM_SYSTEM},
                {"role": "user", "content": context},
                {"role": "assistant", "content": kim_speech}
            ],
            "conv_id": conv_idx,
        })

print(f"Total Kim turns processed: {len(samples) + filtered_no_speech}")
print(f"  Kept (had quoted speech): {len(samples)}")
print(f"  Skipped (pure narration / no speech): {filtered_no_speech}")

# Length stats
lengths = [len(s["messages"][2]["content"]) for s in samples]
print(f"Speech length — min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)/len(lengths):.0f}")

# Filter very long (>500 chars likely complex multi-quote with embedded narration)
samples = [s for s in samples if len(s["messages"][2]["content"]) < 500]
print(f"After length filter (<500 chars): {len(samples)}")

# Split by conversation
random.seed(42)
conv_ids = list(set(s["conv_id"] for s in samples))
random.shuffle(conv_ids)
val_conv = set(conv_ids[:max(len(conv_ids)//20, 5)])
train = [{"messages": s["messages"]} for s in samples if s["conv_id"] not in val_conv]
val = [{"messages": s["messages"]} for s in samples if s["conv_id"] in val_conv]

with open(os.path.join(OUTPUT_DIR, "kim_train.jsonl"), "w", encoding="utf-8") as f:
    for s in train:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "kim_valid.jsonl"), "w", encoding="utf-8") as f:
    for s in val:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nTrain: {len(train)} | Val: {len(val)}")

# Print 3 random samples to verify
print("\n" + "=" * 60)
print("3 RANDOM SAMPLES (to verify cleaning):")
print("=" * 60)
for s in random.sample(train, min(3, len(train))):
    print()
    print("CTX:", s["messages"][1]["content"][:200].replace("\n", " | "))
    print("KIM:", s["messages"][2]["content"])
