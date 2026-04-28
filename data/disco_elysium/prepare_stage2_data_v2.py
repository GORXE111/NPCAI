"""
Stage 2 v2 — fix v1 issues:
1. v1 误把 [Action/Check] 当 skill_check（实际是 state check）
2. v1 只覆盖 309 条，太少

v2 strategy:
- 对所有 1,127 Kim 引语都生成 stage2 sample (with tool_calls list, possibly empty)
- skill_check 来源: Skill actor (Logic/Empathy/...) 出现在 Kim 之前 N 步
- present_choices 来源: Kim 之后紧跟 >=2 player options
- show_character 来源: 新 actor 进场
- 这样模型学到"何时调工具 / 何时不调"的完整分布
"""
import json, os, random, re
from collections import Counter

INPUT = "D:/AIproject/NPCAI/data/disco_elysium/output.json"
OUTPUT_DIR = "D:/AIproject/NPCAI/data/disco_elysium/training_stage2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KIM_SYSTEM = """You are Kim Kitsuragi from Disco Elysium. Output STRICT JSON:
{"dialogue": "<Kim's spoken line>", "tool_calls": [{"name": "...", "args": {...}}]}

Available tools:
- skill_check(skill, message): trigger a DE internal skill (Logic, Empathy, Visual Calculus, Shivers, Half Light, Inland Empire, Authority, Rhetoric, Drama, Conceptualization, Encyclopedia, Volition, Esprit de Corps, Suggestion, Endurance, Pain Threshold, Physical Instrument, Electrochemistry, Hand/Eye Coordination, Perception, Reaction Speed, Savoir Faire, Interfacing, Composure)
- present_choices(options): show 2-5 player choices
- set_expression(actor, emotion): change facial expression
- show_character(actor, slot): bring a character on screen
- hide_character(actor): remove from screen
- play_bgm(track), play_sfx(sound), set_background(location)
- narrate(text), end_scene(next)

Reply ONLY in valid JSON. No code fences, no commentary.
tool_calls may be empty array [] when no tool fires this turn."""

SKILLS = {"Logic","Encyclopedia","Rhetoric","Drama","Conceptualization","Visual Calculus",
          "Volition","Inland Empire","Empathy","Authority","Esprit de Corps","Suggestion",
          "Endurance","Pain Threshold","Physical Instrument","Electrochemistry","Shivers",
          "Half Light","Hand/Eye Coordination","Perception","Reaction Speed","Savoir Faire",
          "Interfacing","Composure"}

def is_kim(actor): return actor and "Kim" in actor and "Kitsuragi" in actor
def is_skill_actor(actor): return actor in SKILLS
def is_player(actor): return actor in ("You", "Player", "Harry")
def is_action_meta(d): return d.startswith("[Action") or d.startswith("[Condition") or d.startswith("[Variable")

def clean_dialogue(text):
    text = re.sub(r'wavs/.*?\.wav\|', '', text)
    return text.strip()

def extract_quoted_speech(dialogue):
    if not dialogue: return None
    quotes = re.findall(r'"([^"]+)"', dialogue)
    quotes = [q.strip() for q in quotes if q.strip()]
    speech = " ".join(quotes) if quotes else None
    if speech:
        speech = re.sub(r'\[(Action|Condition|Variable)[^\]]*\]', '', speech).strip()
    return speech if speech and len(speech) >= 4 else None

def format_context(entries):
    lines = []
    for e in entries:
        actor = e.get("actor", "Narrator")
        d = clean_dialogue(e.get("dialogue", ""))
        if not d: continue
        if is_action_meta(d):
            lines.append(f"[scene: {d[:120]}]")
        elif is_skill_actor(actor):
            speech = extract_quoted_speech(d) or d
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
context_window = 6
seen_actors_per_conv = {}  # track who's been on screen

for conv_idx, conv in enumerate(conversations):
    seen = set(["Kim Kitsuragi"])  # Kim is always present
    for i, entry in enumerate(conv):
        if not is_kim(entry.get("actor", "")): continue
        kim_line = clean_dialogue(entry.get("dialogue", ""))
        kim_speech = extract_quoted_speech(kim_line)
        if not kim_speech or len(kim_speech.split()) < 2: continue

        tool_calls = []

        # 1. Skill actor speaks in last 2 entries → Kim is responding to a skill check
        for j in range(max(0, i - 3), i):
            prev = conv[j]
            prev_actor = prev.get("actor", "")
            if is_skill_actor(prev_actor):
                prev_d = clean_dialogue(prev.get("dialogue", ""))
                msg = extract_quoted_speech(prev_d) or prev_d[:80]
                tc = {"name": "skill_check", "args": {"skill": prev_actor, "message": msg[:120]}}
                if tc not in tool_calls:
                    tool_calls.append(tc)

        # 2. New actor enters in last 4 entries (not Kim, not skill, not player) → show_character
        for j in range(max(0, i - 4), i):
            prev = conv[j]
            prev_actor = prev.get("actor", "")
            if (prev_actor and not is_kim(prev_actor) and not is_skill_actor(prev_actor)
                and not is_player(prev_actor) and prev_actor not in seen
                and not is_action_meta(prev.get("dialogue", ""))):
                tool_calls.append({"name": "show_character", "args": {"actor": prev_actor, "slot": "right"}})
                seen.add(prev_actor)
                break  # one per turn

        # 3. Player options after Kim → present_choices
        options = []
        for j in range(i + 1, min(len(conv), i + 5)):
            nxt = conv[j]
            nxt_actor = nxt.get("actor", "")
            nxt_d = clean_dialogue(nxt.get("dialogue", ""))
            if is_player(nxt_actor):
                speech = extract_quoted_speech(nxt_d) or nxt_d
                if speech and len(speech) > 2:
                    options.append(speech[:120])
                if len(options) >= 4: break
            elif is_kim(nxt_actor) or is_action_meta(nxt_d):
                break
        if len(options) >= 2:
            tool_calls.append({"name": "present_choices", "args": {"options": options}})

        # Build context
        prior = conv[max(0, i - context_window):i]
        context = format_context(prior)
        if not context.strip():
            context = "[Scene begins. Detective approaches Kim.]"

        target = {"dialogue": kim_speech, "tool_calls": tool_calls}
        target_str = json.dumps(target, ensure_ascii=False)

        samples.append({
            "messages": [
                {"role": "system", "content": KIM_SYSTEM},
                {"role": "user", "content": context},
                {"role": "assistant", "content": target_str}
            ],
            "conv_id": conv_idx,
            "n_tools": len(tool_calls),
        })

print(f"Total Stage 2 samples: {len(samples)}")
print(f"  Samples WITH tool_calls: {sum(1 for s in samples if s['n_tools'] > 0)}")
print(f"  Samples with EMPTY tool_calls: {sum(1 for s in samples if s['n_tools'] == 0)}")

tool_dist = Counter()
skill_dist = Counter()
for s in samples:
    target = json.loads(s["messages"][2]["content"])
    for tc in target["tool_calls"]:
        tool_dist[tc["name"]] += 1
        if tc["name"] == "skill_check":
            skill_dist[tc["args"].get("skill","?")] += 1

print("\nTool distribution:")
for t, c in tool_dist.most_common():
    print(f"  {t:25s} {c}")
print("\nSkill distribution (within skill_check):")
for s, c in skill_dist.most_common(15):
    print(f"  {s:25s} {c}")

# Split
random.seed(42)
conv_ids = list(set(s["conv_id"] for s in samples))
random.shuffle(conv_ids)
val_conv = set(conv_ids[:max(len(conv_ids)//20, 5)])
train = [{"messages": s["messages"]} for s in samples if s["conv_id"] not in val_conv]
val = [{"messages": s["messages"]} for s in samples if s["conv_id"] in val_conv]

with open(os.path.join(OUTPUT_DIR, "kim_tool_train.jsonl"), "w", encoding="utf-8") as f:
    for s in train: f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(os.path.join(OUTPUT_DIR, "kim_tool_valid.jsonl"), "w", encoding="utf-8") as f:
    for s in val: f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nTrain: {len(train)} | Val: {len(val)}")

# Show diverse samples
print("\n" + "=" * 60)
print("4 RANDOM SAMPLES (diversity check)")
print("=" * 60)
# Pick 2 with tools, 2 without
with_tools = [s for s in train if s["messages"][2]["content"].count('"tool_calls": []') == 0]
no_tools = [s for s in train if '"tool_calls": []' in s["messages"][2]["content"]]
picks = random.sample(with_tools, 2) + random.sample(no_tools, min(2, len(no_tools)))
for s in picks:
    print()
    print("CTX:", s["messages"][1]["content"][:200].replace("\n", " | "))
    print("TGT:", s["messages"][2]["content"][:400])
