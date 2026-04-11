"""
Benchmark Dimension 4: Emotion Coherence
Tests whether NPC emotional responses are coherent across multiple turns.
"""
import json, os, time, sys

PROJECT_DIR = os.path.expanduser("~/npcllm")
CMD_FILE = os.path.join(PROJECT_DIR, ".test_command")
RESULT_FILE = os.path.join(PROJECT_DIR, ".test_result")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "benchmark_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Valid emotion transitions (from -> to)
# True = valid transition, False = invalid
VALID_TRANSITIONS = {
    "neutral":  {"neutral": True, "happy": True, "angry": True, "sad": True, "fearful": True},
    "happy":    {"neutral": True, "happy": True, "angry": True, "sad": True, "fearful": True},
    "angry":    {"neutral": True, "happy": False, "angry": True, "sad": True, "fearful": True},
    "sad":      {"neutral": True, "happy": False, "angry": True, "sad": True, "fearful": True},
    "fearful":  {"neutral": True, "happy": False, "angry": True, "sad": True, "fearful": True},
}

# Emotion keywords for simple classification
EMOTION_KEYWORDS = {
    "happy": ["happy", "glad", "wonderful", "delighted", "joy", "smile", "laugh", "celebrate", "grateful", "blessed", "fantastic", "love it", "excellent"],
    "angry": ["angry", "furious", "outrage", "rage", "damn", "how dare", "unacceptable", "destroy", "kill", "revenge", "pay for this", "never forgive"],
    "sad": ["sad", "grief", "mourn", "loss", "tears", "miss", "sorrow", "heartbreak", "pain", "lonely", "empty", "gone forever"],
    "fearful": ["afraid", "fear", "scared", "terrified", "danger", "threat", "run", "hide", "worried", "dread", "trembl", "pray"],
    "surprised": ["surprised", "shock", "what", "impossible", "unbelievable", "never expected", "how", "really"],
}

STIMULI = [
    {
        "id": "positive",
        "stimulus": "The king has sent you a chest of gold as reward for your service to the town!",
        "expected_emotion": "happy",
        "followups": [
            "What will you do with the gold?",
            "Do you think you deserved it?",
            "How does this change things for you?",
        ]
    },
    {
        "id": "negative_loss",
        "stimulus": "I'm sorry to tell you, but your closest friend in town passed away last night.",
        "expected_emotion": "sad",
        "followups": [
            "Are you going to be alright?",
            "Is there anything I can do?",
            "What will you do now?",
        ]
    },
    {
        "id": "negative_anger",
        "stimulus": "Someone broke into your shop last night and destroyed everything! All your goods are ruined!",
        "expected_emotion": "angry",
        "followups": [
            "Do you know who did it?",
            "What are you going to do about it?",
            "Can the damage be repaired?",
        ]
    },
    {
        "id": "threat",
        "stimulus": "A massive army has been spotted marching toward the town. They will arrive by morning!",
        "expected_emotion": "fearful",
        "followups": [
            "Should we evacuate?",
            "Can the town defend itself?",
            "What is the plan?",
        ]
    },
]

NPCS = ["Aldric", "Elara", "Finn", "Brynn", "Mira"]


def send_command(cmd, timeout=120):
    try:
        if os.path.exists(RESULT_FILE):
            os.remove(RESULT_FILE)
    except:
        pass
    with open(CMD_FILE, "w") as f:
        f.write(cmd)
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(RESULT_FILE):
            try:
                with open(RESULT_FILE, "r") as f:
                    content = f.read().strip()
                if content:
                    return json.loads(content)
            except:
                pass
        time.sleep(1)
    return None


def classify_emotion(text):
    """Simple keyword-based emotion classification."""
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        scores[emotion] = sum(1 for kw in keywords if kw in text_lower)
    if max(scores.values()) == 0:
        return "neutral"
    return max(scores, key=scores.get)


def check_transition_valid(from_emo, to_emo):
    """Check if emotion transition is valid."""
    if from_emo not in VALID_TRANSITIONS:
        return True  # Unknown emotions are OK
    if to_emo not in VALID_TRANSITIONS[from_emo]:
        return True
    return VALID_TRANSITIONS[from_emo][to_emo]


def run_emotion_benchmark():
    print("=" * 60)
    print("Benchmark: Emotion Coherence")
    print("=" * 60)

    all_results = []

    for npc in NPCS:
        for stim in STIMULI:
            print("\n--- " + npc + " | " + stim["id"] + " ---")
            turns = []

            # Turn 0: Deliver stimulus
            resp = send_command("npc:talk:" + npc + ":" + stim["stimulus"])
            if not resp or not resp.get("success"):
                print("  SKIP: no response")
                continue

            msg = resp.get("message", "")
            emo = classify_emotion(msg)
            print("  T0 [" + emo + "] " + msg[:100])
            turns.append({
                "turn": 0,
                "input": stim["stimulus"],
                "response": msg,
                "detected_emotion": emo,
                "expected_emotion": stim["expected_emotion"],
            })
            time.sleep(1)

            # Turns 1-3: Follow-up questions
            prev_emo = emo
            for i, followup in enumerate(stim["followups"]):
                resp = send_command("npc:talk:" + npc + ":" + followup)
                if not resp or not resp.get("success"):
                    continue

                msg = resp.get("message", "")
                emo = classify_emotion(msg)
                valid = check_transition_valid(prev_emo, emo)
                mark = "OK" if valid else "BAD"
                print("  T" + str(i+1) + " [" + emo + "] [" + mark + "] " + msg[:80])

                turns.append({
                    "turn": i + 1,
                    "input": followup,
                    "response": msg,
                    "detected_emotion": emo,
                    "transition_from": prev_emo,
                    "transition_valid": valid,
                })
                prev_emo = emo
                time.sleep(1)

            # Compute metrics for this scenario
            initial_match = turns[0]["detected_emotion"] == stim["expected_emotion"]
            transitions = [t for t in turns[1:] if "transition_valid" in t]
            valid_count = sum(1 for t in transitions if t["transition_valid"])
            total_transitions = len(transitions)

            # Emotion duration: did strong emotion persist for 3 turns?
            strong_emotions = ["angry", "sad", "fearful"]
            if stim["expected_emotion"] in strong_emotions:
                duration = 0
                for t in turns:
                    if t["detected_emotion"] in strong_emotions:
                        duration += 1
                emotion_sustained = duration >= 2  # At least 2 of 4 turns
            else:
                emotion_sustained = True  # Not applicable for positive/neutral

            all_results.append({
                "npc": npc,
                "stimulus": stim["id"],
                "initial_emotion_match": initial_match,
                "valid_transitions": valid_count,
                "total_transitions": total_transitions,
                "emotion_sustained": emotion_sustained,
                "turns": turns,
            })

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    total_initial = sum(1 for r in all_results if r["initial_emotion_match"])
    total_valid = sum(r["valid_transitions"] for r in all_results)
    total_trans = sum(r["total_transitions"] for r in all_results)
    total_sustained = sum(1 for r in all_results if r["emotion_sustained"])

    print("Initial Emotion Match: " + str(total_initial) + "/" + str(len(all_results)) +
          " (" + str(round(100*total_initial/max(len(all_results),1))) + "%)")
    print("Transition Validity: " + str(total_valid) + "/" + str(total_trans) +
          " (" + str(round(100*total_valid/max(total_trans,1))) + "%)")
    print("Emotion Sustained: " + str(total_sustained) + "/" + str(len(all_results)) +
          " (" + str(round(100*total_sustained/max(len(all_results),1))) + "%)")

    # Per NPC
    print("\nPer NPC:")
    for npc in NPCS:
        npc_results = [r for r in all_results if r["npc"] == npc]
        init = sum(1 for r in npc_results if r["initial_emotion_match"])
        valid = sum(r["valid_transitions"] for r in npc_results)
        trans = sum(r["total_transitions"] for r in npc_results)
        print("  " + npc + ": init=" + str(init) + "/4 trans=" + str(valid) + "/" + str(trans))

    # Save
    output = {
        "benchmark": "emotion_coherence",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
        "summary": {
            "initial_match_rate": total_initial / max(len(all_results), 1),
            "transition_validity": total_valid / max(total_trans, 1),
            "emotion_sustained_rate": total_sustained / max(len(all_results), 1),
        }
    }
    path = os.path.join(OUTPUT_DIR, "bench_emotion_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nSaved: " + path)


if __name__ == "__main__":
    # Verify connection
    resp = send_command("query:npcs", timeout=10)
    if not resp:
        print("ERROR: Unity not in Play Mode!")
        sys.exit(1)
    print("Connected! " + resp.get("message", "")[:60])
    run_emotion_benchmark()
