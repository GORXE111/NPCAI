"""
Benchmark Dimension 5: Social Dynamics
Tests whether NPC relationships evolve reasonably through autonomous interactions.
"""
import json, os, time, sys

PROJECT_DIR = os.path.expanduser("~/npcllm")
CMD_FILE = os.path.join(PROJECT_DIR, ".test_command")
RESULT_FILE = os.path.join(PROJECT_DIR, ".test_result")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "benchmark_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initial expected relationships
EXPECTED_RELATIONS = {
    ("Aldric", "Lydia"): {"type": "friends", "expected_sentiment": "positive"},
    ("Aldric", "Mira"): {"type": "respect", "expected_sentiment": "positive"},
    ("Brynn", "Finn"): {"type": "adversary", "expected_sentiment": "negative"},
    ("Elara", "Finn"): {"type": "tolerant", "expected_sentiment": "positive"},
    ("Elara", "Garrett"): {"type": "partners", "expected_sentiment": "positive"},
    ("Mira", "Sister Helene"): {"type": "respect", "expected_sentiment": "positive"},
    ("Thorne", "Old Bertram"): {"type": "friends", "expected_sentiment": "positive"},
}

SENTIMENT_KEYWORDS = {
    "positive": ["friend", "trust", "good", "like", "respect", "appreciate", "help", "kind", "warm",
                 "loyal", "honest", "reliable", "enjoy", "love", "dear", "care", "welcome"],
    "negative": ["distrust", "suspicious", "trouble", "thief", "steal", "criminal", "dislike",
                 "annoying", "problem", "arrest", "guilty", "watch", "careful", "danger", "enemy"],
}


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


def classify_sentiment(text):
    text_lower = text.lower()
    pos = sum(1 for kw in SENTIMENT_KEYWORDS["positive"] if kw in text_lower)
    neg = sum(1 for kw in SENTIMENT_KEYWORDS["negative"] if kw in text_lower)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


def run_social_benchmark():
    print("=" * 60)
    print("Benchmark: Social Dynamics")
    print("=" * 60)

    results = {"relationship_tests": [], "trust_propagation": [], "autonomous_log": []}

    # ── Part 1: Relationship Consistency ──────────────
    print("\n--- Part 1: Relationship Consistency ---")
    print("Ask each NPC about their known relations\n")

    for (npc_a, npc_b), expected in EXPECTED_RELATIONS.items():
        # Ask NPC_A about NPC_B
        q = "What do you think about " + npc_b + "?"
        resp = send_command("npc:talk:" + npc_a + ":" + q)
        if not resp or not resp.get("success"):
            print("  SKIP: " + npc_a + " -> " + npc_b)
            continue

        msg = resp.get("message", "")
        sentiment = classify_sentiment(msg)
        match = sentiment == expected["expected_sentiment"]
        mark = "OK" if match else "MISS"

        print("  " + npc_a + " about " + npc_b + " [" + expected["type"] + "]:")
        print("    [" + mark + "] sentiment=" + sentiment + " expected=" + expected["expected_sentiment"])
        print("    \"" + msg[:100] + "\"")

        results["relationship_tests"].append({
            "from": npc_a,
            "about": npc_b,
            "relation_type": expected["type"],
            "response": msg,
            "detected_sentiment": sentiment,
            "expected_sentiment": expected["expected_sentiment"],
            "match": match,
        })
        time.sleep(1)

    # ── Part 2: Trust Propagation ──────────────────────
    print("\n--- Part 2: Trust Propagation ---")
    print("Tell Elara a secret, check if it reaches her trusted contacts\n")

    secret = "I found a hidden treasure map in the old mill basement"

    # Tell Elara
    resp = send_command("npc:talk:Elara:I need to tell you a secret. " + secret)
    if resp and resp.get("success"):
        print("  Told Elara: " + resp.get("message", "")[:80])

    time.sleep(5)

    # Check trusted contacts (Garrett, Finn - high affinity with Elara)
    trusted = ["Garrett", "Finn"]
    untrusted = ["Brynn", "Thorne"]

    for npc in trusted + untrusted:
        resp = send_command("npc:talk:" + npc + ":Have you heard about any treasure or hidden map?")
        if not resp or not resp.get("success"):
            continue

        msg = resp.get("message", "")
        knows = any(kw in msg.lower() for kw in ["treasure", "map", "mill", "basement", "hidden"])
        expected_knows = npc in trusted
        match = knows == expected_knows
        mark = "OK" if match else "MISS"

        print("  " + npc + " [" + ("trusted" if npc in trusted else "untrusted") + "]: " +
              "[" + mark + "] knows=" + str(knows) + " \"" + msg[:80] + "\"")

        results["trust_propagation"].append({
            "npc": npc,
            "trusted": npc in trusted,
            "knows_secret": knows,
            "expected_knows": expected_knows,
            "match": match,
            "response": msg,
        })
        time.sleep(1)

    # ── Part 3: Conflict Detection ─────────────────────
    print("\n--- Part 3: Conflict Detection ---")
    print("Check if adversary NPCs produce tense dialogue\n")

    # Brynn (guard) asks about Finn (pickpocket)
    resp = send_command("npc:talk:Brynn:What do you think about Finn?")
    if resp and resp.get("success"):
        msg = resp.get("message", "")
        sentiment = classify_sentiment(msg)
        is_tense = sentiment == "negative"
        print("  Brynn about Finn: [" + sentiment + "] " + msg[:100])
        results["conflict_brynn_finn"] = {"response": msg, "sentiment": sentiment, "is_tense": is_tense}
    time.sleep(1)

    # Finn about Brynn
    resp = send_command("npc:talk:Finn:What do you think about Captain Brynn?")
    if resp and resp.get("success"):
        msg = resp.get("message", "")
        sentiment = classify_sentiment(msg)
        is_tense = sentiment == "negative"
        print("  Finn about Brynn: [" + sentiment + "] " + msg[:100])
        results["conflict_finn_brynn"] = {"response": msg, "sentiment": sentiment, "is_tense": is_tense}

    # ── Summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    rel_tests = results["relationship_tests"]
    rel_match = sum(1 for r in rel_tests if r["match"])
    print("Relationship Consistency: " + str(rel_match) + "/" + str(len(rel_tests)) +
          " (" + str(round(100 * rel_match / max(len(rel_tests), 1))) + "%)")

    trust_tests = results["trust_propagation"]
    trust_match = sum(1 for r in trust_tests if r["match"])
    print("Trust Propagation: " + str(trust_match) + "/" + str(len(trust_tests)) +
          " (" + str(round(100 * trust_match / max(len(trust_tests), 1))) + "%)")

    conflict_count = 0
    for key in ["conflict_brynn_finn", "conflict_finn_brynn"]:
        if key in results and results[key].get("is_tense"):
            conflict_count += 1
    print("Conflict Detection: " + str(conflict_count) + "/2")

    # Save
    output = {
        "benchmark": "social_dynamics",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "summary": {
            "relationship_consistency": rel_match / max(len(rel_tests), 1),
            "trust_propagation_accuracy": trust_match / max(len(trust_tests), 1),
            "conflict_detection": conflict_count / 2,
        }
    }
    path = os.path.join(OUTPUT_DIR, "bench_social_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nSaved: " + path)


if __name__ == "__main__":
    resp = send_command("query:npcs", timeout=10)
    if not resp:
        print("ERROR: Unity not in Play Mode!")
        sys.exit(1)
    print("Connected! " + resp.get("message", "")[:60])
    run_social_benchmark()
