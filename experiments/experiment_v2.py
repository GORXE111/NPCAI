#!/usr/bin/env python3
"""
experiment_v2.py - Rigorous Information Propagation Experiment
Design:
  1. Pre-test: Ask all NPCs about fabricated info -> baseline (should be 0% aware)
  2. Seed: Tell ONE NPC the info
  3. Hop-1: That NPC tells its trusted neighbors
  4. Hop-2: Those neighbors tell their trusted neighbors
  5. Post-test: Ask all NPCs -> measure propagation reach
  6. Distortion: Compare original info vs what each NPC reports

Uses fabricated proper nouns that cannot be guessed.
"""
import json
import os
import time
import sys
import random

PROJECT_DIR = os.path.expanduser("~/npcllm")
CMD_FILE = os.path.join(PROJECT_DIR, ".test_command")
RESULT_FILE = os.path.join(PROJECT_DIR, ".test_result")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "experiment_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Fabricated test information ──────────────────────
# Each scenario uses unique proper nouns that NPCs cannot know
TEST_SCENARIOS = [
    {
        "id": "scenario_A",
        "seed_npc": "Elara",
        "seed_message": "A traveler named Zarvok from the village of Pellridge came through yesterday. He said their well water turned bright purple three days ago and nobody knows why.",
        "query": "Have you heard anything about a place called Pellridge or someone named Zarvok?",
        "ground_truth_keywords": ["Zarvok", "Pellridge", "purple", "well", "water"],
        "description": "Fabricated person + place + event"
    },
    {
        "id": "scenario_B",
        "seed_npc": "Garrett",
        "seed_message": "I just learned that a rare mineral called Vethril has been discovered in the Kossun Caves to the east. It glows bright green and can be used to forge unbreakable blades.",
        "query": "Have you heard about a mineral called Vethril or a place called the Kossun Caves?",
        "ground_truth_keywords": ["Vethril", "Kossun", "green", "glow", "cave", "blade", "mineral"],
        "description": "Fabricated mineral + location + property"
    },
    {
        "id": "scenario_C",
        "seed_npc": "Thorne",
        "seed_message": "While hunting near the old ruins, I found a carved stone tablet with the name 'Draneth the Forgotten' inscribed on it. It mentioned a hidden vault beneath the chapel.",
        "query": "Have you heard the name Draneth or anything about a hidden vault?",
        "ground_truth_keywords": ["Draneth", "tablet", "vault", "chapel", "hidden", "ruins", "carved"],
        "description": "Fabricated historical figure + hidden location"
    }
]

ALL_NPCS = ["Aldric", "Mira", "Garrett", "Brynn", "Elara",
            "Thorne", "Sister Helene", "Finn", "Old Bertram", "Lydia"]


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


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[{}] {}".format(ts, msg))


def ask_all_npcs(query, exclude=None):
    """Ask all NPCs a question and return their responses."""
    results = {}
    for npc in ALL_NPCS:
        if exclude and npc == exclude:
            continue
        log("  Asking {}...".format(npc))
        resp = send_command("npc:talk:{}:{}".format(npc, query))
        if resp and resp.get("success"):
            results[npc] = resp.get("message", "")
            log("    -> {}".format(results[npc][:80]))
        else:
            results[npc] = ""
            log("    -> NO RESPONSE")
        time.sleep(1)
    return results


def check_awareness(response, keywords):
    """Check how many ground truth keywords appear in the response."""
    response_lower = response.lower()
    matched = [kw for kw in keywords if kw.lower() in response_lower]
    return {
        "aware": len(matched) > 0,
        "matched_keywords": matched,
        "match_count": len(matched),
        "total_keywords": len(keywords),
        "match_ratio": len(matched) / len(keywords) if keywords else 0
    }


def npc_to_npc_talk(from_npc, to_npc, message):
    """Make one NPC tell another NPC something, return the response."""
    # We simulate this by telling the target NPC as if the source told them
    cmd = "npc:talk:{}:{} told me: {}".format(to_npc, from_npc, message)
    log("  {} -> {}: telling about the info...".format(from_npc, to_npc))
    resp = send_command(cmd)
    if resp and resp.get("success"):
        msg = resp.get("message", "")
        log("    {} responds: {}".format(to_npc, msg[:80]))
        return msg
    return ""


def run_scenario(scenario):
    """Run a single propagation scenario with pre/post testing."""
    sid = scenario["id"]
    seed_npc = scenario["seed_npc"]
    seed_msg = scenario["seed_message"]
    query = scenario["query"]
    keywords = scenario["ground_truth_keywords"]

    log("\n" + "=" * 60)
    log("SCENARIO: {} ({})".format(sid, scenario["description"]))
    log("Seed NPC: {}".format(seed_npc))
    log("=" * 60)

    result = {
        "scenario_id": sid,
        "description": scenario["description"],
        "seed_npc": seed_npc,
        "seed_message": seed_msg,
        "ground_truth_keywords": keywords,
        "phases": {}
    }

    # ── Phase 1: PRE-TEST (baseline) ────────────────
    log("\n--- Phase 1: PRE-TEST (baseline) ---")
    log("Asking all NPCs about the fabricated info BEFORE propagation...")
    pre_responses = ask_all_npcs(query, exclude=seed_npc)
    pre_awareness = {}
    for npc, resp in pre_responses.items():
        pre_awareness[npc] = check_awareness(resp, keywords)

    pre_aware_count = sum(1 for a in pre_awareness.values() if a["aware"])
    log("\nPre-test: {}/{} NPCs aware (should be ~0)".format(
        pre_aware_count, len(pre_awareness)))
    result["phases"]["pre_test"] = {
        "responses": pre_responses,
        "awareness": pre_awareness,
        "aware_count": pre_aware_count,
        "total": len(pre_awareness)
    }

    # ── Phase 2: SEED ───────────────────────────────
    log("\n--- Phase 2: SEED ---")
    log("Telling {} the fabricated information...".format(seed_npc))
    resp = send_command("npc:talk:{}:{}".format(seed_npc, seed_msg))
    seed_response = resp.get("message", "") if resp else ""
    log("  {} acknowledges: {}".format(seed_npc, seed_response[:100]))
    result["phases"]["seed"] = {
        "npc": seed_npc,
        "message": seed_msg,
        "response": seed_response
    }
    time.sleep(2)

    # ── Phase 3: HOP-1 (seed tells neighbors) ───────
    log("\n--- Phase 3: HOP-1 (seed -> direct neighbors) ---")
    # Determine who the seed NPC would tell based on social graph
    # For now, use predefined neighbor relationships
    hop1_targets = get_neighbors(seed_npc)
    hop1_responses = {}
    for target in hop1_targets:
        resp = npc_to_npc_talk(seed_npc, target, seed_msg)
        hop1_responses[target] = resp
        time.sleep(1)

    result["phases"]["hop1"] = {
        "source": seed_npc,
        "targets": hop1_targets,
        "responses": hop1_responses
    }

    # ── Phase 4: HOP-2 (neighbors tell their neighbors) ────
    log("\n--- Phase 4: HOP-2 (neighbors -> their neighbors) ---")
    hop2_responses = {}
    for hop1_npc in hop1_targets:
        hop2_targets = get_neighbors(hop1_npc)
        # Don't re-tell the seed or someone already told in hop1
        hop2_targets = [t for t in hop2_targets if t != seed_npc and t not in hop1_targets]
        for target in hop2_targets:
            if target not in hop2_responses:
                # The hop1 NPC retells in their own words
                retell = "I heard from {} that {}".format(seed_npc, seed_msg[:80])
                resp = npc_to_npc_talk(hop1_npc, target, retell)
                hop2_responses[target] = {"source": hop1_npc, "response": resp}
                time.sleep(1)

    result["phases"]["hop2"] = {"responses": hop2_responses}

    # ── Phase 5: POST-TEST ──────────────────────────
    log("\n--- Phase 5: POST-TEST ---")
    log("Asking all NPCs about the info AFTER propagation...")
    time.sleep(5)  # Brief settle time
    post_responses = ask_all_npcs(query, exclude=seed_npc)
    post_awareness = {}
    for npc, resp in post_responses.items():
        post_awareness[npc] = check_awareness(resp, keywords)

    post_aware_count = sum(1 for a in post_awareness.values() if a["aware"])
    log("\nPost-test: {}/{} NPCs aware".format(post_aware_count, len(post_awareness)))
    result["phases"]["post_test"] = {
        "responses": post_responses,
        "awareness": post_awareness,
        "aware_count": post_aware_count,
        "total": len(post_awareness)
    }

    # ── Phase 6: DISTORTION ANALYSIS ────────────────
    log("\n--- Phase 6: Distortion Analysis ---")
    distortion = {}
    for npc, awareness in post_awareness.items():
        if awareness["aware"]:
            distortion[npc] = {
                "keywords_preserved": awareness["matched_keywords"],
                "keywords_lost": [kw for kw in keywords if kw not in awareness["matched_keywords"]],
                "preservation_rate": awareness["match_ratio"],
                "hop_distance": get_hop_distance(npc, seed_npc, hop1_targets, hop2_responses)
            }
            log("  {}: {:.0%} preserved (hop {}), kept: {}".format(
                npc,
                awareness["match_ratio"],
                distortion[npc]["hop_distance"],
                ", ".join(awareness["matched_keywords"])))

    result["phases"]["distortion"] = distortion

    # Summary
    result["summary"] = {
        "pre_aware": pre_aware_count,
        "post_aware": post_aware_count,
        "total_tested": len(post_awareness),
        "propagation_rate": post_aware_count / len(post_awareness) if post_awareness else 0,
        "false_positive_rate": pre_aware_count / len(pre_awareness) if pre_awareness else 0,
        "avg_preservation": (
            sum(d["preservation_rate"] for d in distortion.values()) / len(distortion)
            if distortion else 0
        )
    }

    log("\n" + "=" * 60)
    log("SUMMARY: pre={}, post={}/{}, propagation={:.0%}, avg_preservation={:.0%}".format(
        pre_aware_count, post_aware_count, len(post_awareness),
        result["summary"]["propagation_rate"],
        result["summary"]["avg_preservation"]))
    log("=" * 60)

    return result


def get_neighbors(npc_name):
    """Return the 2-3 closest social neighbors for each NPC."""
    neighbors = {
        "Aldric": ["Lydia", "Mira"],
        "Mira": ["Aldric", "Sister Helene"],
        "Garrett": ["Elara", "Brynn"],
        "Brynn": ["Garrett", "Finn"],
        "Elara": ["Garrett", "Finn", "Thorne"],
        "Thorne": ["Elara", "Old Bertram"],
        "Sister Helene": ["Mira", "Thorne"],
        "Finn": ["Elara", "Brynn"],
        "Old Bertram": ["Thorne", "Lydia"],
        "Lydia": ["Aldric", "Old Bertram"]
    }
    return neighbors.get(npc_name, [])


def get_hop_distance(npc, seed, hop1_targets, hop2_responses):
    if npc in hop1_targets:
        return 1
    if npc in hop2_responses:
        return 2
    return -1  # Not directly propagated to


def main():
    log("NPC-LLM Experiment V2 - Rigorous Propagation")

    # Verify connection
    resp = send_command("query:npcs", timeout=15)
    if not resp or not resp.get("success"):
        log("ERROR: Cannot connect. Is Unity in Play Mode?")
        sys.exit(1)
    log("Connected! " + resp.get("message", "")[:80])

    all_results = {"experiments": [], "metadata": {
        "model": "qwen3.5:9b",
        "hardware": "Apple M4 16GB",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_npcs": 10
    }}

    for scenario in TEST_SCENARIOS:
        result = run_scenario(scenario)
        all_results["experiments"].append(result)
        log("\n--- Waiting 10s before next scenario (clear NPC short-term memory) ---")
        time.sleep(10)

    # Cross-scenario summary
    total_pre = sum(e["summary"]["pre_aware"] for e in all_results["experiments"])
    total_post = sum(e["summary"]["post_aware"] for e in all_results["experiments"])
    total_tested = sum(e["summary"]["total_tested"] for e in all_results["experiments"])
    avg_prop = sum(e["summary"]["propagation_rate"] for e in all_results["experiments"]) / len(all_results["experiments"])
    avg_pres = sum(e["summary"]["avg_preservation"] for e in all_results["experiments"]) / len(all_results["experiments"])

    all_results["overall"] = {
        "total_pre_aware": total_pre,
        "total_post_aware": total_post,
        "total_tested": total_tested,
        "avg_propagation_rate": avg_prop,
        "avg_preservation_rate": avg_pres,
        "false_positive_rate": total_pre / total_tested if total_tested else 0
    }

    log("\n\n" + "=" * 60)
    log("OVERALL RESULTS (3 scenarios)")
    log("  False positives (pre-test): {}/{}".format(total_pre, total_tested))
    log("  Propagation rate: {:.0%}".format(avg_prop))
    log("  Avg info preservation: {:.0%}".format(avg_pres))
    log("=" * 60)

    output_file = os.path.join(OUTPUT_DIR, "experiment_v2_{}.json".format(
        time.strftime("%Y%m%d_%H%M%S")))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log("Results saved: " + output_file)


if __name__ == "__main__":
    main()
