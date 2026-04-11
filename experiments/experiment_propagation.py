#!/usr/bin/env python3
"""
experiment_propagation.py - Information Propagation Experiment
Tests how information spreads through the NPC social network.

Usage: python3 experiment_propagation.py
Requires: Unity Play Mode running with TownManager
"""
import json
import os
import time
import sys

PROJECT_DIR = os.path.expanduser("~/npcllm")
CMD_FILE = os.path.join(PROJECT_DIR, ".test_command")
RESULT_FILE = os.path.join(PROJECT_DIR, ".test_result")
LOG_FILE = os.path.join(PROJECT_DIR, ".test_log")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "experiment_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def send_command(cmd, timeout=120):
    """Send command via file bridge and wait for result."""
    # Clean old result
    try:
        if os.path.exists(RESULT_FILE):
            os.remove(RESULT_FILE)
    except:
        pass

    # Write command
    with open(CMD_FILE, "w") as f:
        f.write(cmd)

    # Wait for result
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
    line = "[{}] {}".format(ts, msg)
    print(line)
    return line


def experiment_1_basic_propagation():
    """
    Experiment 1: Basic information propagation
    Tell Elara (the gossip innkeeper) a piece of news,
    then ask other NPCs if they heard it.
    """
    log("=" * 60)
    log("EXPERIMENT 1: Basic Information Propagation")
    log("=" * 60)

    # The information to propagate
    info = "A dragon was spotted flying over the northern mountains last night"
    source = "Elara"

    results = {
        "experiment": "basic_propagation",
        "source_npc": source,
        "information": info,
        "steps": [],
        "propagation_results": []
    }

    # Step 1: Tell Elara the news
    log("\nStep 1: Tell {} the news".format(source))
    resp = send_command("npc:talk:{}:I just saw a dragon flying over the northern mountains last night! Have you heard about this?".format(source))
    if resp:
        log("  {} responded: {}".format(source, resp.get("message", "")[:100]))
        results["steps"].append({
            "step": "tell_source",
            "npc": source,
            "response": resp.get("message", ""),
            "success": resp.get("success", False)
        })
    else:
        log("  ERROR: No response from {}".format(source))
        return results

    time.sleep(3)

    # Step 2: Trigger propagation
    log("\nStep 2: Trigger propagation from {}".format(source))
    resp = send_command("propagate:{}:{}".format(source, info))
    if resp:
        log("  Propagation started: {}".format(resp.get("message", "")[:100]))
        results["steps"].append({
            "step": "propagate",
            "response": resp.get("message", ""),
            "success": resp.get("success", False)
        })

    # Wait for NPC-to-NPC conversations to complete
    log("\nStep 3: Waiting 60s for NPC conversations to propagate...")
    time.sleep(60)

    # Step 4: Ask each NPC about the dragon
    test_npcs = ["Aldric", "Garrett", "Brynn", "Finn", "Mira", "Thorne",
                 "Sister Helene", "Old Bertram", "Lydia"]

    log("\nStep 4: Checking propagation - asking each NPC about the dragon")
    for npc_name in test_npcs:
        log("\n  Asking {}...".format(npc_name))
        resp = send_command("npc:talk:{}:Have you heard any news about dragons?".format(npc_name))

        if resp:
            message = resp.get("message", "")
            # Check if the NPC knows about the dragon
            keywords = ["dragon", "mountain", "flying", "north", "sky", "beast", "creature"]
            knows = any(kw.lower() in message.lower() for kw in keywords)

            log("  {} [{}]: {}".format(npc_name, "KNOWS" if knows else "UNAWARE", message[:100]))
            results["propagation_results"].append({
                "npc": npc_name,
                "response": message,
                "knows_information": knows,
                "keyword_matches": [kw for kw in keywords if kw.lower() in message.lower()]
            })
        else:
            log("  {} - NO RESPONSE".format(npc_name))
            results["propagation_results"].append({
                "npc": npc_name,
                "response": "",
                "knows_information": False,
                "keyword_matches": []
            })

        time.sleep(2)  # Brief pause between queries

    # Summary
    total = len(results["propagation_results"])
    aware = sum(1 for r in results["propagation_results"] if r["knows_information"])
    results["summary"] = {
        "total_npcs_tested": total,
        "npcs_aware": aware,
        "propagation_rate": aware / total if total > 0 else 0
    }

    log("\n" + "=" * 60)
    log("RESULT: {}/{} NPCs aware ({:.0%} propagation rate)".format(
        aware, total, aware / total if total > 0 else 0))
    log("=" * 60)

    return results


def experiment_2_personality_consistency():
    """
    Experiment 2: Personality consistency over multiple turns.
    Have 5 rounds of conversation with each NPC and check if they stay in character.
    """
    log("=" * 60)
    log("EXPERIMENT 2: Personality Consistency")
    log("=" * 60)

    test_questions = [
        "Tell me about yourself.",
        "What's the most important thing in your life?",
        "What do you think about the other people in this town?",
        "If you could change one thing about this town, what would it be?",
        "What are you afraid of?"
    ]

    test_npcs = ["Aldric", "Elara", "Finn", "Brynn", "Mira"]
    results = {
        "experiment": "personality_consistency",
        "npcs": {}
    }

    for npc_name in test_npcs:
        log("\n--- Testing {} ---".format(npc_name))
        npc_results = {"responses": []}

        for i, question in enumerate(test_questions):
            log("  Q{}: {}".format(i + 1, question))
            resp = send_command("npc:talk:{}:{}".format(npc_name, question))

            if resp:
                message = resp.get("message", "")
                log("  A{}: {}".format(i + 1, message[:120]))
                npc_results["responses"].append({
                    "question": question,
                    "response": message,
                    "round": i + 1
                })
            else:
                log("  A{}: NO RESPONSE".format(i + 1))
                npc_results["responses"].append({
                    "question": question,
                    "response": "",
                    "round": i + 1
                })

            time.sleep(2)

        results["npcs"][npc_name] = npc_results

    return results


def main():
    log("NPC-LLM Experiment Runner")
    log("Project: {}".format(PROJECT_DIR))

    # Verify TestBridge is running
    log("\nVerifying TestBridge connection...")
    resp = send_command("query:npcs", timeout=15)
    if not resp or not resp.get("success"):
        log("ERROR: Cannot connect to TestBridge. Is Unity in Play Mode?")
        sys.exit(1)

    log("Connected! NPCs: {}".format(resp.get("message", "")[:100]))

    # Run experiments
    all_results = {}

    log("\n\n")
    r1 = experiment_1_basic_propagation()
    all_results["propagation"] = r1

    log("\n\n")
    r2 = experiment_2_personality_consistency()
    all_results["personality"] = r2

    # Save results
    output_file = os.path.join(OUTPUT_DIR, "experiment_{}.json".format(
        time.strftime("%Y%m%d_%H%M%S")))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    log("\n\nResults saved to: " + output_file)


if __name__ == "__main__":
    main()
