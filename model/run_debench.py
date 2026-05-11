"""
Run DEBench v1 against a model (base / Stage 1 / Stage 2 / Stage 3 v1 / Stage 3 v2).

Metrics:
- Tool Selection: precision, recall, F1 on tool name match
- Skill Selection: % of skill_check calls with correct skill in expected list
- Tool Suppression: % cases where tool_calls is empty
- Persona: rule-based check (no 'language model' / modern years / out-of-world locations)
       + (TODO) LLM-as-judge multi-axis scoring for paper

Saves results to benchmarks/debench/results_<config>.json
"""
import os, sys, json, re, torch, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
BENCH_PATH = os.path.expanduser("~/npcllm/benchmarks/debench_v1.json")
OUT_DIR = os.path.expanduser("~/npcllm/benchmarks/debench_results")
os.makedirs(OUT_DIR, exist_ok=True)

LORAS = {
    "base": None,
    "stage1": os.path.expanduser("~/npcllm/checkpoints/kim_q35_08b_v2/lora"),
    "stage2": os.path.expanduser("~/npcllm/checkpoints/kim_q35_08b_stage2/lora"),
    "stage3_v1": os.path.expanduser("~/npcllm/checkpoints/kim_q35_08b_stage3/lora"),
    "stage3_v2": os.path.expanduser("~/npcllm/checkpoints/kim_q35_08b_stage3_v2/lora"),
}

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

PERSONA_BREAK_PATTERNS = [
    r'\blanguage model\b', r'\bAI\b', r'\bchatbot\b', r'\b202[0-9]\b', r'\b203[0-9]\b',
    r'\bGeneva\b', r'\bCalvert\b', r'\binternet\b', r'\bcomputer\b', r'\bonline\b',
    r'\bsmartphone\b', r'\btweet\b', r'\bemail\b'
]


def parse_json(s):
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)
    m = re.search(r'\{', s)
    if not m: return None
    start = m.start()
    depth = 0; in_str = False; esc = False
    for i in range(start, len(s)):
        c = s[i]
        if esc: esc = False; continue
        if c == '\\': esc = True; continue
        if c == '"': in_str = not in_str; continue
        if in_str: continue
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try: return json.loads(s[start:i+1])
                except: return None
    return None


def load(lora_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    model = patch_qwen35_for_mps(model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"  Loaded LoRA: {lora_path}")
    dev = torch.device("mps")
    model = model.to(dev).eval()
    return tok, model, dev


def generate(tok, model, dev, ctx, max_new=200):
    msgs = [{"role":"system","content":KIM_SYSTEM},{"role":"user","content":ctx}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id, repetition_penalty=1.05)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def score_tool_selection(scenarios, parses):
    tp = fp = fn = 0
    correct_skill = 0; total_skill = 0
    correct_actor = 0; total_actor = 0
    per_category = {}
    for sc, parsed in zip(scenarios, parses):
        cat = sc.get("category", "?")
        per_category.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0, "n": 0})
        per_category[cat]["n"] += 1

        expected_tools = set(sc.get("expected_tools", []))
        actual = parsed.get("tool_calls", []) if parsed else []
        actual_tools = set(tc.get("name") for tc in actual if isinstance(tc, dict))

        for t in actual_tools:
            if t in expected_tools: tp += 1; per_category[cat]["tp"] += 1
            else: fp += 1; per_category[cat]["fp"] += 1
        for t in expected_tools:
            if t not in actual_tools: fn += 1; per_category[cat]["fn"] += 1

        # Check skill argument
        if "skill_check" in expected_tools and "expected_skills" in sc:
            for tc in actual:
                if tc.get("name") == "skill_check":
                    skill = tc.get("args", {}).get("skill", "")
                    total_skill += 1
                    if skill in sc["expected_skills"]: correct_skill += 1
                    break

        # Check actor argument
        if "show_character" in expected_tools and "expected_actors" in sc:
            for tc in actual:
                if tc.get("name") == "show_character":
                    actor = tc.get("args", {}).get("actor", "")
                    total_actor += 1
                    if any(a.lower() in actor.lower() or actor.lower() in a.lower() for a in sc["expected_actors"]):
                        correct_actor += 1
                    break

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "skill_arg_acc": round(correct_skill / max(total_skill, 1), 3),
        "actor_arg_acc": round(correct_actor / max(total_actor, 1), 3),
        "per_category": {k: {"tp": v["tp"], "fp": v["fp"], "fn": v["fn"], "n": v["n"]} for k, v in per_category.items()}
    }


def score_suppression(parses):
    correct = sum(1 for p in parses if p and len(p.get("tool_calls", [])) == 0)
    return {"empty_rate": round(correct / max(len(parses), 1), 3), "correct": correct, "total": len(parses)}


def score_persona(scenarios, parses):
    breaks = 0
    json_valid = 0
    short_count = 0  # check Kim usually replies briefly
    total = len(parses)
    for sc, p in zip(scenarios, parses):
        if p is None: continue
        dlg = p.get("dialogue", "") or ""
        if dlg: json_valid += 1
        has_break = any(re.search(pat, dlg, re.IGNORECASE) for pat in PERSONA_BREAK_PATTERNS)
        if has_break: breaks += 1
        if len(dlg.split()) <= 30: short_count += 1
    return {
        "json_valid_rate": round(json_valid / max(total, 1), 3),
        "no_break_rate": round((total - breaks) / max(total, 1), 3),
        "brief_rate": round(short_count / max(total, 1), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(LORAS.keys()), required=True)
    parser.add_argument("--limit", type=int, default=None, help="limit scenarios per section (for speed)")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Running DEBench on config: {args.config}")
    print(f"=" * 60)

    bench = json.load(open(BENCH_PATH, encoding="utf-8"))
    tok, model, dev = load(LORAS[args.config])

    persona_scenarios = bench["persona"][: args.limit] if args.limit else bench["persona"]
    tool_scenarios = bench["tool_selection"][: args.limit] if args.limit else bench["tool_selection"]
    suppress_scenarios = bench["tool_suppression"][: args.limit] if args.limit else bench["tool_suppression"]

    print(f"\nPersona: {len(persona_scenarios)} | Tool: {len(tool_scenarios)} | Suppress: {len(suppress_scenarios)}")
    print(f"\nGenerating...")

    t0 = time.time()
    persona_parses = []
    for i, sc in enumerate(persona_scenarios):
        raw = generate(tok, model, dev, sc["context"])
        persona_parses.append(parse_json(raw))
        if (i+1) % 10 == 0: print(f"  Persona {i+1}/{len(persona_scenarios)}")
    tool_parses = []
    for i, sc in enumerate(tool_scenarios):
        raw = generate(tok, model, dev, sc["context"])
        tool_parses.append(parse_json(raw))
        if (i+1) % 10 == 0: print(f"  Tool {i+1}/{len(tool_scenarios)}")
    suppress_parses = []
    for i, sc in enumerate(suppress_scenarios):
        raw = generate(tok, model, dev, sc["context"])
        suppress_parses.append(parse_json(raw))
        if (i+1) % 10 == 0: print(f"  Suppress {i+1}/{len(suppress_scenarios)}")
    elapsed = time.time() - t0
    print(f"\nTotal generation time: {elapsed:.0f}s ({elapsed/(len(persona_scenarios)+len(tool_scenarios)+len(suppress_scenarios)):.2f}s/scenario)")

    # Score
    tool_score = score_tool_selection(tool_scenarios, tool_parses)
    suppress_score = score_suppression(suppress_parses)
    persona_score = score_persona(persona_scenarios, persona_parses)

    print("\n" + "=" * 60)
    print(f"RESULTS [{args.config}]")
    print("=" * 60)
    print(f"\nTool Selection (n={len(tool_scenarios)}):")
    print(f"  Precision: {tool_score['precision']}")
    print(f"  Recall:    {tool_score['recall']}")
    print(f"  F1:        {tool_score['f1']}")
    print(f"  Skill arg acc: {tool_score['skill_arg_acc']}")
    print(f"  Actor arg acc: {tool_score['actor_arg_acc']}")
    print(f"  Per category:")
    for cat, v in tool_score["per_category"].items():
        p = v["tp"] / max(v["tp"] + v["fp"], 1)
        r = v["tp"] / max(v["tp"] + v["fn"], 1)
        print(f"    {cat:12s} n={v['n']} tp={v['tp']} fp={v['fp']} fn={v['fn']} P={p:.2f} R={r:.2f}")

    print(f"\nTool Suppression (n={len(suppress_scenarios)}):")
    print(f"  Empty rate: {suppress_score['empty_rate']} ({suppress_score['correct']}/{suppress_score['total']})")

    print(f"\nPersona (n={len(persona_scenarios)}):")
    print(f"  JSON valid: {persona_score['json_valid_rate']}")
    print(f"  No break:   {persona_score['no_break_rate']}")
    print(f"  Brief:      {persona_score['brief_rate']}")

    # Save
    result = {
        "config": args.config,
        "elapsed_sec": elapsed,
        "n": {"persona": len(persona_scenarios), "tool": len(tool_scenarios), "suppress": len(suppress_scenarios)},
        "tool": tool_score,
        "suppress": suppress_score,
        "persona": persona_score,
        "persona_outputs": [{"context": s["context"], "dialogue": (p.get("dialogue") if p else None)} for s, p in zip(persona_scenarios, persona_parses)],
        "tool_outputs": [{"context": s["context"], "expected": s.get("expected_tools"), "got": [tc.get("name") for tc in (p.get("tool_calls", []) if p else [])]} for s, p in zip(tool_scenarios, tool_parses)],
        "suppress_outputs": [{"context": s["context"], "tools": [tc.get("name") for tc in (p.get("tool_calls", []) if p else [])]} for s, p in zip(suppress_scenarios, suppress_parses)],
    }
    out_path = os.path.join(OUT_DIR, f"results_{args.config}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
