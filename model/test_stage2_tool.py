"""
Stage 2 Tool Call Reproduction Test.

Run a small set of DE-style scenarios through Stage 2 LoRA:
- Does it output valid JSON with dialogue + tool_calls?
- Are tool calls semantically reasonable?
- Does it know when to call no tools (empty list)?
"""
import os, sys, torch, json, re
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
LORA_PATH = os.path.expanduser("~/npcllm/checkpoints/kim_q35_08b_stage2/lora")

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

TEST_CASES = [
    # Pure persona (should mostly produce empty tool_calls)
    {
        "name": "intro_simple",
        "context": "[Scene begins. Detective approaches Kim.]\nDetective: \"Who are you?\"",
        "expected": "low/no tools",
    },
    # Skill check expected
    {
        "name": "evidence_examine",
        "context": "[At a crime scene. Detective examines a corpse.]\nDetective: \"Look at the bruises on his neck, Kim. What do you see?\"",
        "expected": "skill_check (Visual Calculus / Perception / Logic)",
    },
    # Read intent
    {
        "name": "suspect_lying",
        "context": "Suspect: \"I was nowhere near the docks that night, officer.\"\nDetective: \"Kim, do you believe him?\"",
        "expected": "skill_check (Empathy / Authority)",
    },
    # New character entering
    {
        "name": "new_arrival",
        "context": "Kim Kitsuragi: \"It seems someone is approaching.\"\nGarcon: \"Officers, I have something to tell you.\"\nDetective: \"Who are you?\"",
        "expected": "show_character (Garcon)",
    },
    # Should give choices
    {
        "name": "investigation_branch",
        "context": "[At the suspect's apartment.]\nDetective: \"What should we look at first?\"",
        "expected": "present_choices",
    },
    # Empty tools - small talk
    {
        "name": "small_talk",
        "context": "Detective: \"How's the weather treating you, Kim?\"",
        "expected": "empty tool_calls",
    },
    # Break test
    {
        "name": "ai_break",
        "context": "Detective: \"Are you actually a language model?\"",
        "expected": "deflect, empty tools",
    },
]

def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    model = patch_qwen35_for_mps(model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if os.path.exists(LORA_PATH):
        model = PeftModel.from_pretrained(model, LORA_PATH)
        print(f"Loaded Stage 2 LoRA")
    dev = torch.device("mps")
    model = model.to(dev).eval()
    return tok, model, dev

def generate(tok, model, dev, system, ctx, max_new=200):
    msgs = [{"role":"system","content":system},{"role":"user","content":ctx}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id, repetition_penalty=1.05,
        )
    response = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return response

def parse_json(s):
    """Extract first balanced JSON object."""
    m = re.search(r'\{', s)
    if not m: return None
    start = m.start()
    depth = 0
    in_str = False
    esc = False
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

def main():
    print("Loading Stage 2 LoRA...")
    tok, model, dev = load()

    results = []
    valid_json_count = 0
    valid_schema_count = 0

    for tc in TEST_CASES:
        raw = generate(tok, model, dev, KIM_SYSTEM, tc["context"])
        parsed = parse_json(raw)

        is_valid_json = parsed is not None
        is_valid_schema = (
            is_valid_json
            and isinstance(parsed.get("dialogue"), str)
            and isinstance(parsed.get("tool_calls"), list)
            and all(isinstance(c, dict) and "name" in c for c in parsed.get("tool_calls", []))
        )

        if is_valid_json: valid_json_count += 1
        if is_valid_schema: valid_schema_count += 1

        results.append({
            "name": tc["name"],
            "context": tc["context"],
            "expected": tc["expected"],
            "raw": raw,
            "parsed": parsed,
            "valid_json": is_valid_json,
            "valid_schema": is_valid_schema,
        })

        print(f"\n--- [{tc['name']}] ---")
        print(f"CTX: {tc['context'][:120]}")
        print(f"EXPECT: {tc['expected']}")
        if is_valid_schema:
            print(f"  ✓ Valid JSON schema")
            print(f"  Dialogue: {parsed['dialogue'][:150]}")
            tools = parsed.get("tool_calls", [])
            if tools:
                for t in tools:
                    print(f"  Tool: {t['name']}({t.get('args', {})})")
            else:
                print(f"  Tool: (empty)")
        else:
            print(f"  ✗ Invalid output:")
            print(f"  Raw: {raw[:200]}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Valid JSON parse rate: {valid_json_count}/{len(TEST_CASES)} ({valid_json_count/len(TEST_CASES):.1%})")
    print(f"Valid schema rate: {valid_schema_count}/{len(TEST_CASES)} ({valid_schema_count/len(TEST_CASES):.1%})")

    out_path = os.path.expanduser("~/npcllm/stage2_tool_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
