"""
Kim Persona Reproduction Test.

Run a small set of canonical Disco Elysium scenarios through:
  - Base Qwen3.5-0.8B  (no Kim training)
  - Stage 1 LoRA       (Kim persona SFT)

Side-by-side comparison; eyeball whether Stage 1 actually captures Kim's voice.
This is a smoke test before we commit to building Stage 2/3 on top.
"""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
LORA_PATH = os.path.expanduser("~/npcllm/checkpoints/kim_q35_08b/lora")

KIM_SYSTEM = """You are Kim Kitsuragi, a 43-year-old lieutenant from RCM Precinct 41 in Revachol. You are partnered with a deeply troubled detective on a murder investigation. Your speech style is clinical, dry, and morally anchored — measured pauses, careful word choice, and quiet wit. You frequently say things like "Let me make a note of that", "Detective, please", "That is... an interesting approach". You are observant but reserved, and you protect your partner from his worst impulses without being preachy. Reply only with what Kim would say or do — do not break character."""

# Canonical DE-style probes
TEST_CASES = [
    {
        "name": "introduction",
        "context": "[Scene begins. Detective approaches Kim.]\nDetective: \"Who are you?\"",
    },
    {
        "name": "absurd_confession",
        "context": "Detective: \"I have something to confess, Kim. I am a Vampyr.\"",
    },
    {
        "name": "drug_offer",
        "context": "Detective: \"Want some of these pills I found? They might help.\"",
    },
    {
        "name": "introspection",
        "context": "Detective: \"Kim... what keeps you going on a job like this?\"",
    },
    {
        "name": "evidence_check",
        "context": "[At a crime scene. Detective examines the corpse.]\nDetective: \"Look at the bruises on his neck, Kim. What do you see?\"",
    },
    {
        "name": "anti_authority",
        "context": "Detective: \"Maybe we should just let this case go. Who cares about a dead mercenary?\"",
    },
    {
        "name": "empathy_ask",
        "context": "Detective: \"Kim, do you ever miss your old partner?\"",
    },
    {
        "name": "modern_break_test",
        "context": "Detective: \"What year is it in real life?\"",
    },
    {
        "name": "ai_break_test",
        "context": "Detective: \"Are you actually a language model?\"",
    },
    {
        "name": "small_talk",
        "context": "Detective: \"How's the weather treating you?\"",
    },
]

def load(use_lora):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    model = patch_qwen35_for_mps(model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if use_lora and os.path.exists(LORA_PATH):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_PATH)
        print(f"  Loaded LoRA from {LORA_PATH}")
    dev = torch.device("mps")
    model = model.to(dev).eval()
    return tok, model, dev

def generate(tok, model, dev, system, user_ctx, max_new=80):
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_ctx},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False,
            temperature=1.0, pad_token_id=tok.eos_token_id, repetition_penalty=1.1,
        )
    response = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    # Clean up thinking + role markers
    import re
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<think>.*', '', response, flags=re.DOTALL)
    response = re.sub(r'(?:^|\n)\s*(?:assistant|user|system)\s*\n', '\n', response, flags=re.IGNORECASE)
    return response.strip()

def main():
    out = {"base": [], "lora": []}

    print("=" * 70)
    print("Loading Base Qwen3.5-0.8B (no Kim training)")
    print("=" * 70)
    tok, base, dev = load(use_lora=False)
    for tc in TEST_CASES:
        resp = generate(tok, base, dev, KIM_SYSTEM, tc["context"])
        out["base"].append({"name": tc["name"], "context": tc["context"], "response": resp})
        print(f"\n--- [{tc['name']}] ---")
        print(f"USER: {tc['context'][:120]}")
        print(f"BASE: {resp[:200]}")
    del base
    torch.mps.empty_cache()

    print("\n" + "=" * 70)
    print("Loading Stage 1 LoRA (Kim persona)")
    print("=" * 70)
    tok, lora, dev = load(use_lora=True)
    for tc in TEST_CASES:
        resp = generate(tok, lora, dev, KIM_SYSTEM, tc["context"])
        out["lora"].append({"name": tc["name"], "context": tc["context"], "response": resp})
        print(f"\n--- [{tc['name']}] ---")
        print(f"USER: {tc['context'][:120]}")
        print(f"LORA: {resp[:200]}")

    # Save
    out_path = os.path.expanduser("~/npcllm/kim_persona_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Print side-by-side summary
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE SUMMARY")
    print("=" * 70)
    for i, tc in enumerate(TEST_CASES):
        print(f"\n### [{tc['name']}]")
        print(f"USER: {tc['context'][:100]}")
        print(f"  BASE: {out['base'][i]['response'][:150]}")
        print(f"  LORA: {out['lora'][i]['response'][:150]}")

if __name__ == "__main__":
    main()
