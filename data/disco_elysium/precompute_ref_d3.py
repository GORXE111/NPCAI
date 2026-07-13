"""
Phase 1 of Offline DPO v3.1.D:
Load REFERENCE model (v3.1 LoRA, frozen) ONLY, compute logprob(chosen) and
logprob(rejected) for every pair, save to disk. Free ref before training.

Memory: only 1× 2B fp32 (~7.5 GB) — fits in 16GB Mac comfortably.
"""
import json, os, torch, torch.nn.functional as F, sys
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-2B"
# D3: reference is the WARM-START LoRA (has non-zero set_expression/present_choices)
STAGE2_LORA = os.path.expanduser("~/npcllm/checkpoints/kim_q35_2b_stage3_warmstart_d3/lora")
DATA_DIR = os.path.expanduser("~/npcllm/data_kim_dpo_v3_1_D2")
OUT_FILE = os.path.join(DATA_DIR, "ref_logprobs_d3.json")

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def compute_logprob(model, tok, dev, system, prompt, response, max_len=512):
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prefix_text = tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)

    full_ids = tok(full_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
    prefix_ids = tok(prefix_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)

    prefix_len = prefix_ids.shape[1]
    if full_ids.shape[1] <= prefix_len: return None

    with torch.no_grad():
        outputs = model(input_ids=full_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]
    target_start = max(prefix_len - 1, 0)
    target_logits = shift_logits[:, target_start:]
    target_labels = shift_labels[:, target_start:]
    log_probs = F.log_softmax(target_logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Phase 1 (Offline DPO): precomputing reference logprobs")
    print(f"  ref LoRA: {STAGE2_LORA}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    base = patch_qwen35_for_mps(base)
    ref = PeftModel.from_pretrained(base, STAGE2_LORA)
    dev = torch.device("mps")
    ref = ref.to(dev).eval()
    for p in ref.parameters(): p.requires_grad = False
    print("Ref model loaded.")

    results = {"train": [], "valid": []}
    for split_name, fname in [("train", "dpo_train.jsonl"), ("valid", "dpo_valid.jsonl")]:
        data = load_jsonl(os.path.join(DATA_DIR, fname))
        print(f"\nProcessing {split_name}: {len(data)} pairs")
        for i, s in enumerate(data):
            lp_c = compute_logprob(ref, tok, dev, s["system"], s["prompt"], s["chosen"])
            lp_r = compute_logprob(ref, tok, dev, s["system"], s["prompt"], s["rejected"])
            results[split_name].append({"lp_chosen": lp_c, "lp_rejected": lp_r})
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(data)}")
            # Periodic memory cleanup
            if (i + 1) % 30 == 0:
                torch.mps.empty_cache()

    with open(OUT_FILE, "w") as f:
        json.dump(results, f)
    print(f"\nSaved precomputed ref logprobs to {OUT_FILE}")
    print(f"  train: {len(results['train'])} pairs")
    print(f"  valid: {len(results['valid'])} pairs")

if __name__ == "__main__":
    main()
