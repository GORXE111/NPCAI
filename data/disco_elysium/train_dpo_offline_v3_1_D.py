"""
Phase 2 of Offline DPO v3.1.D:
Load POLICY model only, train using precomputed ref logprobs from disk.

Memory: only 1× 2B fp32 + LoRA grads + optimizer (~10 GB) — fits comfortably.

DPO loss (with precomputed ref):
  loss = -log_sigmoid(beta * (lp_chosen_policy - lp_rejected_policy
                             - (lp_chosen_ref - lp_rejected_ref)))
"""
import json, os, random, torch, torch.nn.functional as F, sys
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-2B"
STAGE2_LORA = os.path.expanduser("~/npcllm/checkpoints/kim_q35_2b_stage2/lora")
DATA_DIR = os.path.expanduser("~/npcllm/data_kim_dpo_v3_1_D")
REF_LP_FILE = os.path.join(DATA_DIR, "ref_logprobs.json")
OUT = os.path.expanduser("~/npcllm/checkpoints/kim_q35_2b_stage3_dpo_v3_1_D")

LR = 5e-7
BETA = 0.1
EPOCHS = 1
GRAD_ACCUM = 8

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
    with torch.set_grad_enabled(model.training):
        outputs = model(input_ids=full_ids)
        logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]
    target_start = max(prefix_len - 1, 0)
    target_logits = shift_logits[:, target_start:]
    target_labels = shift_labels[:, target_start:]
    log_probs = F.log_softmax(target_logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum()

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Phase 2 (Offline DPO): training policy with precomputed ref logprobs")
    print(f"  LR={LR}  BETA={BETA}  EPOCHS={EPOCHS}")

    # Load precomputed ref logprobs
    if not os.path.exists(REF_LP_FILE):
        raise RuntimeError(f"Missing precomputed ref file: {REF_LP_FILE}. Run precompute_ref_dpo_v3_1_D.py first.")
    ref_lp = json.load(open(REF_LP_FILE))
    print(f"Loaded ref logprobs: train={len(ref_lp['train'])}  valid={len(ref_lp['valid'])}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print("Loading policy model (trainable LoRA from v3.1)...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    base = patch_qwen35_for_mps(base)
    policy = PeftModel.from_pretrained(base, STAGE2_LORA, is_trainable=True)
    policy.print_trainable_parameters()
    dev = torch.device("mps")
    policy = policy.to(dev)

    train_data = load_jsonl(os.path.join(DATA_DIR, "dpo_train.jsonl"))
    val_data = load_jsonl(os.path.join(DATA_DIR, "dpo_valid.jsonl"))
    print(f"Train: {len(train_data)} Val: {len(val_data)}")

    # Attach ref logprobs by index
    for i, s in enumerate(train_data):
        s["lp_chosen_ref"] = ref_lp["train"][i]["lp_chosen"]
        s["lp_rejected_ref"] = ref_lp["train"][i]["lp_rejected"]
    for i, s in enumerate(val_data):
        s["lp_chosen_ref"] = ref_lp["valid"][i]["lp_chosen"]
        s["lp_rejected_ref"] = ref_lp["valid"][i]["lp_rejected"]

    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)

    os.makedirs(OUT, exist_ok=True)
    best_val_acc = 0.0
    best_val_loss = float("inf")

    def eval_val():
        policy.eval()
        correct = 0; total = 0; loss_sum = 0
        with torch.no_grad():
            for s in val_data:
                if s["lp_chosen_ref"] is None or s["lp_rejected_ref"] is None: continue
                lp_cp = compute_logprob(policy, tok, dev, s["system"], s["prompt"], s["chosen"])
                lp_rp = compute_logprob(policy, tok, dev, s["system"], s["prompt"], s["rejected"])
                if lp_cp is None or lp_rp is None: continue
                pi_logr = lp_cp - lp_rp
                ref_logr = s["lp_chosen_ref"] - s["lp_rejected_ref"]
                logits = BETA * (pi_logr.item() - ref_logr)
                loss = -F.logsigmoid(torch.tensor(logits))
                loss_sum += loss.item()
                if logits > 0: correct += 1
                total += 1
        return loss_sum / max(total, 1), correct / max(total, 1)

    vloss, vacc = eval_val()
    print(f"Initial: Val Loss={vloss:.4f}  Val Acc={vacc:.3f}  (baseline policy==ref, expected ~0.5)")
    policy.save_pretrained(os.path.join(OUT, "lora"))
    tok.save_pretrained(os.path.join(OUT, "lora"))
    best_val_acc = vacc
    best_val_loss = vloss

    eval_every = max(len(train_data) // 4, 30)
    step_in_accum = 0

    for ep in range(EPOCHS):
        policy.train()
        random.shuffle(train_data)
        cum_loss = 0; n_done = 0
        opt.zero_grad()

        for step, s in enumerate(train_data):
            if s["lp_chosen_ref"] is None or s["lp_rejected_ref"] is None: continue
            try:
                lp_cp = compute_logprob(policy, tok, dev, s["system"], s["prompt"], s["chosen"])
                lp_rp = compute_logprob(policy, tok, dev, s["system"], s["prompt"], s["rejected"])
                if lp_cp is None or lp_rp is None: continue
                pi_logr = lp_cp - lp_rp
                ref_logr = s["lp_chosen_ref"] - s["lp_rejected_ref"]
                logits = BETA * (pi_logr - ref_logr)
                loss = -F.logsigmoid(logits) / GRAD_ACCUM
                loss.backward()
                cum_loss += -F.logsigmoid(logits).item()
                n_done += 1
                step_in_accum += 1

                if step_in_accum >= GRAD_ACCUM:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                    opt.zero_grad()
                    step_in_accum = 0
                    if (step+1) % (GRAD_ACCUM * 4) == 0:
                        torch.mps.empty_cache()
            except RuntimeError as e:
                print(f"  step {step}: {e}")
                opt.zero_grad(); step_in_accum = 0
                torch.mps.empty_cache()
                continue

            if (step + 1) % 30 == 0:
                avg_loss = cum_loss / max(n_done, 1)
                print(f"  E{ep+1} {step+1}/{len(train_data)} L:{avg_loss:.4f}")

            if (step + 1) % eval_every == 0:
                vloss, vacc = eval_val()
                print(f"  E{ep+1} Step {step+1} | Val Loss={vloss:.4f}  Val Acc={vacc:.3f}")
                if vacc > best_val_acc:
                    best_val_acc = vacc
                    best_val_loss = vloss
                    policy.save_pretrained(os.path.join(OUT, "lora"))
                    tok.save_pretrained(os.path.join(OUT, "lora"))
                    print(f"    -> Best! (acc {vacc:.3f})")
                policy.train()

        if step_in_accum > 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad()
            step_in_accum = 0
        vloss, vacc = eval_val()
        avg_loss = cum_loss / max(n_done, 1)
        print(f"E{ep+1}/{EPOCHS} | Train L:{avg_loss:.4f} | Val Loss={vloss:.4f}  Val Acc={vacc:.3f}")
        if vacc > best_val_acc:
            best_val_acc = vacc
            best_val_loss = vloss
            policy.save_pretrained(os.path.join(OUT, "lora"))
            tok.save_pretrained(os.path.join(OUT, "lora"))
            print("  -> Best!")

    print(f"\nDone! Best Val Acc: {best_val_acc:.3f}  Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
