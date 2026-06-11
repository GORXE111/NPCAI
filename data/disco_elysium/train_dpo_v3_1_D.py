"""
Stage 3 v3.1.D — DPO on top of Stage 2 v3.1 (2B), targeting branch/emotion gaps.

Adapted from train_dpo_v3.py (0.8B). Key changes:
- Qwen3.5-2B base (not 0.8B)
- Reference = Stage 2 v3.1 LoRA (kim_q35_2b_stage2)
- Policy = same LoRA, trainable
- LR 5e-7 (matches old DPO), Beta 0.1
- 1 epoch
- Custom DPO loss (no TRL dependency)

Memory: 2 × 2B float32 + grads + optimizer ≈ 18-22 GB
        Mac 16GB will swap heavily, slow but should fit.
"""
import json, os, random, torch, torch.nn.functional as F, sys
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-2B"
STAGE2_LORA = os.path.expanduser("~/npcllm/checkpoints/kim_q35_2b_stage2/lora")
DATA = os.path.expanduser("~/npcllm/data_kim_dpo_v3_1_D/dpo_train.jsonl")
VALID = os.path.expanduser("~/npcllm/data_kim_dpo_v3_1_D/dpo_valid.jsonl")
OUT = os.path.expanduser("~/npcllm/checkpoints/kim_q35_2b_stage3_dpo_v3_1_D")

LR = 5e-7              # Standard DPO LR (much lower than SFT 5e-5)
BETA = 0.1             # KL temperature
EPOCHS = 1
GRAD_ACCUM = 8

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def compute_logprob(model, tok, dev, system, prompt, response, max_len=512):
    """Sum of log P(response | system, prompt) over response tokens only."""
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


def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Stage 3 DPO v3.1.D (2B)")
    print(f"  LR={LR}  BETA={BETA}  EPOCHS={EPOCHS}")
    print(f"  base LoRA: {STAGE2_LORA}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print("Loading reference model (v3.1, frozen)...")
    base_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    base_ref = patch_qwen35_for_mps(base_ref)
    ref_model = PeftModel.from_pretrained(base_ref, STAGE2_LORA)
    dev = torch.device("mps")
    ref_model = ref_model.to(dev).eval()
    for p in ref_model.parameters(): p.requires_grad = False

    print("Loading policy model (v3.1, trainable)...")
    base_pol = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    base_pol = patch_qwen35_for_mps(base_pol)
    policy_model = PeftModel.from_pretrained(base_pol, STAGE2_LORA, is_trainable=True)
    policy_model = policy_model.to(dev)
    policy_model.print_trainable_parameters()

    train_data = load_jsonl(DATA)
    val_data = load_jsonl(VALID)
    print(f"Train: {len(train_data)} Val: {len(val_data)}")

    trainable = [p for p in policy_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)

    os.makedirs(OUT, exist_ok=True)
    best_val_acc = 0.0
    best_val_loss = float("inf")

    def eval_val():
        policy_model.eval()
        correct = 0; total = 0; loss_sum = 0
        with torch.no_grad():
            for s in val_data:
                lp_cp = compute_logprob(policy_model, tok, dev, s["system"], s["prompt"], s["chosen"])
                lp_rp = compute_logprob(policy_model, tok, dev, s["system"], s["prompt"], s["rejected"])
                lp_cr = compute_logprob(ref_model, tok, dev, s["system"], s["prompt"], s["chosen"])
                lp_rr = compute_logprob(ref_model, tok, dev, s["system"], s["prompt"], s["rejected"])
                if any(x is None for x in [lp_cp, lp_rp, lp_cr, lp_rr]): continue
                pi_logr = lp_cp - lp_rp
                ref_logr = lp_cr - lp_rr
                logits = BETA * (pi_logr - ref_logr)
                loss = -F.logsigmoid(logits)
                loss_sum += loss.item()
                if logits > 0: correct += 1
                total += 1
        return loss_sum / max(total, 1), correct / max(total, 1)

    # Baseline eval (before training: 50% chance, since policy == ref)
    vloss, vacc = eval_val()
    print(f"Initial: Val Loss={vloss:.4f}  Val Acc={vacc:.3f}  (baseline ~0.5)")
    # Save initial checkpoint as baseline
    policy_model.save_pretrained(os.path.join(OUT, "lora"))
    tok.save_pretrained(os.path.join(OUT, "lora"))

    eval_every = max(len(train_data) // 4, 30)
    step_in_accum = 0

    for ep in range(EPOCHS):
        policy_model.train()
        random.shuffle(train_data)
        cum_loss = 0; n_done = 0
        opt.zero_grad()

        for step, s in enumerate(train_data):
            try:
                lp_cp = compute_logprob(policy_model, tok, dev, s["system"], s["prompt"], s["chosen"])
                lp_rp = compute_logprob(policy_model, tok, dev, s["system"], s["prompt"], s["rejected"])
                with torch.no_grad():
                    lp_cr = compute_logprob(ref_model, tok, dev, s["system"], s["prompt"], s["chosen"])
                    lp_rr = compute_logprob(ref_model, tok, dev, s["system"], s["prompt"], s["rejected"])
                if any(x is None for x in [lp_cp, lp_rp, lp_cr, lp_rr]):
                    continue
                pi_logr = lp_cp - lp_rp
                ref_logr = lp_cr - lp_rr
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
            except RuntimeError as e:
                print(f"  step {step}: {e}")
                opt.zero_grad(); step_in_accum = 0
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
                    policy_model.save_pretrained(os.path.join(OUT, "lora"))
                    tok.save_pretrained(os.path.join(OUT, "lora"))
                    print(f"    -> Best! (acc {vacc:.3f})")
                policy_model.train()

        # End-of-epoch eval
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
            policy_model.save_pretrained(os.path.join(OUT, "lora"))
            tok.save_pretrained(os.path.join(OUT, "lora"))
            print("  -> Best!")

    print(f"\nDone! Best Val Acc: {best_val_acc:.3f}  Best Val Loss: {best_val_loss:.4f}")
    return best_val_acc

if __name__ == "__main__":
    train()
