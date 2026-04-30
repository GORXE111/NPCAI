"""
Stage 3: DPO on tool faithfulness, on top of Stage 2 LoRA.

Implementation: Custom DPO loss (avoids trl dependency hassles on MPS).
- Reference model = Stage 2 LoRA (frozen)
- Policy model   = Stage 2 LoRA copy (LoRA params trainable)
- Loss = -log sigmoid(beta * (logp_chosen_policy - logp_chosen_ref - logp_rejected_policy + logp_rejected_ref))
"""
import json, os, random, torch, torch.nn.functional as F, sys
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
STAGE2_LORA = os.path.expanduser("~/npcllm/checkpoints/kim_q35_08b_stage2/lora")
DATA = os.path.expanduser("~/npcllm/data_kim_v2/synthetic_train.jsonl")
VALID = os.path.expanduser("~/npcllm/data_kim_v2/synthetic_valid.jsonl")
OUT = os.path.expanduser("~/npcllm/checkpoints/kim_q35_08b_stage3")

LR = 5e-7              # Standard DPO LR (much lower than SFT 5e-5)
BETA = 0.1             # KL temperature
EPOCHS = 1             # DPO usually 1 epoch
GRAD_ACCUM = 8

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def compute_logprob(model, tok, dev, system, prompt, response, max_len=512):
    """Compute sum of log P(response | system, prompt) for the response tokens only."""
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    # Compute the assistant span by tokenizing prefix
    msgs_pre = msgs[:2]
    prefix_text = tok.apply_chat_template(msgs_pre, tokenize=False, add_generation_prompt=True)

    full_ids = tok(full_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)
    prefix_ids = tok(prefix_text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(dev)

    prefix_len = prefix_ids.shape[1]
    if full_ids.shape[1] <= prefix_len: return None

    # Compute logits
    with torch.set_grad_enabled(model.training):
        outputs = model(input_ids=full_ids)
        logits = outputs.logits  # (1, T, V)

    # Shift: predict token at position t+1 from logits at position t
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]

    # Mask: only count tokens AFTER prefix_len
    # The token at position prefix_len is predicted from logits at prefix_len-1
    target_start = prefix_len - 1
    if target_start < 0: target_start = 0
    target_logits = shift_logits[:, target_start:]
    target_labels = shift_labels[:, target_start:]

    log_probs = F.log_softmax(target_logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum()


def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Stage 3 DPO: lr={LR} beta={BETA} epochs={EPOCHS}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # === Reference model (Stage 2 frozen) ===
    print("Loading reference model (Stage 2 frozen)...")
    base_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    base_ref = patch_qwen35_for_mps(base_ref)
    ref_model = PeftModel.from_pretrained(base_ref, STAGE2_LORA)
    dev = torch.device("mps")
    ref_model = ref_model.to(dev).eval()
    for p in ref_model.parameters(): p.requires_grad = False

    # === Policy model (Stage 2 trainable) ===
    print("Loading policy model (Stage 2 trainable)...")
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
    best_val_acc = 0
    best_val_loss = float("inf")

    def eval_val():
        policy_model.eval()
        correct = 0; total = 0; loss_sum = 0
        with torch.no_grad():
            for s in val_data:
                lp_chosen_pol = compute_logprob(policy_model, tok, dev, s["system"], s["prompt"], s["chosen"])
                lp_rejected_pol = compute_logprob(policy_model, tok, dev, s["system"], s["prompt"], s["rejected"])
                lp_chosen_ref = compute_logprob(ref_model, tok, dev, s["system"], s["prompt"], s["chosen"])
                lp_rejected_ref = compute_logprob(ref_model, tok, dev, s["system"], s["prompt"], s["rejected"])
                if any(x is None for x in [lp_chosen_pol, lp_rejected_pol, lp_chosen_ref, lp_rejected_ref]): continue
                pi_logratios = lp_chosen_pol - lp_rejected_pol
                ref_logratios = lp_chosen_ref - lp_rejected_ref
                logits = BETA * (pi_logratios - ref_logratios)
                loss = -F.logsigmoid(logits)
                loss_sum += loss.item()
                if logits > 0: correct += 1
                total += 1
        return loss_sum / max(total,1), correct / max(total,1)

    eval_every = len(train_data) // 4
    step_in_accum = 0

    for ep in range(EPOCHS):
        policy_model.train()
        random.shuffle(train_data)
        cum_loss = 0; n_done = 0
        opt.zero_grad()

        for step, s in enumerate(train_data):
            try:
                lp_chosen_pol = compute_logprob(policy_model, tok, dev, s["system"], s["prompt"], s["chosen"])
                lp_rejected_pol = compute_logprob(policy_model, tok, dev, s["system"], s["prompt"], s["rejected"])
                with torch.no_grad():
                    lp_chosen_ref = compute_logprob(ref_model, tok, dev, s["system"], s["prompt"], s["chosen"])
                    lp_rejected_ref = compute_logprob(ref_model, tok, dev, s["system"], s["prompt"], s["rejected"])
                if any(x is None for x in [lp_chosen_pol, lp_rejected_pol, lp_chosen_ref, lp_rejected_ref]):
                    continue
                pi_logratios = lp_chosen_pol - lp_rejected_pol
                ref_logratios = lp_chosen_ref - lp_rejected_ref
                logits = BETA * (pi_logratios - ref_logratios)
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

            except Exception as e:
                print(f"  Step {step} error: {str(e)[:80]}")
                opt.zero_grad()
                step_in_accum = 0
                continue

            if (step+1) % 100 == 0:
                print(f"  E{ep+1} {step+1}/{len(train_data)} L:{cum_loss/max(n_done,1):.4f}")

            if (step+1) % eval_every == 0:
                v_loss, v_acc = eval_val()
                print(f"  E{ep+1} Step {step+1} | Val Loss: {v_loss:.4f} | Val Acc (chosen>rejected): {v_acc:.2%}")
                if v_acc > best_val_acc:
                    best_val_acc = v_acc
                    best_val_loss = v_loss
                    policy_model.save_pretrained(os.path.join(OUT, "lora"))
                    tok.save_pretrained(os.path.join(OUT, "lora"))
                    print(f"    -> Best! ({v_acc:.2%})")
                policy_model.train()

        # End epoch eval
        v_loss, v_acc = eval_val()
        print(f"E{ep+1}/{EPOCHS} | Train: {cum_loss/max(n_done,1):.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2%}")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_val_loss = v_loss
            policy_model.save_pretrained(os.path.join(OUT, "lora"))
            tok.save_pretrained(os.path.join(OUT, "lora"))
            print("  -> Best!")

    print(f"Done! Best Val Acc: {best_val_acc:.2%}, Best Val Loss: {best_val_loss:.4f}")
    return best_val_acc

if __name__ == "__main__":
    train()
