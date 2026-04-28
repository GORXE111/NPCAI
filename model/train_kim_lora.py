"""
Stage 1 LoRA on Qwen3.5-0.8B for Kim Kitsuragi (Disco Elysium).
Conservative hyperparameters based on v2 lessons learned.
"""
import json, os, random, torch, torch.nn as nn, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
BASE_DIR = os.path.expanduser("~/npcllm")
TRAIN = os.path.join(BASE_DIR, "data_kim/kim_train.jsonl")
VALID = os.path.join(BASE_DIR, "data_kim/kim_valid.jsonl")
OUT = os.path.join(BASE_DIR, "checkpoints/kim_q35_08b")

# Conservative hyperparameters (from v2 lessons)
LR = 5e-5
DROPOUT = 0.15
LORA_R = 16
EPOCHS = 4
PATIENCE = 2
WEIGHT_DECAY = 0.05

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"Kim SFT: Qwen3.5-0.8B | lr={LR} dropout={DROPOUT} r={LORA_R} epochs={EPOCHS}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    model = patch_qwen35_for_mps(model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_R*2,
        lora_dropout=DROPOUT,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    dev = torch.device("mps")
    model = model.to(dev)

    train_data = load_jsonl(TRAIN)
    val_data = load_jsonl(VALID)
    print(f"Train: {len(train_data)} Val: {len(val_data)}")

    def encode(d, ml=384):
        text = tok.apply_chat_template(d["messages"], tokenize=False, add_generation_prompt=False)
        enc = tok(text, truncation=True, max_length=ml, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        lb = ids.clone()
        lb[mask==0] = -100
        return ids, mask, lb

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    patience = 0
    os.makedirs(OUT, exist_ok=True)

    def eval_val():
        model.eval()
        tl = 0; n = 0
        with torch.no_grad():
            for d in val_data:
                ids, mask, lb = encode(d)
                ids = ids.unsqueeze(0).to(dev); mask = mask.unsqueeze(0).to(dev); lb = lb.unsqueeze(0).to(dev)
                o = model(input_ids=ids, attention_mask=mask, labels=lb)
                if not torch.isnan(o.loss): tl += o.loss.item(); n += 1
        return tl / max(n,1)

    eval_every = len(train_data) // 2

    for ep in range(EPOCHS):
        model.train()
        random.shuffle(train_data)
        tl = 0; opt.zero_grad()
        for step, d in enumerate(train_data):
            ids, mask, lb = encode(d)
            ids = ids.unsqueeze(0).to(dev); mask = mask.unsqueeze(0).to(dev); lb = lb.unsqueeze(0).to(dev)
            o = model(input_ids=ids, attention_mask=mask, labels=lb)
            if torch.isnan(o.loss):
                opt.zero_grad(); continue
            (o.loss / 8).backward()
            tl += o.loss.item()
            if (step+1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
            if (step+1) % 200 == 0:
                print(f"  E{ep+1} {step+1}/{len(train_data)} L:{tl/(step+1):.4f}")

            if (step+1) % eval_every == 0:
                v = eval_val()
                print(f"  E{ep+1} Step {step+1} | Val: {v:.4f}")
                if v < best_val:
                    best_val = v
                    patience = 0
                    model.save_pretrained(os.path.join(OUT, "lora"))
                    tok.save_pretrained(os.path.join(OUT, "lora"))
                    print(f"    -> Best! ({v:.4f})")
                else:
                    patience += 1
                    print(f"    No improvement (patience {patience}/{PATIENCE})")
                    if patience >= PATIENCE:
                        print("  EARLY STOP")
                        print(f"Done! Best Val: {best_val:.4f}")
                        return best_val
                model.train()

        t_avg = tl / len(train_data)
        v = eval_val()
        print(f"E{ep+1}/{EPOCHS} | Train: {t_avg:.4f} | Val: {v:.4f}")
        if v < best_val:
            best_val = v
            patience = 0
            model.save_pretrained(os.path.join(OUT, "lora"))
            tok.save_pretrained(os.path.join(OUT, "lora"))
            print("  -> Best!")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  EARLY STOP"); break

    print(f"Done! Best Val: {best_val:.4f}")
    return best_val

if __name__ == "__main__":
    train()
