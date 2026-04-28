"""
Stage 2: Tool-augmented SFT on Qwen3.5-0.8B + Stage 1 v2 LoRA.
Continue training from Stage 1 v2 LoRA, on the JSON tool-call format.
"""
import json, os, random, torch, sys
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
STAGE1_LORA = os.path.expanduser("~/npcllm/checkpoints/kim_q35_08b_v2/lora")
DATA_DIR = os.path.expanduser("~/npcllm/data_kim_v2")
OUT = os.path.expanduser("~/npcllm/checkpoints/kim_q35_08b_stage2")

LR = 5e-5
DROPOUT = 0.15
EPOCHS = 4
PATIENCE = 2
WEIGHT_DECAY = 0.05

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Stage 2: Tool-augmented SFT on top of Stage 1 v2")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    model = patch_qwen35_for_mps(model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Load Stage 1 v2 LoRA, then make it trainable to continue
    model = PeftModel.from_pretrained(model, STAGE1_LORA, is_trainable=True)
    model.print_trainable_parameters()
    dev = torch.device("mps")
    model = model.to(dev)

    train_data = load_jsonl(os.path.join(DATA_DIR, "kim_tool_train.jsonl"))
    val_data = load_jsonl(os.path.join(DATA_DIR, "kim_tool_valid.jsonl"))
    print(f"Train: {len(train_data)} Val: {len(val_data)}")

    def encode(d, ml=512):
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
            if patience >= PATIENCE: break

    print(f"Done! Best Val: {best_val:.4f}")
    return best_val

if __name__ == "__main__":
    train()
