"""
Stage 1: Train LoRA adapters for NPC personality.
Freeze: base model + memory module + emotion head
Train: LoRA parameters only
"""
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from npc_model import NPCModel

# ── Config ──────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # 3B: ~6GB float16, fits M4 16GB
import platform
if platform.system() == "Darwin":
    DATA_DIR = os.path.expanduser("~/npcllm/training_data/_combined")
    OUTPUT_DIR = os.path.expanduser("~/npcllm/checkpoints/stage1_3b")
else:
    DATA_DIR = "D:/AIproject/npcllm_paper/training_data/_combined"
    OUTPUT_DIR = "D:/AIproject/npcllm_paper/checkpoints/stage1"
EPOCHS = 5
LR = 2e-4
BATCH_SIZE = 2
MAX_SEQ_LEN = 256
GRAD_ACCUM = 4


class NPCDialogueDataset(Dataset):
    """Load JSONL chat data for causal LM training."""

    def __init__(self, filepath, tokenizer, max_length=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                messages = data["messages"]
                # Format as chat template
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                self.samples.append(text)

        print(f"Loaded {len(self.samples)} samples from {filepath}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        # Labels = input_ids (causal LM), mask padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def train():
    print("=" * 60)
    print("Stage 1: LoRA Training")
    print("=" * 60)

    # Build model
    model = NPCModel(
        model_name=MODEL_NAME,
        memory_dim=256,
        num_memory_layers=4,
        lora_rank=8,
        load_in_4bit=False  # float16 for 0.5B
    )
    model.freeze_for_stage(1)

    tokenizer = model.tokenizer

    # Move to device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model.base_model = model.base_model.to(device)
        print(f"Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = next(model.base_model.parameters()).device
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Dataset
    train_ds = NPCDialogueDataset(
        os.path.join(DATA_DIR, "train.jsonl"), tokenizer, MAX_SEQ_LEN
    )
    val_ds = NPCDialogueDataset(
        os.path.join(DATA_DIR, "valid.jsonl"), tokenizer, MAX_SEQ_LEN
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Optimizer (only LoRA params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Training loop
    print(f"\nTraining: {EPOCHS} epochs, {len(train_ds)} samples, "
          f"batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")

    best_val_loss = float('inf')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.base_model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            total_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 10 == 0:
                avg = total_loss / (step + 1)
                print(f"  Epoch {epoch+1} Step {step+1}/{len(train_loader)} | Loss: {avg:.4f}")

        train_avg = total_loss / len(train_loader)

        # Validation
        model.base_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

        val_avg = val_loss / max(len(val_loader), 1)
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Train Loss: {train_avg:.4f} | Val Loss: {val_avg:.4f}")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            model.save_npc_modules(OUTPUT_DIR)
            print(f"  -> Best model saved! Val Loss: {val_avg:.4f}")

    print(f"\nTraining complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
