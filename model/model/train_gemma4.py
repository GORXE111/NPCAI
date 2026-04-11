import json, os, torch, sys

MODEL_NAME = 'google/gemma-4-E4B-it'
DATA_DIR = os.path.expanduser('~/npcllm/training_data/_combined')
OUTPUT_DIR = os.path.expanduser('~/npcllm/checkpoints/stage1_gemma4')
EPOCHS = 5
LR = 2e-4
BATCH_SIZE = 1
MAX_SEQ_LEN = 256
GRAD_ACCUM = 8

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def unwrap_clippable_linear(model):
    """Replace all Gemma4ClippableLinear with their inner nn.Linear."""
    replaced = 0
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if 'ClippableLinear' in type(child).__name__ and hasattr(child, 'linear'):
                setattr(module, attr_name, child.linear)
                replaced += 1
    print(f'Unwrapped {replaced} ClippableLinear modules -> nn.Linear')
    return model

class NPCDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, tokenizer, max_length=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                text = tokenizer.apply_chat_template(data['messages'], tokenize=False, add_generation_prompt=False)
                self.samples.append(text)
        print(f'Loaded {len(self.samples)} samples')

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.samples[idx], truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def train():
    print('=' * 60)
    print(f'Stage 1: LoRA - {MODEL_NAME} on MPS')
    print('=' * 60)

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Unwrap ClippableLinear so peft can handle them
    model = unwrap_clippable_linear(model)

    print('Applying LoRA...')
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        model = model.to(device)
        print('Using MPS')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    train_ds = NPCDataset(os.path.join(DATA_DIR, 'train.jsonl'), tokenizer, MAX_SEQ_LEN)
    val_ds = NPCDataset(os.path.join(DATA_DIR, 'valid.jsonl'), tokenizer, MAX_SEQ_LEN)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

    print(f'Training: {EPOCHS} epochs, {len(train_ds)} samples')

    best_val = float('inf')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            total_loss += out.loss.item()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % 20 == 0:
                print(f'  Epoch {epoch+1} Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}')

        train_avg = total_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
                val_loss += out.loss.item()
        val_avg = val_loss / max(len(val_loader), 1)

        print(f'Epoch {epoch+1}/{EPOCHS} | Train: {train_avg:.4f} | Val: {val_avg:.4f}')
        if val_avg < best_val:
            best_val = val_avg
            model.save_pretrained(os.path.join(OUTPUT_DIR, 'lora'))
            tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'lora'))
            print(f'  -> Best! Val: {val_avg:.4f}')

    print(f'Done! Best Val: {best_val:.4f}')

if __name__ == '__main__':
    train()
