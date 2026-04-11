import json, os, random, torch, torch.nn as nn

MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
STAGE1_CKPT = os.path.expanduser('~/npcllm/checkpoints/stage1_qwen35_2b_large/lora')
OUTPUT_DIR = os.path.expanduser('~/npcllm/checkpoints/stage2_memory_large')
DATA_FILE = os.path.expanduser('~/npcllm/training_data/_memory_large/train.json')
VAL_FILE = os.path.expanduser('~/npcllm/training_data/_memory_large/valid.json')
EPOCHS = 20
LR = 5e-4
NUM_MEM_TOKENS = 8
GRAD_ACCUM = 4

class MemoryPrefixEncoder(nn.Module):
    def __init__(self, embed_dim, num_tokens=8, mem_dim=256):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(embed_dim, mem_dim)
        self.to_prefix = nn.Sequential(nn.Linear(mem_dim, embed_dim * num_tokens), nn.GELU())
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, memory_embeds):
        pooled = memory_embeds.mean(dim=2)
        combined = pooled.mean(dim=1)
        mem_vec = self.proj(combined)
        prefix = self.to_prefix(mem_vec)
        prefix = prefix.view(-1, self.num_tokens, memory_embeds.shape[-1])
        return prefix * torch.sigmoid(self.gate)

def train():
    print('Stage 2: Memory Prefix - Large Data')
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    base = base.to(device).eval()
    base.enable_input_require_grads()
    for p in base.parameters():
        p.requires_grad = False

    mem_enc = MemoryPrefixEncoder(base.config.hidden_size, NUM_MEM_TOKENS, 256).to(device)
    print('Params: ' + str(sum(p.numel() for p in mem_enc.parameters())))

    with open(DATA_FILE) as f:
        train_d = json.load(f)
    with open(VAL_FILE) as f:
        val_d = json.load(f)
    print('Train: ' + str(len(train_d)) + ', Val: ' + str(len(val_d)))

    opt = torch.optim.AdamW(mem_enc.parameters(), lr=LR, weight_decay=0.01)
    embed_layer = base.model.embed_tokens
    best_val = float('inf')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        mem_enc.train()
        random.shuffle(train_d)
        total_loss = 0
        opt.zero_grad()

        for step, s in enumerate(train_d):
            mem_text = '\n\nMemories:\n' + '\n'.join('- ' + m for m in s['m']) if s['m'] else ''
            msgs = [{'role':'system','content':s['s']+mem_text},{'role':'user','content':s['u']},{'role':'assistant','content':s['a']}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            enc = tokenizer(text, truncation=True, max_length=248, padding='max_length', return_tensors='pt').to(device)

            with torch.no_grad():
                inp_emb = embed_layer(enc['input_ids'])

            if s['m']:
                me = tokenizer(s['m'], padding=True, truncation=True, max_length=32, return_tensors='pt').to(device)
                with torch.no_grad():
                    me_emb = embed_layer(me['input_ids'])
                prefix = mem_enc(me_emb.unsqueeze(0))
            else:
                prefix = torch.zeros(1, NUM_MEM_TOKENS, inp_emb.shape[-1], device=device, dtype=inp_emb.dtype)

            combined = torch.cat([prefix, inp_emb], dim=1)
            pmask = torch.ones(1, NUM_MEM_TOKENS, device=device, dtype=enc['attention_mask'].dtype)
            cmask = torch.cat([pmask, enc['attention_mask']], dim=1)
            labels = enc['input_ids'].clone()
            labels[enc['attention_mask'] == 0] = -100
            plabels = torch.full((1, NUM_MEM_TOKENS), -100, device=device, dtype=labels.dtype)
            clabels = torch.cat([plabels, labels], dim=1)

            out = base(inputs_embeds=combined, attention_mask=cmask, labels=clabels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            total_loss += out.loss.item()

            if (step + 1) % GRAD_ACCUM == 0 or step == len(train_d) - 1:
                torch.nn.utils.clip_grad_norm_(mem_enc.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        t_avg = total_loss / len(train_d)

        mem_enc.eval()
        vl = 0
        with torch.no_grad():
            for s in val_d:
                mem_text = '\n\nMemories:\n' + '\n'.join('- ' + m for m in s['m']) if s['m'] else ''
                msgs = [{'role':'system','content':s['s']+mem_text},{'role':'user','content':s['u']},{'role':'assistant','content':s['a']}]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                enc = tokenizer(text, truncation=True, max_length=248, padding='max_length', return_tensors='pt').to(device)
                inp_emb = embed_layer(enc['input_ids'])

                if s['m']:
                    me = tokenizer(s['m'], padding=True, truncation=True, max_length=32, return_tensors='pt').to(device)
                    me_emb = embed_layer(me['input_ids'])
                    prefix = mem_enc(me_emb.unsqueeze(0))
                else:
                    prefix = torch.zeros(1, NUM_MEM_TOKENS, inp_emb.shape[-1], device=device, dtype=inp_emb.dtype)

                combined = torch.cat([prefix, inp_emb], dim=1)
                pmask = torch.ones(1, NUM_MEM_TOKENS, device=device, dtype=enc['attention_mask'].dtype)
                cmask = torch.cat([pmask, enc['attention_mask']], dim=1)
                labels = enc['input_ids'].clone()
                labels[enc['attention_mask'] == 0] = -100
                plabels = torch.full((1, NUM_MEM_TOKENS), -100, device=device, dtype=labels.dtype)
                clabels = torch.cat([plabels, labels], dim=1)
                out = base(inputs_embeds=combined, attention_mask=cmask, labels=clabels)
                vl += out.loss.item()

        v_avg = vl / max(len(val_d), 1)
        gate = round(torch.sigmoid(mem_enc.gate).item(), 4)
        print('Epoch ' + str(epoch+1) + '/' + str(EPOCHS) + ' | T:' + str(round(t_avg,3)) + ' V:' + str(round(v_avg,3)) + ' G:' + str(gate))
        if v_avg < best_val:
            best_val = v_avg
            torch.save(mem_enc.state_dict(), os.path.join(OUTPUT_DIR, 'memory_encoder.pt'))
            print('  -> Best!')

    print('Done! Best Val: ' + str(round(best_val, 4)) + ' Gate: ' + str(round(torch.sigmoid(mem_enc.gate).item(), 4)))

if __name__ == '__main__':
    train()
