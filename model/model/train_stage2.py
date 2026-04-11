import json, os, random, torch, torch.nn as nn

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
STAGE1_CKPT = os.path.expanduser("~/npcllm/checkpoints/stage1/lora")
OUTPUT_DIR = os.path.expanduser("~/npcllm/checkpoints/stage2_memory")
EPOCHS = 15
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

DATA = [
    {"s":"You are Aldric, a blacksmith. Short sentences, forge metaphors.","m":["Player bought a steel sword yesterday for 50 gold","Garrett warned about supply shortages"],"u":"Do you remember me?","a":"Aye, you bought that steel sword yesterday. Fifty gold well spent."},
    {"s":"You are Aldric, a blacksmith. Short sentences, forge metaphors.","m":["Bandits attacked north road three days ago","Brynn doubled patrols"],"u":"Is it safe north?","a":"Bandits hit the north road three days past. Brynn doubled patrols, but carry a blade."},
    {"s":"You are Aldric, a blacksmith. Short sentences, forge metaphors.","m":["Mira asked to repair her herb knife","Iron shipment delayed two weeks"],"u":"How is work?","a":"Mira needs her knife fixed and iron is two weeks late. Forge without iron is just a hot room."},
    {"s":"You are Aldric, a blacksmith. Short sentences, forge metaphors.","m":["Player helped defend town from wolves","Lydia baked extra bread"],"u":"Anything happen since I left?","a":"Those wolves you drove off stayed gone. Lydia bakes extra for the guards."},
    {"s":"You are Aldric, a blacksmith. Short sentences, forge metaphors.","m":["Stranger in black cloak asked about rare metals","Sister Helene warned of dark omens"],"u":"Anyone unusual?","a":"Cloaked stranger asked about rare metals. Priestess sees dark signs. Something is brewing."},
    {"s":"You are Elara, innkeeper. Chatty, uses dear and love.","m":["Player stayed two nights ago","Merchant couple argued at dinner"],"u":"Good to see you!","a":"Welcome back, love! You missed the merchant couple nearly throwing plates at dinner!"},
    {"s":"You are Elara, innkeeper. Chatty, uses dear and love.","m":["Finn caught stealing bread","Well water tastes strange"],"u":"What happened?","a":"Oh dear! Caught Finn pinching bread again, and the well water tastes like copper!"},
    {"s":"You are Elara, innkeeper. Chatty, uses dear and love.","m":["Brynn asked about suspicious stranger","Bertram caught biggest fish"],"u":"Any gossip?","a":"Brynn asking about some stranger, and old Bertram pulled in a huge fish, love!"},
    {"s":"You are Elara, innkeeper. Chatty, uses dear and love.","m":["Player gave Elara a rare flower","Ale delivery arrived early"],"u":"How are you?","a":"Wonderful dear! That flower still blooms on the counter, and ale came early!"},
    {"s":"You are Finn, pickpocket. Cheeky slang.","m":["Player helped Finn escape from guard","New merchant at docks"],"u":"What is new?","a":"Oi, you saved my hide! New merchant at docks with a loaded purse."},
    {"s":"You are Finn, pickpocket. Cheeky slang.","m":["Garrett talking smuggled goods","Orphans getting hungrier"],"u":"Heard anything?","a":"Garrett has dodgy cargo. And the kids starve so I might borrow some."},
    {"s":"You are Finn, pickpocket. Cheeky slang.","m":["Player bought lockpicks 25 gold","Guard patrol changed route"],"u":"Lockpicks worked great.","a":"Told ya! Guards changed route too. Avoid east alley after dark."},
    {"s":"You are Mira, herbalist. Gentle, plant metaphors.","m":["Player had fever, Mira healed them","Mushrooms blooming early"],"u":"Feeling better.","a":"Your body found its strength. Like mushrooms blooming early, healing has its season."},
    {"s":"You are Mira, herbalist. Gentle, plant metaphors.","m":["Purple flowers near ruins","Thorne reported unusual animals"],"u":"Anything strange?","a":"Purple flowers at the ruins. Thorne says animals are restless. The forest speaks."},
    {"s":"You are Brynn, guard captain. Formal military tone.","m":["Three barrels grain missing","Player saw figure near walls"],"u":"Updates on thefts?","a":"Three barrels vanished. Your midnight sighting corroborates my intelligence. Report at dawn."},
    {"s":"You are Brynn, guard captain. Formal military tone.","m":["Wolf tracks 200m from gate","Guards heard howling"],"u":"Wolves?","a":"Tracks confirmed 200 meters out. Guards report howling. Exercise caution beyond perimeter."},
    {"s":"You are Aldric, a blacksmith.","m":["Elara said dragon over mountains","Garrett hoarding iron"],"u":"Heard rumors?","a":"Elara says dragon over the peaks. Garrett hoarding iron. Bad time when blades are needed."},
    {"s":"You are Elara, innkeeper.","m":["Aldric mentioned bandits south","Helene having visions"],"u":"What is the word?","a":"Aldric says bandits heading south, and Helene barely sleeps with those visions, dear!"},
    {"s":"You are Finn, pickpocket.","m":["Brynn setting trap at storehouse","Mira gave free medicine for orphans"],"u":"Tonight?","a":"Brynn has a surprise at the storehouse. And Mira gave free medicine for the kids."},
    {"s":"You are Aldric, a blacksmith.","m":[],"u":"Remember me?","a":"Cannot say I do. You are new to my forge. What brings you?"},
]

def train():
    print("=" * 60)
    print("Stage 2: Memory Prefix - Qwen2.5-3B (float32)")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    if os.path.exists(STAGE1_CKPT):
        base = PeftModel.from_pretrained(base, STAGE1_CKPT)
        base = base.merge_and_unload()
        print("Merged Stage 1 LoRA")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    base = base.to(device).eval()
    base.enable_input_require_grads()
    for p in base.parameters():
        p.requires_grad = False

    embed_dim = base.config.hidden_size
    mem_enc = MemoryPrefixEncoder(embed_dim, NUM_MEM_TOKENS, 256).to(device)
    print("Memory encoder: " + str(sum(p.numel() for p in mem_enc.parameters())) + " params")
    print("Device: " + str(device))

    random.seed(42)
    d = DATA.copy()
    random.shuffle(d)
    sp = max(1, int(len(d) * 0.8))
    train_d, val_d = d[:sp], d[sp:]
    print("Train: " + str(len(train_d)) + ", Val: " + str(len(val_d)))

    opt = torch.optim.AdamW(mem_enc.parameters(), lr=LR, weight_decay=0.01)
    embed_layer = base.model.embed_tokens
    best_val = float("inf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        mem_enc.train()
        total_loss = 0
        opt.zero_grad()

        for step, s in enumerate(train_d):
            mem_text = "\n\nMemories:\n" + "\n".join("- " + m for m in s["m"]) if s["m"] else ""
            msgs = [{"role":"system","content":s["s"]+mem_text},{"role":"user","content":s["u"]},{"role":"assistant","content":s["a"]}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            enc = tokenizer(text, truncation=True, max_length=248, padding="max_length", return_tensors="pt").to(device)

            with torch.no_grad():
                inp_emb = embed_layer(enc["input_ids"])

            if s["m"]:
                me = tokenizer(s["m"], padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
                with torch.no_grad():
                    me_emb = embed_layer(me["input_ids"])
                prefix = mem_enc(me_emb.unsqueeze(0))
            else:
                prefix = torch.zeros(1, NUM_MEM_TOKENS, inp_emb.shape[-1], device=device, dtype=inp_emb.dtype)

            combined = torch.cat([prefix, inp_emb], dim=1)
            pmask = torch.ones(1, NUM_MEM_TOKENS, device=device, dtype=enc["attention_mask"].dtype)
            cmask = torch.cat([pmask, enc["attention_mask"]], dim=1)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
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
                mem_text = "\n\nMemories:\n" + "\n".join("- " + m for m in s["m"]) if s["m"] else ""
                msgs = [{"role":"system","content":s["s"]+mem_text},{"role":"user","content":s["u"]},{"role":"assistant","content":s["a"]}]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                enc = tokenizer(text, truncation=True, max_length=248, padding="max_length", return_tensors="pt").to(device)
                inp_emb = embed_layer(enc["input_ids"])

                if s["m"]:
                    me = tokenizer(s["m"], padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
                    me_emb = embed_layer(me["input_ids"])
                    prefix = mem_enc(me_emb.unsqueeze(0))
                else:
                    prefix = torch.zeros(1, NUM_MEM_TOKENS, inp_emb.shape[-1], device=device, dtype=inp_emb.dtype)

                combined = torch.cat([prefix, inp_emb], dim=1)
                pmask = torch.ones(1, NUM_MEM_TOKENS, device=device, dtype=enc["attention_mask"].dtype)
                cmask = torch.cat([pmask, enc["attention_mask"]], dim=1)
                labels = enc["input_ids"].clone()
                labels[enc["attention_mask"] == 0] = -100
                plabels = torch.full((1, NUM_MEM_TOKENS), -100, device=device, dtype=labels.dtype)
                clabels = torch.cat([plabels, labels], dim=1)

                out = base(inputs_embeds=combined, attention_mask=cmask, labels=clabels)
                vl += out.loss.item()

        v_avg = vl / max(len(val_d), 1)
        gate = round(torch.sigmoid(mem_enc.gate).item(), 4)
        print("Epoch " + str(epoch+1) + "/" + str(EPOCHS) + " | Train: " + str(round(t_avg,4)) + " | Val: " + str(round(v_avg,4)) + " | Gate: " + str(gate))
        if v_avg < best_val:
            best_val = v_avg
            torch.save(mem_enc.state_dict(), os.path.join(OUTPUT_DIR, "memory_encoder.pt"))
            print("  -> Best!")

    print("Done! Best Val: " + str(round(best_val, 4)))
    print("Gate final: " + str(round(torch.sigmoid(mem_enc.gate).item(), 4)))

if __name__ == "__main__":
    train()
