"""Train Stage 2+3 on Qwen3.5-2B (larger model may avoid LoRA overfitting)."""
import json, os, random, torch, torch.nn as nn, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-2B"
BASE_DIR = os.path.expanduser("~/npcllm")
S2_TRAIN = os.path.join(BASE_DIR, "training_data/stage2_large/train.json")
S2_VALID = os.path.join(BASE_DIR, "training_data/stage2_large/valid.json")
S3_TRAIN = os.path.join(BASE_DIR, "training_data/stage3_curated/train.json")
S3_VALID = os.path.join(BASE_DIR, "training_data/stage3_curated/valid.json")
S2_OUT = os.path.join(BASE_DIR, "checkpoints/q35_2b_stage2")
S3_OUT = os.path.join(BASE_DIR, "checkpoints/q35_2b_stage3")
EMOTIONS = ["neutral","happy","angry","sad","fearful","surprised","disgusted","contemptuous"]

class MemoryPrefixEncoder(nn.Module):
    def __init__(self, ed, nt=8, md=256):
        super().__init__()
        self.nt = nt
        self.proj = nn.Linear(ed, md)
        self.to_prefix = nn.Sequential(nn.Linear(md, ed * nt), nn.GELU())
        self.gate = nn.Parameter(torch.zeros(1))
    def forward(self, me):
        p = me.mean(dim=2).mean(dim=1)
        return self.to_prefix(self.proj(p)).view(-1, self.nt, me.shape[-1]) * torch.sigmoid(self.gate)

class EmotionHead(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.c = nn.Sequential(nn.Linear(hs, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 8))
    def forward(self, h): return self.c(h[:, -1, :])

def train_s2():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("STAGE 2: Memory Prefix - Qwen3.5-2B")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    base = patch_qwen35_for_mps(base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dev = torch.device("mps")
    base = base.to(dev).eval()
    base.enable_input_require_grads()
    for p in base.parameters(): p.requires_grad = False
    me = MemoryPrefixEncoder(base.config.hidden_size).to(dev)
    ckpt_path = os.path.join(S2_OUT, "memory_encoder.pt")
    start_ep = 0; bv = float("inf")
    if os.path.exists(ckpt_path):
        me.load_state_dict(torch.load(ckpt_path, map_location=dev))
        print(f"Resumed, gate: {torch.sigmoid(me.gate).item():.4f}")
    print(f"Params: {sum(p.numel() for p in me.parameters())}")
    td = json.load(open(S2_TRAIN)); vd = json.load(open(S2_VALID))
    print(f"Train: {len(td)} Val: {len(vd)}")
    opt = torch.optim.AdamW(me.parameters(), lr=5e-4, weight_decay=0.01)
    el = base.model.embed_tokens
    os.makedirs(S2_OUT, exist_ok=True)
    for ep in range(start_ep, 10):
        me.train(); random.shuffle(td); tl = 0; opt.zero_grad()
        for i, s in enumerate(td):
            try:
                mt = "\n\nMemories:\n" + "\n".join("- " + m for m in s["m"]) if s["m"] else ""
                ms = [{"role":"system","content":s["s"]+mt},{"role":"user","content":s["u"]},{"role":"assistant","content":s["a"]}]
                tx = tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)
                ec = tok(tx, truncation=True, max_length=248, padding="max_length", return_tensors="pt").to(dev)
                with torch.no_grad(): ie = el(ec["input_ids"])
                if s["m"]:
                    mc = tok(s["m"], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
                    with torch.no_grad(): mee = el(mc["input_ids"])
                    pf = me(mee.unsqueeze(0)).to(ie.dtype)
                else:
                    pf = torch.zeros(1, 8, ie.shape[-1], device=dev, dtype=ie.dtype)
                cb = torch.cat([pf, ie], dim=1)
                pm = torch.ones(1, 8, device=dev, dtype=ec["attention_mask"].dtype)
                cm = torch.cat([pm, ec["attention_mask"]], dim=1)
                lb = ec["input_ids"].clone(); lb[ec["attention_mask"]==0] = -100
                pl = torch.full((1,8), -100, device=dev, dtype=lb.dtype)
                cl = torch.cat([pl, lb], dim=1)
                o = base(inputs_embeds=cb, attention_mask=cm, labels=cl)
                if torch.isnan(o.loss): continue
                (o.loss / 4).backward(); tl += o.loss.item()
            except Exception as e:
                opt.zero_grad(); continue
            if (i+1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(me.parameters(), 1.0); opt.step(); opt.zero_grad()
            if (i+1) % 1000 == 0:
                print(f"  S2 E{ep+1} {i+1}/{len(td)} L:{tl/(i+1):.3f}")
        ta = tl / max(len(td),1)
        me.eval(); vl = vc = 0
        with torch.no_grad():
            for s in vd:
                try:
                    mt = "\n\nMemories:\n" + "\n".join("- " + m for m in s["m"]) if s["m"] else ""
                    ms = [{"role":"system","content":s["s"]+mt},{"role":"user","content":s["u"]},{"role":"assistant","content":s["a"]}]
                    tx = tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)
                    ec = tok(tx, truncation=True, max_length=248, padding="max_length", return_tensors="pt").to(dev)
                    ie = el(ec["input_ids"])
                    if s["m"]:
                        mc = tok(s["m"], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
                        mee = el(mc["input_ids"])
                        pf = me(mee.unsqueeze(0)).to(ie.dtype)
                    else:
                        pf = torch.zeros(1,8,ie.shape[-1],device=dev,dtype=ie.dtype)
                    cb = torch.cat([pf,ie],1)
                    pm = torch.ones(1,8,device=dev,dtype=ec["attention_mask"].dtype)
                    cm = torch.cat([pm,ec["attention_mask"]],1)
                    lb = ec["input_ids"].clone(); lb[ec["attention_mask"]==0]=-100
                    pl = torch.full((1,8),-100,device=dev,dtype=lb.dtype)
                    cl = torch.cat([pl,lb],1)
                    o = base(inputs_embeds=cb,attention_mask=cm,labels=cl)
                    if not torch.isnan(o.loss): vl += o.loss.item(); vc += 1
                except: pass
        va = vl / max(vc,1)
        g = torch.sigmoid(me.gate).item()
        print(f"S2 E{ep+1}/10 T:{ta:.3f} V:{va:.3f} G:{g:.4f}")
        if va < bv: bv = va; torch.save(me.state_dict(), ckpt_path); print("  -> Best!")
    print(f"S2 Done! Val:{bv:.4f}")
    del base, me, opt; torch.mps.empty_cache()
    return bv

def train_s3():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\nSTAGE 3: Emotion Head - Qwen3.5-2B")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    base = patch_qwen35_for_mps(base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dev = torch.device("mps")
    base = base.to(dev).eval()
    for p in base.parameters(): p.requires_grad = False
    eh = EmotionHead(base.config.hidden_size).to(dev)
    ckpt = os.path.join(S3_OUT, "emotion_head.pt")
    if os.path.exists(ckpt):
        eh.load_state_dict(torch.load(ckpt, map_location=dev))
        print("Resumed")
    td = json.load(open(S3_TRAIN)); vd = json.load(open(S3_VALID))
    print(f"Train: {len(td)} Val: {len(vd)}")
    opt = torch.optim.AdamW(eh.parameters(), lr=1e-3, weight_decay=0.01)
    cr = nn.CrossEntropyLoss(); best = 0
    os.makedirs(S3_OUT, exist_ok=True)
    for ep in range(20):
        eh.train(); random.shuffle(td); c = t = 0
        for i in range(0, len(td), 8):
            b = td[i:i+8]; tx = []; lb = []
            for s in b:
                ms = [{"role":"system","content":s.get("system","NPC")},{"role":"user","content":s.get("user","")},{"role":"assistant","content":s.get("assistant","")}]
                tx.append(tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False))
                e = s.get("emotion",0)
                lb.append(EMOTIONS.index(e) if isinstance(e,str) and e in EMOTIONS else (e if isinstance(e,int) else 0))
            ec = tok(tx, truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
            lt = torch.tensor(lb, device=dev)
            with torch.no_grad(): h = base(**ec, output_hidden_states=True).hidden_states[-1]
            lo = eh(h); loss = cr(lo, lt)
            opt.zero_grad(); loss.backward(); opt.step()
            c += (lo.argmax(-1)==lt).sum().item(); t += len(lb)
        ta = c/t
        eh.eval(); vc = vt = 0
        with torch.no_grad():
            for i in range(0, len(vd), 8):
                b = vd[i:i+8]; tx = []; lb = []
                for s in b:
                    ms = [{"role":"system","content":s.get("system","NPC")},{"role":"user","content":s.get("user","")},{"role":"assistant","content":s.get("assistant","")}]
                    tx.append(tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False))
                    e = s.get("emotion",0)
                    lb.append(EMOTIONS.index(e) if isinstance(e,str) and e in EMOTIONS else (e if isinstance(e,int) else 0))
                ec = tok(tx, truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
                lt = torch.tensor(lb, device=dev)
                h = base(**ec, output_hidden_states=True).hidden_states[-1]
                vc += (eh(h).argmax(-1)==lt).sum().item(); vt += len(lb)
        va = vc/max(vt,1)
        print(f"S3 E{ep+1}/20 TA:{ta:.3f} VA:{va:.3f}")
        if va > best: best = va; torch.save(eh.state_dict(), ckpt); print("  -> Best!")
    print(f"S3 Done! Acc:{best:.4f}")
    return best

if __name__ == "__main__":
    t0 = time.time()
    s2 = train_s2()
    s3 = train_s3()
    print(f"\nDONE ({(time.time()-t0)/60:.1f}m) S2:{s2:.4f} S3:{s3:.4f}")
