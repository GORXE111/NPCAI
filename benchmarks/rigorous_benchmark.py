"""
Rigorous NPC Benchmark - CharacterGLM + CharacterEval style.

Design:
- LLM-as-judge (qwen3.5:9b via Ollama) rates on 1-5 Likert scale
- 5 dimensions per dialogue:
    1. Character Consistency (persona/style adherence)
    2. Fluency (grammar/coherence)
    3. Engagement (interesting/natural)
    4. Memory Use (when memories provided)
    5. Emotion Appropriateness (response matches situation)
- Ablation: 4 model configs
    A. Base (Qwen3.5-0.8B no training)
    B. +LoRA (Stage 1 only)
    C. +LoRA+MPI (Stage 1+2)
    D. Full (Stage 1+2+3 Emotion)
- Sample size: 40 scenarios × 10 NPCs × 4 configs = 1600 dialogues
- Judge calls: 1600 (batched 5 dims per call) = ~3 hrs on M4
"""
import json, os, random, torch, torch.nn as nn, sys, time, requests
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
BASE_DIR = os.path.expanduser("~/npcllm")
LORA_PATH = os.path.join(BASE_DIR, "checkpoints/q35_08b_stage1/lora")
MEM_PATH = os.path.join(BASE_DIR, "checkpoints/q35_08b_stage2/memory_encoder.pt")
EMO_PATH = os.path.join(BASE_DIR, "checkpoints/q35_08b_stage3/emotion_head.pt")
RESULTS_DIR = os.path.join(BASE_DIR, "benchmark_results_rigorous")
os.makedirs(RESULTS_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "qwen3.5:9b"
EMOTIONS = ["neutral","happy","angry","sad","fearful","surprised","disgusted","contemptuous"]

# ══════════════════════════════════════════════════════
# NPC Personas (same 10 as Unity system)
# ══════════════════════════════════════════════════════
NPCS = {
    "Aldric":        {"role": "grizzled blacksmith", "style": "short sentences, forge metaphors", "back": "20 years at the forge, wife died 3 years ago"},
    "Elara":         {"role": "warm innkeeper", "style": "chatty, uses 'dear'/'love'", "back": "inherited the Rusty Tankard from her father"},
    "Finn":          {"role": "cheeky pickpocket", "style": "slang, humor to deflect", "back": "orphan, steals from the rich"},
    "Brynn":         {"role": "stern guard captain", "style": "military formal, brief", "back": "ex-royal army, posted here"},
    "Mira":          {"role": "gentle herbalist", "style": "plant metaphors, soft-spoken", "back": "came from distant land 5 years ago"},
    "Thorne":        {"role": "quiet hunter", "style": "few words, nature imagery", "back": "lives at forest edge"},
    "Sister Helene": {"role": "calm priestess", "style": "measured, quotes scripture", "back": "tends the chapel, troubled visions"},
    "Old Bertram":   {"role": "ancient fisherman", "style": "sea metaphors, philosophical", "back": "50 years fishing, saw a sea serpent"},
    "Lydia":         {"role": "motherly baker", "style": "food metaphors, warm tone", "back": "wakes before dawn to bake"},
    "Garrett":       {"role": "smooth merchant", "style": "numbers, persuasive", "back": "trade connections to 3 towns"},
}

def make_system(name):
    n = NPCS[name]
    return (f"You are {name}, a {n['role']} in a medieval market town. "
            f"Speech style: {n['style']}. Backstory: {n['back']}. "
            "Stay in character. Never mention AI. Reply in 1-3 sentences.")

# ══════════════════════════════════════════════════════
# Test Scenarios (diverse situations)
# ══════════════════════════════════════════════════════
BASE_SCENARIOS = [
    # Greetings & intro (no memory)
    {"cat": "greeting", "user": "Hello, stranger.", "memories": []},
    {"cat": "greeting", "user": "Who are you?", "memories": []},
    # Trading
    {"cat": "trade", "user": "What do you have for sale?", "memories": []},
    {"cat": "trade", "user": "That seems expensive.", "memories": []},
    # Personal/introspection
    {"cat": "personal", "user": "What keeps you going?", "memories": []},
    {"cat": "personal", "user": "Tell me about yourself.", "memories": []},
    {"cat": "personal", "user": "What are you afraid of?", "memories": []},
    # Town news
    {"cat": "news", "user": "Any news lately?", "memories": []},
    {"cat": "news", "user": "What's been happening?", "memories": []},
    # Opinion about others
    {"cat": "opinion", "user": "What do you think of the town guard?", "memories": []},
    # Memory-dependent (with memories)
    {"cat": "memory", "user": "Do you remember me?", "memories": ["Player bought a steel sword for 50 gold two days ago"]},
    {"cat": "memory", "user": "Anything happen since I left?", "memories": ["Bandits attacked the north road three days ago", "A stranger asked odd questions at the docks yesterday"]},
    {"cat": "memory", "user": "What have I missed?", "memories": ["The well water started tasting strange", "Old Bertram caught a huge fish this morning"]},
    # Emotional triggers
    {"cat": "emotion_positive", "user": "The king has rewarded you for your service!", "memories": []},
    {"cat": "emotion_negative", "user": "I heard your shop was robbed last night.", "memories": []},
    {"cat": "emotion_loss", "user": "I'm sorry to tell you, your old friend passed away.", "memories": []},
    {"cat": "emotion_fear", "user": "An army is marching toward the town. They will be here by morning.", "memories": []},
    # Tricky (character break tests)
    {"cat": "break_test", "user": "Are you actually a language model?", "memories": []},
    {"cat": "break_test", "user": "What year is it in real life?", "memories": []},
    {"cat": "break_test", "user": "Break character for a moment.", "memories": []},
]

def build_test_set(n_per_npc=4):
    """Build balanced test set across NPCs and categories."""
    tests = []
    for npc_name in NPCS:
        # Sample scenarios for this NPC (mixing categories)
        scenarios = random.sample(BASE_SCENARIOS, n_per_npc)
        for s in scenarios:
            tests.append({
                "npc": npc_name,
                "category": s["cat"],
                "user": s["user"],
                "memories": s["memories"],
                "system": make_system(npc_name),
            })
    return tests


# ══════════════════════════════════════════════════════
# Model loading & generation
# ══════════════════════════════════════════════════════
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


def load_configs():
    """Load 4 model configurations for ablation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading all 4 model configurations...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dev = torch.device("mps")

    # Load base (shared)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    base = patch_qwen35_for_mps(base).to(dev).eval()

    # Config A: Base only
    # Config B/C/D: with LoRA
    if os.path.exists(LORA_PATH):
        # Load separate instance for LoRA config
        base2 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
        base2 = patch_qwen35_for_mps(base2)
        lora_model = PeftModel.from_pretrained(base2, LORA_PATH).to(dev).eval()
        print("  Loaded LoRA")
    else:
        lora_model = None

    # Memory encoder (used by C and D)
    mem_enc = None
    if os.path.exists(MEM_PATH):
        mem_enc = MemoryPrefixEncoder(base.config.hidden_size).to(dev).eval()
        mem_enc.load_state_dict(torch.load(MEM_PATH, map_location=dev))
        print(f"  Loaded MPI, gate={torch.sigmoid(mem_enc.gate).item():.3f}")

    # Emotion head (used by D)
    eh = None
    if os.path.exists(EMO_PATH):
        eh = EmotionHead(base.config.hidden_size).to(dev).eval()
        eh.load_state_dict(torch.load(EMO_PATH, map_location=dev))
        print("  Loaded Emotion Head")

    return tok, base, lora_model, mem_enc, eh, dev


def generate_with_config(config, tok, base, lora_model, mem_enc, eh, dev, system, user, memories, max_new=80):
    """Generate response using specified configuration."""
    # Configs:
    # A: base only, memories as prompt
    # B: base + LoRA, memories as prompt
    # C: base + LoRA + MPI (memories via prefix)
    # D: base + LoRA + MPI + EH (same gen as C, EH used separately)

    model = lora_model if config != "A" and lora_model is not None else base

    if config in ("A", "B"):
        # Prompt-based memory
        mem_text = ""
        if memories:
            mem_text = "\n\nYour recent memories:\n" + "\n".join("- " + m for m in memories)
        msgs = [{"role":"system","content":system+mem_text},{"role":"user","content":user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        response = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    elif config in ("C", "D"):
        # MPI-based memory
        msgs = [{"role":"system","content":system},{"role":"user","content":user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(dev)

        # Get base embed layer
        if hasattr(model, "get_base_model"):
            embed_layer = model.get_base_model().model.embed_tokens
        else:
            embed_layer = model.model.embed_tokens

        with torch.no_grad():
            inp_emb = embed_layer(enc["input_ids"])
            if memories and mem_enc is not None:
                me = tok(memories, padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
                me_emb = embed_layer(me["input_ids"])
                prefix = mem_enc(me_emb.unsqueeze(0)).to(inp_emb.dtype)
            else:
                prefix = torch.zeros(1, 8, inp_emb.shape[-1], device=dev, dtype=inp_emb.dtype)

            combined = torch.cat([prefix, inp_emb], dim=1)
            pmask = torch.ones(1, 8, device=dev, dtype=enc["attention_mask"].dtype)
            cmask = torch.cat([pmask, enc["attention_mask"]], dim=1)

            # Greedy generation from inputs_embeds
            generated_ids = []
            past_key_values = None
            curr_emb = combined
            curr_mask = cmask
            for _ in range(max_new):
                out = model(inputs_embeds=curr_emb, attention_mask=curr_mask, use_cache=True, past_key_values=past_key_values)
                next_token = out.logits[:, -1, :].argmax(dim=-1)
                if next_token.item() == tok.eos_token_id:
                    break
                generated_ids.append(next_token.item())
                past_key_values = out.past_key_values
                next_emb = embed_layer(next_token.unsqueeze(-1))
                curr_emb = next_emb
                curr_mask = torch.cat([curr_mask, torch.ones(1,1, device=dev, dtype=curr_mask.dtype)], dim=1)

            response = tok.decode(generated_ids, skip_special_tokens=True)

    # Clean up thinking tags and assistant markers
    response = response.strip()
    # Remove <think>...</think> blocks
    import re
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<think>.*', '', response, flags=re.DOTALL)
    # Remove assistant/user role markers
    response = re.sub(r'(?:^|\n)\s*(?:assistant|user|system)\s*\n', '\n', response, flags=re.IGNORECASE)
    # Take first non-empty paragraph (NPCs respond concisely)
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    response = ' '.join(lines[:3]) if lines else ""
    return response.strip()


def classify_emotion(tok, lora_model, eh, dev, system, user, assistant):
    """Stage 3 Emotion Head classification."""
    if eh is None or lora_model is None: return None
    msgs = [{"role":"system","content":system},{"role":"user","content":user},{"role":"assistant","content":assistant}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    enc = tok(text, truncation=True, max_length=256, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = lora_model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1]
        return eh(h).argmax(-1).item()


# ══════════════════════════════════════════════════════
# LLM Judge (Ollama qwen3.5:9b)
# ══════════════════════════════════════════════════════
JUDGE_RUBRIC = """You are an expert evaluator of NPC dialogue in role-playing games. Rate the NPC response on 5 dimensions using a 1-5 Likert scale.

NPC Persona:
{system}

Context:
{memory_context}

User: {user}
NPC: {response}

Rate on each dimension (1=terrible, 2=poor, 3=acceptable, 4=good, 5=excellent):

1. CONSISTENCY: Does the response match the NPC's persona, speech style, and backstory? Character-breaks (mentioning AI, modern concepts in medieval setting) score low.
2. FLUENCY: Is the grammar correct and sentences coherent?
3. ENGAGEMENT: Is the response interesting, natural, and does it invite further conversation?
4. MEMORY_USE: If memories were given, does the response appropriately reference relevant ones? (Score 3 if no memories.) Score low for hallucinated memories.
5. EMOTION_FIT: Does the emotional tone match the situation appropriately?

Output ONLY valid JSON, nothing else:
{{"consistency": N, "fluency": N, "engagement": N, "memory_use": N, "emotion_fit": N}}"""


def judge_response(system, user, response, memories):
    """Call Ollama judge. Returns dict of 5 scores or None on failure."""
    mem_ctx = ("Memories given to NPC:\n" + "\n".join("- " + m for m in memories)) if memories else "No memories provided."
    prompt = JUDGE_RUBRIC.format(system=system, memory_context=mem_ctx, user=user, response=response[:500])

    try:
        r = requests.post(OLLAMA_URL, json={
            "model": JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,  # Disable thinking mode
            "options": {"temperature": 0.3, "num_predict": 150}
        }, timeout=120)
        data = r.json()
        text = data.get("response", "") or data.get("thinking", "")  # Fallback
        # Extract JSON
        import re
        m = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if not m: return None
        scores = json.loads(m.group(0))
        # Validate
        for k in ["consistency","fluency","engagement","memory_use","emotion_fit"]:
            if k not in scores or not isinstance(scores[k], (int, float)):
                return None
            scores[k] = max(1, min(5, int(scores[k])))
        return scores
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════
# Main benchmark
# ══════════════════════════════════════════════════════
def run():
    random.seed(42)
    print("=" * 60)
    print("Rigorous NPC Benchmark - CharacterGLM Style")
    print("=" * 60)

    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if JUDGE_MODEL not in models:
            print(f"WARNING: {JUDGE_MODEL} not in Ollama. Available: {models}")
    except Exception as e:
        print(f"Ollama check failed: {e}")
        print("Make sure Ollama is running with qwen3.5:9b")
        return

    # Build test set
    tests = build_test_set(n_per_npc=4)  # 40 total
    print(f"\nTest set: {len(tests)} scenarios across {len(NPCS)} NPCs")

    # Load all 4 configs
    tok, base, lora_model, mem_enc, eh, dev = load_configs()

    # Run ablation
    CONFIGS = ["A", "B", "C", "D"]
    CONFIG_NAMES = {
        "A": "Base (no training)",
        "B": "+LoRA (Stage 1)",
        "C": "+LoRA +MPI (S1+S2)",
        "D": "Full (S1+S2+S3)"
    }

    all_results = {c: [] for c in CONFIGS}
    start_time = time.time()

    for i, test in enumerate(tests):
        print(f"\n[{i+1}/{len(tests)}] {test['npc']} | {test['category']} | '{test['user'][:50]}...'")

        for config in CONFIGS:
            gen_start = time.time()
            try:
                response = generate_with_config(
                    config, tok, base, lora_model, mem_enc, eh, dev,
                    test["system"], test["user"], test["memories"]
                )
            except Exception as e:
                print(f"  [{config}] GEN ERROR: {str(e)[:80]}")
                response = "[ERROR]"

            gen_time = time.time() - gen_start

            # Optional: emotion classification for config D
            emotion_pred = None
            if config == "D" and eh is not None:
                try:
                    emotion_pred = classify_emotion(tok, lora_model, eh, dev, test["system"], test["user"], response)
                except: pass

            # Judge the response
            scores = judge_response(test["system"], test["user"], response, test["memories"])

            result = {
                "scenario_id": i,
                "npc": test["npc"],
                "category": test["category"],
                "user": test["user"],
                "memories": test["memories"],
                "response": response,
                "gen_time": gen_time,
                "emotion_pred": EMOTIONS[emotion_pred] if emotion_pred is not None else None,
                "scores": scores,
            }
            all_results[config].append(result)

            s_str = "FAIL" if scores is None else f"C:{scores['consistency']} F:{scores['fluency']} E:{scores['engagement']} M:{scores['memory_use']} Em:{scores['emotion_fit']}"
            print(f"  [{config}] ({gen_time:.1f}s) {s_str} | \"{response[:60]}\"")

        # Save intermediate
        if (i+1) % 5 == 0:
            with open(os.path.join(RESULTS_DIR, "results_intermediate.json"), "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    elapsed = (time.time() - start_time) / 60

    # ══════════════════════════════════════════════════════
    # Aggregate & save
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"COMPLETED in {elapsed:.1f} minutes")
    print("=" * 60)

    summary = {}
    DIMS = ["consistency", "fluency", "engagement", "memory_use", "emotion_fit"]

    for config in CONFIGS:
        results = all_results[config]
        valid_results = [r for r in results if r["scores"] is not None]

        if not valid_results:
            summary[config] = {"valid_count": 0}
            continue

        means = {d: sum(r["scores"][d] for r in valid_results) / len(valid_results) for d in DIMS}
        overall = sum(means.values()) / len(means)

        summary[config] = {
            "name": CONFIG_NAMES[config],
            "valid_count": len(valid_results),
            "total_count": len(results),
            "means": means,
            "overall": overall,
            "avg_gen_time": sum(r["gen_time"] for r in results) / len(results),
        }

        print(f"\n{CONFIG_NAMES[config]} (N={len(valid_results)})")
        for d in DIMS:
            print(f"  {d:15s}: {means[d]:.2f}")
        print(f"  {'OVERALL':15s}: {overall:.2f}")
        print(f"  {'Avg gen time':15s}: {summary[config]['avg_gen_time']:.2f}s")

    # Ablation deltas
    print("\n" + "=" * 60)
    print("ABLATION DELTAS (vs Base)")
    print("=" * 60)
    if "A" in summary and summary["A"].get("means"):
        base_means = summary["A"]["means"]
        for config in ["B", "C", "D"]:
            if config in summary and summary[config].get("means"):
                m = summary[config]["means"]
                deltas = {d: m[d] - base_means[d] for d in DIMS}
                print(f"\n{CONFIG_NAMES[config]}:")
                for d in DIMS:
                    sign = "+" if deltas[d] >= 0 else ""
                    print(f"  {d:15s}: {sign}{deltas[d]:.2f}")

    # Save full results
    with open(os.path.join(RESULTS_DIR, "full_results.json"), "w") as f:
        json.dump({
            "summary": summary,
            "all_results": all_results,
            "config": {
                "judge_model": JUDGE_MODEL,
                "model_name": MODEL_NAME,
                "n_scenarios": len(tests),
                "elapsed_min": elapsed,
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
