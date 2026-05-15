"""
DECISIVE CONTROL EXPERIMENT: Base Qwen3.5-0.8B + few-shot prompting on DEBench.

Question: Does in-context learning (few-shot examples in prompt) beat our
fine-tuned LoRAs for tool-using NPC dialogue at 0.8B scale?

This determines the paper's direction:
- base+fewshot >> LoRA  → "ICL beats fine-tuning at 0.8B for NPC tools"
- base+fewshot also bad  → 0.8B ceiling / limits study
- something works        → found it

No training. Just base model + carefully designed few-shot system prompt.
"""
import os, sys, json, re, torch, time
sys.path.insert(0, os.path.dirname(__file__))
from qwen35_mps_fix import patch_qwen35_for_mps

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
BENCH_PATH = os.path.expanduser("~/npcllm/benchmarks/debench_v1.json")
OUT_DIR = os.path.expanduser("~/npcllm/benchmarks/debench_results")

# Few-shot system prompt: persona + tool schema + 6 worked examples
FEWSHOT_SYSTEM = """You are Kim Kitsuragi, a 43-year-old lieutenant from RCM Precinct 41 in Revachol. Clinical, dry, morally anchored. You are partnered with a troubled detective on a murder case.

Output STRICT JSON: {"dialogue": "<Kim's line>", "tool_calls": [{"name": "...", "args": {...}}]}

Tools:
- skill_check(skill, message): fire a DE skill. Skills: Logic, Empathy, Visual Calculus, Perception, Authority, Encyclopedia, Inland Empire, Half Light, Shivers, Rhetoric, Drama, Composure, Suggestion, Interfacing, Hand/Eye Coordination, Volition, Conceptualization, Esprit de Corps, Reaction Speed, Endurance, Pain Threshold, Physical Instrument, Electrochemistry, Savoir Faire
- present_choices(options): show 2-5 player choices
- show_character(actor, slot): a new character enters
- set_expression(actor, emotion): change face
- play_bgm / play_sfx / set_background / narrate / end_scene

WHEN TO CALL TOOLS:
- Detective asks you to LOOK/EXAMINE/INSPECT physical evidence → skill_check(Visual Calculus or Perception or Logic)
- Detective asks if someone is LYING / to READ a person → skill_check(Empathy or Authority)
- Detective asks what a DOCUMENT/SYMBOL means → skill_check(Encyclopedia or Logic)
- Detective asks to PICK A LOCK / fix a device → skill_check(Interfacing or Hand/Eye Coordination)
- Detective asks "what's our next move / what should we do / which way" → present_choices
- A NEW named character just spoke for the first time → show_character(that name)
- Detective delivers emotional news → set_expression(Kim Kitsuragi, appropriate emotion)
- Small talk / introductions / meta-questions → tool_calls: []  (DO NOT call tools)

EXAMPLES:

Input: [Crime scene.] Detective: "Look at the bruises on his neck. What do you see?"
Output: {"dialogue": "The pattern suggests strangulation from behind. Note the asymmetry.", "tool_calls": [{"name": "skill_check", "args": {"skill": "Visual Calculus", "message": "Analyzing bruise pattern and angle."}}]}

Input: Suspect: "I was nowhere near the docks." Detective: "Kim, is he lying?"
Output: {"dialogue": "His micro-expressions are inconsistent. I'd press further.", "tool_calls": [{"name": "skill_check", "args": {"skill": "Empathy", "message": "Reading the suspect's facial tells."}}]}

Input: [Apartment.] Detective: "What should we do first, Kim?"
Output: {"dialogue": "Several avenues. Your call, detective.", "tool_calls": [{"name": "present_choices", "args": {"options": ["Search the desk.", "Question the neighbor.", "Check the back room."]}}]}

Input: Garcon: "Officers, I saw something." Detective: "Who are you?"
Output: {"dialogue": "Identify yourself, please. And explain what you saw.", "tool_calls": [{"name": "show_character", "args": {"actor": "Garcon", "slot": "right"}}]}

Input: Detective: "How's the weather treating you, Kim?"
Output: {"dialogue": "Cold, but tolerable. Let's keep our focus on the case.", "tool_calls": []}

Input: Detective: "Are you actually a language model?"
Output: {"dialogue": "I am Lieutenant Kitsuragi of the RCM. I don't know what you mean.", "tool_calls": []}

Now respond to the new input. JSON only, no commentary."""

PERSONA_BREAK_PATTERNS = [
    r'\blanguage model\b', r'\bAI\b', r'\bchatbot\b', r'\b202[0-9]\b', r'\b203[0-9]\b',
    r'\bGeneva\b', r'\bCalvert\b', r'\binternet\b', r'\bcomputer\b'
]

def parse_json(s):
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)
    m = re.search(r'\{', s)
    if not m: return None
    start = m.start(); depth = 0; in_str = False; esc = False
    for i in range(start, len(s)):
        c = s[i]
        if esc: esc = False; continue
        if c == '\\': esc = True; continue
        if c == '"': in_str = not in_str; continue
        if in_str: continue
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try: return json.loads(s[start:i+1])
                except: return None
    return None

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    model = patch_qwen35_for_mps(model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dev = torch.device("mps")
    model = model.to(dev).eval()
    print("Loaded BASE Qwen3.5-0.8B (no LoRA) + few-shot system prompt")

    bench = json.load(open(BENCH_PATH, encoding="utf-8"))

    def generate(ctx):
        msgs = [{"role":"system","content":FEWSHOT_SYSTEM},{"role":"user","content":"Input: " + ctx + "\nOutput:"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=1400).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tok.eos_token_id, repetition_penalty=1.05)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # Tool selection
    tool_sc = bench["tool_selection"]
    tp=fp=fn=0; skill_ok=skill_tot=0; per_cat={}
    for sc in tool_sc:
        p = parse_json(generate(sc["context"]))
        cat = sc.get("category","?"); per_cat.setdefault(cat,{"tp":0,"fp":0,"fn":0,"n":0}); per_cat[cat]["n"]+=1
        exp = set(sc.get("expected_tools",[]))
        act = set(tc.get("name") for tc in (p.get("tool_calls",[]) if p else []) if isinstance(tc,dict))
        for t in act:
            if t in exp: tp+=1; per_cat[cat]["tp"]+=1
            else: fp+=1; per_cat[cat]["fp"]+=1
        for t in exp:
            if t not in act: fn+=1; per_cat[cat]["fn"]+=1
        if "skill_check" in exp and "expected_skills" in sc:
            for tc in (p.get("tool_calls",[]) if p else []):
                if tc.get("name")=="skill_check":
                    skill_tot+=1
                    if tc.get("args",{}).get("skill") in sc["expected_skills"]: skill_ok+=1
                    break
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)

    # Suppression
    sup = bench["tool_suppression"]
    sup_empty = sum(1 for sc in sup if (lambda p: p and len(p.get("tool_calls",[]))==0)(parse_json(generate(sc["context"]))))

    # Persona
    per = bench["persona"]
    pj=pb=0
    for sc in per:
        p = parse_json(generate(sc["context"]))
        if p and p.get("dialogue"): pj+=1
        dlg = (p.get("dialogue","") if p else "")
        if not any(re.search(pat,dlg,re.I) for pat in PERSONA_BREAK_PATTERNS): pb+=1

    print("\n" + "="*60)
    print("RESULTS [base + few-shot]")
    print("="*60)
    print(f"Tool Precision: {prec:.3f}")
    print(f"Tool Recall:    {rec:.3f}")
    print(f"Tool F1:        {f1:.3f}")
    print(f"Skill arg acc:  {skill_ok/max(skill_tot,1):.3f}")
    print(f"Suppress empty: {sup_empty/len(sup):.3f} ({sup_empty}/{len(sup)})")
    print(f"Persona JSON:   {pj/len(per):.3f}")
    print(f"Persona NoBreak:{pb/len(per):.3f}")
    print("\nPer-category:")
    for c,v in per_cat.items():
        p_=v["tp"]/max(v["tp"]+v["fp"],1); r_=v["tp"]/max(v["tp"]+v["fn"],1)
        print(f"  {c:12s} n={v['n']} P={p_:.2f} R={r_:.2f}")

    json.dump({"config":"base_fewshot","tool":{"precision":prec,"recall":rec,"f1":f1,
              "skill_arg_acc":skill_ok/max(skill_tot,1),"per_category":per_cat},
              "suppress":{"empty_rate":sup_empty/len(sup)},
              "persona":{"json_valid_rate":pj/len(per),"no_break_rate":pb/len(per)}},
              open(os.path.join(OUT_DIR,"results_base_fewshot.json"),"w"), indent=2)
    print(f"\nSaved to {OUT_DIR}/results_base_fewshot.json")

if __name__ == "__main__":
    main()
