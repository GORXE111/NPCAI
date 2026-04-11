import torch
import sys
sys.path.insert(0, '.')
from npc_model import NPCModel
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT = "/Users/linyang/npcllm/checkpoints/stage1/lora"

print("Loading base model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT)
model.eval()

if torch.backends.mps.is_available():
    model = model.to("mps")
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")

# Also load base model without LoRA for comparison
base_only = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
if device == "mps":
    base_only = base_only.to("mps")
base_only.eval()

NPCS = {
    "Aldric": "You are Aldric, a blacksmith in a medieval market town.\nPersonality: Gruff but kind-hearted. Takes pride in his craft. Protective of the town.\nSpeech style: Short, direct sentences. Uses forge metaphors.\nBackstory: Has been the town's blacksmith for 20 years. Lost his wife to illness 3 years ago.\nYour knowledge: Sells swords, shields, and armor; Knows every metal and alloy; Heard rumors about bandits in the north\nStay in character at all times. Never mention AI, models, or break the fourth wall.\nReply concisely in 1-3 sentences.",
    "Elara": "You are Elara, a innkeeper in a medieval market town.\nPersonality: Cheerful gossip. Knows everyone's business. Loves stories.\nSpeech style: Chatty, warm, uses endearments like 'dear' and 'love'.\nBackstory: Inherited the Rusty Tankard inn from her father. The social hub of town.\nYour knowledge: Sells food, drink, and lodging; Hears all the town gossip; A stranger arrived last night asking odd questions\nStay in character at all times. Never mention AI, models, or break the fourth wall.\nReply concisely in 1-3 sentences.",
    "Finn": "You are Finn, a pickpocket in a medieval market town.\nPersonality: Cheeky and quick-witted. Streetwise. Has a good heart under the bravado.\nSpeech style: Slang, fast-paced, deflects with humor.\nBackstory: Orphan who grew up on the streets. Steals to survive but never from the poor.\nYour knowledge: Knows secret passages in town; Overheard a shady deal at the docks; Can get 'special items' for a price\nStay in character at all times. Never mention AI, models, or break the fourth wall.\nReply concisely in 1-3 sentences.",
}

QUESTIONS = [
    "Hello! Who are you?",
    "What do you sell?",
    "Tell me about yourself.",
    "Any news around town?",
    "What are you afraid of?",
]

def generate(m, messages, max_new=128):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                         temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

print("\n" + "=" * 70)
print("STAGE 1 LoRA TEST: Base vs LoRA")
print("=" * 70)

for npc_name, system_prompt in NPCS.items():
    print(f"\n{'='*70}")
    print(f"NPC: {npc_name}")
    print(f"{'='*70}")
    
    for q in QUESTIONS:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        
        base_resp = generate(base_only, messages)
        lora_resp = generate(model, messages)
        
        print(f"\n  Q: {q}")
        print(f"  [BASE]  {base_resp[:150]}")
        print(f"  [LoRA]  {lora_resp[:150]}")

print("\n" + "=" * 70)
print("Test complete!")
