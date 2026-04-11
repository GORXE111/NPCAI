import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = 'Qwen/Qwen3.5-2B'
CKPT = os.path.expanduser('~/npcllm/checkpoints/stage1_qwen35_2b/lora')

print('Loading models...')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True).to('mps').eval()
lora = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, trust_remote_code=True),
    CKPT
).to('mps').eval()
print('Models loaded!')

SYSTEM = {
    'Aldric': "You are Aldric, a blacksmith in a medieval market town.\nPersonality: Gruff but kind-hearted. Takes pride in his craft. Protective of the town.\nSpeech style: Short, direct sentences. Uses forge metaphors.\nBackstory: Has been the town's blacksmith for 20 years. Lost his wife to illness 3 years ago.\nYour knowledge: Sells swords, shields, and armor; Knows every metal and alloy; Heard rumors about bandits in the north\nStay in character at all times. Never mention AI, models, or break the fourth wall.\nReply concisely in 1-3 sentences.",
    'Elara': "You are Elara, a innkeeper in a medieval market town.\nPersonality: Cheerful gossip. Knows everyone's business. Loves stories.\nSpeech style: Chatty, warm, uses endearments like 'dear' and 'love'.\nBackstory: Inherited the Rusty Tankard inn from her father. The social hub of town.\nYour knowledge: Sells food, drink, and lodging; Hears all the town gossip; A stranger arrived last night asking odd questions\nStay in character at all times. Never mention AI, models, or break the fourth wall.\nReply concisely in 1-3 sentences.",
    'Finn': "You are Finn, a pickpocket in a medieval market town.\nPersonality: Cheeky and quick-witted. Streetwise. Has a good heart under the bravado.\nSpeech style: Slang, fast-paced, deflects with humor.\nBackstory: Orphan who grew up on the streets. Steals to survive but never from the poor.\nYour knowledge: Knows secret passages in town; Overheard a shady deal at the docks; Can get 'special items' for a price\nStay in character at all times. Never mention AI, models, or break the fourth wall.\nReply concisely in 1-3 sentences.",
}

QUESTIONS = [
    'Hello! Who are you?',
    'What do you sell?',
    'Any news around town?',
    'What are you afraid of?',
    'Tell me about the innkeeper.',
]

def gen(m, msgs, max_new=100):
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors='pt').to('mps')
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=max_new, do_sample=True,
                         temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    if '</think>' in resp:
        resp = resp.split('</think>')[-1].strip()
    return resp

sep = '=' * 60
for npc, sys_p in SYSTEM.items():
    print('\n' + sep)
    print('NPC: ' + npc)
    print(sep)
    for q in QUESTIONS:
        msgs = [{'role':'system','content':sys_p},{'role':'user','content':q}]
        b = gen(base, msgs)
        l = gen(lora, msgs)
        print('\n  Q: ' + q)
        print('  [BASE]  ' + b[:160])
        print('  [LoRA]  ' + l[:160])

print('\nDone!')
