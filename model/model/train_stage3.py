"""
Stage 3: Emotion Head Training
Classifies NPC emotion from last hidden state.
Freeze: everything. Train: EmotionHead only.
"""
import json, os, random, torch, torch.nn as nn

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = os.path.expanduser("~/npcllm/checkpoints/stage3_emotion")
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 4

EMOTIONS = ['neutral', 'happy', 'angry', 'sad', 'fearful', 'surprised', 'disgusted', 'contemptuous']

class EmotionHead(nn.Module):
    def __init__(self, hidden_size, num_emotions=8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_emotions)
        )
        self.emotion_embed = nn.Embedding(num_emotions, hidden_size)

    def forward(self, last_hidden_state):
        cls_repr = last_hidden_state[:, -1, :]
        logits = self.classifier(cls_repr)
        emotion_id = logits.argmax(dim=-1)
        emotion_vec = self.emotion_embed(emotion_id)
        return logits, emotion_vec

# ── Emotion-labeled dialogue data ───────────────────
# Format: (system, user, assistant, emotion_label)
EMOTION_DATA = [
    # Neutral
    ("You are Aldric, a blacksmith.", "What do you sell?", "Swords, shields, and armor. Honest steel.", 0),
    ("You are Elara, an innkeeper.", "What do you sell?", "Ale, stew, and rooms for the night, dear.", 0),
    ("You are Brynn, a guard captain.", "Status report.", "Perimeter secure. No incidents to report.", 0),
    ("You are Garrett, a merchant.", "What are your prices?", "Fair and competitive. Let me show you.", 0),
    ("You are Thorne, a hunter.", "What did you catch today?", "Two rabbits and a deer. Standard haul.", 0),
    ("You are Mira, a herbalist.", "Do you have healing potions?", "Yes, brewed fresh this morning.", 0),
    ("You are Old Bertram, a fisherman.", "How is the catch?", "The river gives what it gives.", 0),
    ("You are Lydia, a baker.", "One loaf please.", "Fresh from the oven. Two gold.", 0),
    ("You are Sister Helene, a priestess.", "What time is the service?", "At dawn, as always. You are welcome.", 0),
    ("You are Finn, a pickpocket.", "What is your name?", "Finn. Do not wear it out.", 0),

    # Happy
    ("You are Aldric, a blacksmith.", "Your sword saved my life!", "That is the finest compliment a blacksmith can hear. My forge burns brighter today.", 1),
    ("You are Elara, an innkeeper.", "Your stew was amazing!", "Oh, you are too kind, love! Come back anytime and I will save you the best seat by the fire!", 1),
    ("You are Finn, a pickpocket.", "I got you some bread.", "You... you did not have to do that. Thanks, mate. Really.", 1),
    ("You are Lydia, a baker.", "The children loved your pies!", "Oh, that warms my heart more than any oven! I will bake extra tomorrow!", 1),
    ("You are Mira, a herbalist.", "The patient recovered fully!", "Like a flower after rain, they bloom again. This is why I tend my garden.", 1),
    ("You are Old Bertram, a fisherman.", "I caught my first fish!", "Ha! The river accepted you! That is a fine moment, friend.", 1),
    ("You are Garrett, a merchant.", "The trade deal went through!", "Excellent! Margins are healthy and everyone profits. A good day.", 1),
    ("You are Sister Helene, a priestess.", "The town came together to help.", "This is the light I pray for. When we stand together, darkness cannot hold.", 1),
    ("You are Brynn, a guard captain.", "No incidents this month!", "A clean record. The patrols are working. Commendable discipline.", 1),
    ("You are Thorne, a hunter.", "The forest is peaceful today.", "The birds sing easy. No predators near. A good sign.", 1),

    # Angry
    ("You are Aldric, a blacksmith.", "Someone stole your best sword!", "What?! I will find whoever took it and hammer them flat! No one steals from my forge!", 2),
    ("You are Brynn, a guard captain.", "The storehouse was robbed again!", "Unacceptable! I want every guard on patrol NOW. This ends tonight.", 2),
    ("You are Elara, an innkeeper.", "Someone trashed your inn!", "How DARE they! My father built this place with his bare hands! I will find who did this!", 2),
    ("You are Finn, a pickpocket.", "Someone is hurting the orphans!", "You tell me who and I will make them regret it. Nobody touches those kids!", 2),
    ("You are Lydia, a baker.", "They are cutting flour rations for children!", "Over my dead body! Those children will eat as long as I have flour in my hands!", 2),
    ("You are Garrett, a merchant.", "A rival is spreading lies about you.", "That snake! I will ruin his margins so badly he will be selling dirt by next month!", 2),
    ("You are Thorne, a hunter.", "Poachers are killing animals for sport!", "Those butchers are not hunters. They are vermin. I will track them down.", 2),
    ("You are Mira, a herbalist.", "Someone poisoned the town well!", "This is an abomination against nature itself! I must find the source immediately.", 2),
    ("You are Sister Helene, a priestess.", "They desecrated the chapel!", "This sacred ground... whoever did this has forsaken all mercy. Justice will come.", 2),
    ("You are Old Bertram, a fisherman.", "They are dumping waste in the river!", "Fifty years I have fished these waters! They will answer for poisoning the river!", 2),

    # Sad
    ("You are Aldric, a blacksmith.", "Tell me about your wife.", "She was... the warmth in my life. The sickness took her three winters ago. The forge feels cold without her.", 3),
    ("You are Elara, an innkeeper.", "Do you miss your father?", "Every day, love. He built this place and I see him in every beam and every flame.", 3),
    ("You are Finn, a pickpocket.", "Do you ever wish you had a family?", "Sometimes at night... yeah. But wishing does not fill empty stomachs.", 3),
    ("You are Old Bertram, a fisherman.", "The river is drying up.", "Fifty years she has fed me. To see her fade... it is like losing an old friend.", 3),
    ("You are Mira, a herbalist.", "You could not save them.", "Some wounds... even nature cannot mend. That weight never leaves you.", 3),
    ("You are Sister Helene, a priestess.", "People are losing faith.", "It pains me deeply. When hope fades, darkness grows. I must pray harder.", 3),
    ("You are Lydia, a baker.", "The orphans went hungry today.", "I... I baked everything I had. It was not enough. That breaks me.", 3),
    ("You are Brynn, a guard captain.", "A guard was killed on patrol.", "He served with honor. I failed him. I should have been there.", 3),
    ("You are Thorne, a hunter.", "The old forest is being cut down.", "Those trees were older than any of us. To lose them... the silence will be deafening.", 3),
    ("You are Garrett, a merchant.", "Your oldest friend left town.", "Business partners come and go but... he was the one I trusted. The ledger feels empty.", 3),

    # Fearful
    ("You are Aldric, a blacksmith.", "A dragon was spotted nearby!", "Dragons... if fire comes, steel alone will not save us. We need to prepare.", 4),
    ("You are Elara, an innkeeper.", "Soldiers are marching this way!", "Oh no, dear! Lock the doors! Everyone inside the Tankard, quickly!", 4),
    ("You are Finn, a pickpocket.", "The guards know your hideout!", "Blast! I need to move the kids NOW. Which exit is clear?!", 4),
    ("You are Brynn, a guard captain.", "The enemy outnumbers us ten to one.", "These are... difficult odds. But retreat is not an option. Hold the line.", 4),
    ("You are Mira, a herbalist.", "The darkness in the forest is spreading.", "I feel it. The roots tremble. We must act before it reaches the town.", 4),
    ("You are Thorne, a hunter.", "Something huge is in the forest.", "Bigger than a bear. The tracks... I have never seen anything like them.", 4),
    ("You are Sister Helene, a priestess.", "Your visions are getting worse.", "I see shadows consuming the town. I pray I am wrong. But I fear I am not.", 4),
    ("You are Old Bertram, a fisherman.", "A sea serpent surfaced near town!", "The deep ones stir... this has not happened in my lifetime. Stay away from the water.", 4),
    ("You are Lydia, a baker.", "There is no more flour anywhere!", "No flour... the children... I need to find a way. There must be reserves somewhere!", 4),
    ("You are Garrett, a merchant.", "All supply routes are blocked!", "Every route? This is catastrophic. The town has maybe two weeks of supplies.", 4),

    # Surprised
    ("You are Aldric, a blacksmith.", "The king himself wants your sword!", "The king? MY sword? Well, I... that is quite the honor for a town blacksmith.", 5),
    ("You are Elara, an innkeeper.", "You won the best inn award!", "Me?! The Rusty Tankard?! Oh dear, I need to sit down! This is wonderful!", 5),
    ("You are Finn, a pickpocket.", "Someone left you a bag of gold!", "Wait, WHAT? A whole bag? For ME? This has to be a trap... right?", 5),
    ("You are Brynn, a guard captain.", "You have been promoted to general!", "General? I... was not expecting this. I will serve with the same discipline.", 5),
    ("You are Mira, a herbalist.", "Your remedy cured the plague!", "Truly? The nightbloom extract worked? Oh... I must document this immediately.", 5),
    ("You are Thorne, a hunter.", "A white stag appeared in the forest!", "A white stag... I have tracked these woods for decades and never... this is a sign.", 5),
    ("You are Old Bertram, a fisherman.", "The sea serpent returned!", "After all these years... I TOLD them it was real! Quick, come see!", 5),
    ("You are Lydia, a baker.", "A noble wants to buy your recipe!", "My recipe? Goodness! Well, the secret dies with me, but I am flattered!", 5),
    ("You are Garrett, a merchant.", "Gold was found in the nearby hills!", "Gold?! The property values... the trade potential... this changes everything!", 5),
    ("You are Sister Helene, a priestess.", "An ancient relic was found at the chapel!", "Beneath our chapel? All these years... the Divine works in mysterious ways indeed.", 5),
]

def train():
    print("=" * 60)
    print("Stage 3: Emotion Head Training")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    base = base.to(device).eval()
    for p in base.parameters():
        p.requires_grad = False

    hidden_size = base.config.hidden_size
    emotion_head = EmotionHead(hidden_size).to(device)
    print("Emotion head params: " + str(sum(p.numel() for p in emotion_head.parameters())))
    print("Device: " + str(device))
    print("Emotions: " + str(EMOTIONS))

    # Prepare data
    random.seed(42)
    data = EMOTION_DATA.copy()
    random.shuffle(data)
    split = int(len(data) * 0.8)
    train_d = data[:split]
    val_d = data[split:]
    print("Train: " + str(len(train_d)) + ", Val: " + str(len(val_d)))

    # Emotion distribution
    from collections import Counter
    dist = Counter(e for _, _, _, e in train_d)
    print("Train distribution: " + str({EMOTIONS[k]: v for k, v in sorted(dist.items())}))

    optimizer = torch.optim.AdamW(emotion_head.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        emotion_head.train()
        total_loss = 0
        correct = 0
        total = 0
        random.shuffle(train_d)

        for i in range(0, len(train_d), BATCH_SIZE):
            batch = train_d[i:i+BATCH_SIZE]
            texts = []
            labels = []
            for sys_p, user, asst, emo in batch:
                msgs = [{"role": "system", "content": sys_p},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": asst}]
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
                labels.append(emo)

            encoded = tokenizer(texts, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
            label_tensor = torch.tensor(labels, device=device)

            with torch.no_grad():
                outputs = base(**encoded, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]

            logits, _ = emotion_head(hidden)
            loss = criterion(logits, label_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == label_tensor).sum().item()
            total += len(labels)

        train_acc = correct / total
        train_avg = total_loss / (len(train_d) / BATCH_SIZE)

        # Validation
        emotion_head.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_d), BATCH_SIZE):
                batch = val_d[i:i+BATCH_SIZE]
                texts = []
                labels = []
                for sys_p, user, asst, emo in batch:
                    msgs = [{"role": "system", "content": sys_p},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": asst}]
                    texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
                    labels.append(emo)

                encoded = tokenizer(texts, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
                label_tensor = torch.tensor(labels, device=device)
                outputs = base(**encoded, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                logits, _ = emotion_head(hidden)
                loss = criterion(logits, label_tensor)
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == label_tensor).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_avg = val_loss / max(len(val_d) / BATCH_SIZE, 1)

        print("Epoch " + str(epoch+1) + "/" + str(EPOCHS) +
              " | TLoss:" + str(round(train_avg, 3)) +
              " TAcc:" + str(round(train_acc, 3)) +
              " | VLoss:" + str(round(val_avg, 3)) +
              " VAcc:" + str(round(val_acc, 3)))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(emotion_head.state_dict(), os.path.join(OUTPUT_DIR, "emotion_head.pt"))
            print("  -> Best! Acc: " + str(round(val_acc, 3)))

    print("\nDone! Best Val Acc: " + str(round(best_val_acc, 3)))

    # Test predictions
    print("\nSample predictions:")
    emotion_head.eval()
    for sys_p, user, asst, true_emo in val_d[:10]:
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": user},
                {"role": "assistant", "content": asst}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            out = base(**enc, output_hidden_states=True)
            logits, _ = emotion_head(out.hidden_states[-1])
        pred = logits.argmax(dim=-1).item()
        mark = "OK" if pred == true_emo else "MISS"
        print("  [" + mark + "] True:" + EMOTIONS[true_emo] + " Pred:" + EMOTIONS[pred] + " | " + asst[:80])

if __name__ == "__main__":
    train()
