"""
npc_model.py - NPC-Augmented Language Model
Architecture: Base LLM + LoRA + Memory Cross-Attention + Emotion Head

Three-stage training:
  Stage 1: LoRA adapters (freeze base + memory + emotion)
  Stage 2: Memory module + gate (freeze base + LoRA)
  Stage 3: Emotion head (freeze everything else)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


# ══════════════════════════════════════════════════════
# Memory Cross-Attention Layer
# ══════════════════════════════════════════════════════

class MemoryEncoder(nn.Module):
    """Encode text memories into fixed-size vectors."""

    def __init__(self, embed_dim, memory_dim=256):
        super().__init__()
        self.proj = nn.Linear(embed_dim, memory_dim)

    def encode(self, memory_embeds):
        """
        Args:
            memory_embeds: (batch, num_memories, seq_len, embed_dim)
                           - pre-embedded memory texts
        Returns:
            (batch, num_memories, memory_dim)
        """
        # Mean pool over sequence length
        pooled = memory_embeds.mean(dim=2)  # (batch, num_memories, embed_dim)
        return self.proj(pooled)             # (batch, num_memories, memory_dim)


class MemoryAttentionLayer(nn.Module):
    """Cross-attention from hidden states to memory bank."""

    def __init__(self, hidden_size, memory_dim=256, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            kdim=memory_dim,
            vdim=memory_dim,
            batch_first=True
        )
        self.gate = nn.Parameter(torch.zeros(1))  # Learnable gate, init 0
        self.memory_proj = nn.Linear(memory_dim, memory_dim)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, memory_bank):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            memory_bank:   (batch, num_memories, memory_dim)
        Returns:
            (batch, seq_len, hidden_size)
        """
        if memory_bank is None or memory_bank.shape[1] == 0:
            return hidden_states

        memory = self.memory_proj(memory_bank)
        attn_out, _ = self.cross_attn(
            self.layer_norm(hidden_states), memory, memory
        )
        # Gated residual: gate starts at 0, doesn't break original model
        return hidden_states + torch.sigmoid(self.gate) * attn_out


class MemoryAugmentedLayer(nn.Module):
    """Wraps an original transformer layer with memory cross-attention.
    Proxies all attribute access to the original layer for compatibility."""

    def __init__(self, original_layer, memory_attn):
        super().__init__()
        self.original_layer = original_layer
        self.memory_attn = memory_attn

    def __getattr__(self, name):
        """Proxy attribute access to original layer for compatibility with transformers."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def forward(self, hidden_states, **kwargs):
        # Remove memory_bank from kwargs before passing to original layer
        memory_bank = kwargs.pop('memory_bank', None)

        # Run original layer
        outputs = self.original_layer(hidden_states, **kwargs)

        # Extract hidden states (handle tuple output)
        if isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        # Apply memory cross-attention
        if memory_bank is not None:
            hidden = self.memory_attn(hidden, memory_bank)

        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden


# ══════════════════════════════════════════════════════
# Emotion Head
# ══════════════════════════════════════════════════════

class EmotionHead(nn.Module):
    """Classifies NPC emotion from last hidden state."""

    EMOTIONS = ['neutral', 'happy', 'angry', 'sad', 'fearful',
                'surprised', 'disgusted', 'contemptuous']

    def __init__(self, hidden_size, num_emotions=8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_emotions)
        )
        # Emotion embedding for next-turn injection
        self.emotion_embed = nn.Embedding(num_emotions, hidden_size)

    def forward(self, last_hidden_state):
        """
        Args:
            last_hidden_state: (batch, seq_len, hidden_size)
        Returns:
            logits: (batch, num_emotions)
            emotion_vec: (batch, hidden_size) - for next turn injection
        """
        cls_repr = last_hidden_state[:, -1, :]    # last token
        logits = self.classifier(cls_repr)
        emotion_id = logits.argmax(dim=-1)
        emotion_vec = self.emotion_embed(emotion_id)
        return logits, emotion_vec

    @staticmethod
    def get_emotion_name(idx):
        return EmotionHead.EMOTIONS[idx] if idx < len(EmotionHead.EMOTIONS) else 'unknown'


# ══════════════════════════════════════════════════════
# Complete NPC Model
# ══════════════════════════════════════════════════════

class NPCModel(nn.Module):
    """
    Full NPC-augmented language model:
    Base LLM + LoRA + Memory Cross-Attention + Emotion Head
    """

    def __init__(self, model_name, memory_dim=256, num_memory_layers=4,
                 lora_rank=8, lora_alpha=16, load_in_4bit=True):
        super().__init__()

        # Load base model
        print(f"Loading base model: {model_name}")
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # Determine device
        if torch.cuda.is_available():
            device_map = "auto"
            model_dtype = torch.float16
        elif torch.backends.mps.is_available():
            device_map = None  # Manual .to("mps") later
            model_dtype = torch.float16  # MPS supports float16, saves memory
            quantization_config = None   # bitsandbytes not supported on MPS
        else:
            device_map = None
            model_dtype = torch.float32
            quantization_config = None

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            dtype=model_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_size = self.base_model.config.hidden_size
        embed_dim = self.base_model.config.hidden_size

        # LoRA
        print("Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()

        # Memory modules
        print("Adding Memory Cross-Attention...")
        self.memory_encoder = MemoryEncoder(embed_dim, memory_dim)
        self.memory_dim = memory_dim
        self.num_memory_layers = num_memory_layers

        # Insert memory attention into last N layers
        total_layers = len(self.base_model.base_model.model.model.layers)
        self.memory_layer_indices = list(range(
            total_layers - num_memory_layers, total_layers
        ))
        for idx in self.memory_layer_indices:
            original = self.base_model.base_model.model.model.layers[idx]
            mem_attn = MemoryAttentionLayer(hidden_size, memory_dim)
            self.base_model.base_model.model.model.layers[idx] = MemoryAugmentedLayer(
                original, mem_attn
            )

        # Emotion head
        print("Adding Emotion Head...")
        self.emotion_head = EmotionHead(hidden_size)

        print(f"NPCModel ready: {model_name}, memory_dim={memory_dim}, "
              f"memory_layers={num_memory_layers}, lora_rank={lora_rank}")

    def encode_memories(self, memory_texts, max_length=64):
        """
        Encode a list of memory strings into memory vectors.

        Args:
            memory_texts: list of strings (NPC memories)
            max_length: max tokens per memory
        Returns:
            memory_bank: (1, num_memories, memory_dim)
        """
        if not memory_texts:
            return None

        encoded = self.tokenizer(
            memory_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.base_model.device)

        with torch.no_grad():
            embeds = self.base_model.base_model.model.model.embed_tokens(encoded.input_ids)

        # (num_memories, seq, embed) -> unsqueeze batch
        # Ensure memory encoder is on same device and dtype
        device = embeds.device
        self.memory_encoder = self.memory_encoder.to(device=device, dtype=embeds.dtype)
        memory_vecs = self.memory_encoder.encode(embeds.unsqueeze(0))
        return memory_vecs

    def freeze_for_stage(self, stage):
        """
        Freeze parameters for each training stage.
        Stage 1: Train LoRA only
        Stage 2: Train Memory module only
        Stage 3: Train Emotion head only
        """
        # First freeze everything
        for param in self.parameters():
            param.requires_grad = False

        if stage == 1:
            # Unfreeze LoRA parameters
            for name, param in self.base_model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True

        elif stage == 2:
            # Unfreeze memory modules
            self.memory_encoder.requires_grad_(True)
            for idx in self.memory_layer_indices:
                layer = self.base_model.base_model.model.model.layers[idx]
                if isinstance(layer, MemoryAugmentedLayer):
                    layer.memory_attn.requires_grad_(True)

        elif stage == 3:
            # Unfreeze emotion head
            self.emotion_head.requires_grad_(True)

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Stage {stage}: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.3f}%)")

    def save_npc_modules(self, path):
        """Save only the NPC-specific modules (not the base model)."""
        import os
        os.makedirs(path, exist_ok=True)

        # Save LoRA
        self.base_model.save_pretrained(os.path.join(path, "lora"))

        # Save memory modules
        torch.save({
            'memory_encoder': self.memory_encoder.state_dict(),
            'memory_layers': {
                str(idx): self.base_model.base_model.model.model.layers[idx].memory_attn.state_dict()
                for idx in self.memory_layer_indices
                if isinstance(self.base_model.base_model.model.model.layers[idx], MemoryAugmentedLayer)
            }
        }, os.path.join(path, "memory_module.pt"))

        # Save emotion head
        torch.save(self.emotion_head.state_dict(),
                   os.path.join(path, "emotion_head.pt"))

        print(f"NPC modules saved to {path}")


# ══════════════════════════════════════════════════════
# Quick Test
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"

    print("=" * 60)
    print("NPC Model Architecture Test")
    print("=" * 60)

    model = NPCModel(
        model_name=model_name,
        memory_dim=256,
        num_memory_layers=4,
        lora_rank=8,
        load_in_4bit=False  # Use float16 for testing
    )

    # Test memory encoding
    memories = [
        "Player bought a sword yesterday",
        "The merchant mentioned rising taxes",
        "A stranger arrived at the inn last night"
    ]
    memory_bank = model.encode_memories(memories)
    print(f"\nMemory bank shape: {memory_bank.shape}")

    # Test stage freezing
    for stage in [1, 2, 3]:
        model.freeze_for_stage(stage)

    print("\nArchitecture test passed!")
