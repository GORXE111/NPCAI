"""
Fix Qwen3.5 MPS dtype mismatch in GatedDeltaNet (linear_attn).
Problem: .float() calls inside forward create mixed float32/float16 tensors,
MPS Metal backend crashes on mixed-dtype matrix multiplication.
Solution: Wrap GatedDeltaNet.forward to force all computation in float32.
"""
import torch
import functools


def patch_qwen35_for_mps(model):
    """
    Monkey-patch all Qwen3_5GatedDeltaNet layers to run in float32 on MPS.
    Also force all model parameters to float32 to prevent backward pass dtype issues.
    """
    # Force ENTIRE model to float32
    model = model.float()

    patched = 0
    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3_5GatedDeltaNet":
            _patch_deltanet(module)
            # Force all params and buffers to float32
            for p in module.parameters():
                p.data = p.data.float()
            for b in module.buffers():
                b.data = b.data.float()
            patched += 1
    print("Patched " + str(patched) + " GatedDeltaNet layers for MPS float32")
    return model


def _patch_deltanet(module):
    """Wrap the forward method to ensure float32 throughout."""
    original_forward = module.forward

    @functools.wraps(original_forward)
    def safe_forward(hidden_states, cache_params=None, attention_mask=None, **kwargs):
        input_dtype = hidden_states.dtype

        # Cast all module parameters and buffers to float32
        original_params = {}
        for pname, p in module.named_parameters():
            if p.dtype != torch.float32:
                original_params[pname] = p.dtype
                p.data = p.data.float()
        for bname, b in module.named_buffers():
            if b.dtype != torch.float32:
                b.data = b.data.float()

        # Cast inputs to float32
        hidden_states = hidden_states.float()
        if attention_mask is not None and attention_mask.dtype != torch.float32:
            if attention_mask.is_floating_point():
                attention_mask = attention_mask.float()

        # Run forward in float32
        output = original_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)

        # Cast output back to original dtype
        if isinstance(output, tuple):
            output = tuple(o.to(input_dtype) if isinstance(o, torch.Tensor) and o.is_floating_point() else o for o in output)
        elif isinstance(output, torch.Tensor):
            output = output.to(input_dtype)

        return output

    module.forward = safe_forward


def test_patch():
    """Test that the patch fixes MPS inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen3.5-0.8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply patch
    model = patch_qwen35_for_mps(model)

    device = torch.device("mps")
    model = model.to(device).eval()

    # Test 1: Single inference
    print("\nTest 1: Single inference...")
    inputs = tokenizer("Hello, who are you?", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    print("  Hidden shape: " + str(out.hidden_states[-1].shape))
    print("  Hidden dtype: " + str(out.hidden_states[-1].dtype))

    # Test 2: Loss computation
    print("\nTest 2: Loss computation...")
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        out = model(**inputs, labels=labels)
    print("  Loss: " + str(out.loss.item()))
    is_nan = torch.isnan(out.loss).item()
    print("  Is NaN: " + str(is_nan))

    # Test 3: Batch inference
    print("\nTest 3: Batch inference...")
    texts = ["Hello!", "What do you sell?", "Any news?"]
    batch = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
    print("  Batch hidden shape: " + str(out.hidden_states[-1].shape))

    # Test 4: Backward pass (for training)
    print("\nTest 4: Backward pass...")
    model.enable_input_require_grads()
    inputs = tokenizer("Test backward", return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()
    out = model(**inputs, labels=labels)
    if not torch.isnan(out.loss):
        out.loss.backward()
        print("  Backward OK!")
    else:
        print("  Loss is NaN, backward skipped")

    print("\nAll tests passed!" if not is_nan else "\nLoss NaN issue persists - need deeper fix")


if __name__ == "__main__":
    test_patch()
