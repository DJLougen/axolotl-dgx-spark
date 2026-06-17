#!/usr/bin/env python3
"""Quick check that Unsloth loads a 4-bit Qwen3-0.6B + LoRA and runs 1 step on GB10."""
import warnings, os
warnings.filterwarnings("ignore")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cap", torch.cuda.get_device_capability(0), flush=True)

from unsloth import FastLanguageModel
print("unsloth imported OK", flush=True)

model, tok = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-0.6B", max_seq_length=1024, load_in_4bit=True, dtype=None,
)
print("model loaded 4bit OK", flush=True)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing=False, random_state=1234,
)
print("LoRA attached OK", flush=True)

# one fwd/bwd step
ids = torch.randint(0, 1000, (2, 512), device="cuda")
out = model(input_ids=ids, labels=ids)
out.loss.backward()
print(f"1 step OK  loss={out.loss.item():.3f}  peak_resv={torch.cuda.max_memory_reserved()/1e9:.2f}GB", flush=True)
print("UNSLOTH_SMOKE_OK", flush=True)
