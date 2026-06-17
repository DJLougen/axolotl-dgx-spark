#!/usr/bin/env python3
"""Probe GB10 viability for QLoRA path: bnb 4-bit quant, liger kernels, sdpa attention."""
import warnings
warnings.filterwarnings("ignore")
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cap", torch.cuda.get_device_capability(0))

# 1) bitsandbytes 4-bit quant/dequant on GPU
try:
    import bitsandbytes.functional as F
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    q, state = F.quantize_4bit(x)
    deq = F.dequantize_4bit(q, state)
    err = (deq.float() - x.float()).abs().mean().item()
    print(f"[bnb] 4bit quant/dequant OK  mean_abs_err={err:.4f}")
except Exception as e:
    print(f"[bnb] 4bit FAILED: {type(e).__name__}: {e}")

# 2) bnb 4-bit Linear forward+backward (the actual QLoRA base path)
try:
    import bitsandbytes as bnb
    lin = bnb.nn.Linear4bit(512, 512, bias=False, compute_dtype=torch.bfloat16).cuda()
    inp = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = lin(inp)
    out.sum().backward()
    print(f"[bnb] Linear4bit fwd+bwd OK  out={tuple(out.shape)} grad_ok={inp.grad is not None}")
except Exception as e:
    print(f"[bnb] Linear4bit FAILED: {type(e).__name__}: {e}")

# 3) liger kernels import + a triton RMSNorm on sm_121
try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    rms = LigerRMSNorm(512).cuda().to(torch.bfloat16)
    y = rms(torch.randn(8, 128, 512, device="cuda", dtype=torch.bfloat16))
    print(f"[liger] RMSNorm triton kernel OK on sm_121  out={tuple(y.shape)}")
except Exception as e:
    print(f"[liger] FAILED: {type(e).__name__}: {e}")

# 4) SDPA attention (the practical attention backend; flash-attn2 unavailable on aarch64)
try:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    q = torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
    o = sdpa(q, q, q, is_causal=True)
    print(f"[sdpa] OK  out={tuple(o.shape)}")
except Exception as e:
    print(f"[sdpa] FAILED: {type(e).__name__}: {e}")
print("DONE")
