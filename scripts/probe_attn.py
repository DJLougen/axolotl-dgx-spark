#!/usr/bin/env python3
"""Which SDPA backend does GB10 use, and is it fused (flash-class) or slow math?
Times each available backend for a realistic Qwen3 attention shape, plus flex_attention."""
import warnings, time
warnings.filterwarnings("ignore")
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.nn.attention import SDPBackend, sdpa_kernel

dev = "cuda"
B, Hq, Hkv, S, D = 4, 16, 8, 1024, 128  # Qwen3-1.7B-ish: GQA 16 q / 8 kv heads, head_dim 128
dt = torch.bfloat16

print("torch", torch.__version__, "cap", torch.cuda.get_device_capability(0))
print("cudnn version:", torch.backends.cudnn.version())
print("global SDPA flags: flash=%s cudnn=%s mem_eff=%s math=%s" % (
    torch.backends.cuda.flash_sdp_enabled(),
    torch.backends.cuda.cudnn_sdp_enabled(),
    torch.backends.cuda.mem_efficient_sdp_enabled(),
    torch.backends.cuda.math_sdp_enabled()))

def make():
    q = torch.randn(B, Hq, S, D, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(B, Hkv, S, D, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(B, Hkv, S, D, device=dev, dtype=dt, requires_grad=True)
    return q, k, v

def bench(fn, iters=50):
    q, k, v = make()
    for _ in range(5):  # warm
        o = fn(q, k, v); o.sum().backward(); q.grad = k.grad = v.grad = None
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters):
        o = fn(q, k, v); o.sum().backward(); q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1e3  # ms/iter (fwd+bwd)

def run_backend(name, backend):
    try:
        def fn(q, k, v):
            with sdpa_kernel([backend]):
                # GQA: expand kv heads to match q for the kernels that need it
                kk = k.repeat_interleave(Hq // Hkv, dim=1)
                vv = v.repeat_interleave(Hq // Hkv, dim=1)
                return sdpa(q, kk, vv, is_causal=True)
        ms = bench(fn)
        print(f"  {name:18s}: {ms:6.2f} ms/iter (fwd+bwd)")
    except Exception as e:
        print(f"  {name:18s}: UNAVAILABLE ({type(e).__name__}: {str(e)[:60]})")

print("\nSDPA backends (fwd+bwd, lower=better):")
run_backend("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION)
run_backend("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION)
run_backend("EFFICIENT_ATTN", SDPBackend.EFFICIENT_ATTENTION)
run_backend("MATH", SDPBackend.MATH)

# default auto-selection
def fn_auto(q, k, v):
    kk = k.repeat_interleave(Hq // Hkv, dim=1); vv = v.repeat_interleave(Hq // Hkv, dim=1)
    return sdpa(q, kk, vv, is_causal=True)
print(f"  {'AUTO (default)':18s}: {bench(fn_auto):6.2f} ms/iter")

# flex_attention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    fa = torch.compile(flex_attention)
    def causal(b, h, qi, ki): return qi >= ki
    bm = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=dev)
    def fn_flex(q, k, v):
        kk = k.repeat_interleave(Hq // Hkv, dim=1); vv = v.repeat_interleave(Hq // Hkv, dim=1)
        return fa(q, kk, vv, block_mask=bm)
    print(f"  {'FLEX_ATTENTION':18s}: {bench(fn_flex, iters=30):6.2f} ms/iter")
except Exception as e:
    print(f"  {'FLEX_ATTENTION':18s}: UNAVAILABLE ({type(e).__name__}: {str(e)[:80]})")
print("DONE")
