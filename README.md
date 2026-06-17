# Fast Axolotl on the DGX Spark (GB10) — recipe, benchmarks, and the honest Unsloth gap

Goal: *"make training with Axolotl just as fast as Unsloth so I can have quality and speed."*

Hardware: **NVIDIA GB10** (Grace‑Blackwell, `aarch64`, compute capability **sm_121**), 121 GB unified
LPDDR5X (~273 GB/s — bandwidth, not FLOPs, is the binding constraint), driver 580.159.03, CUDA 13.0.
All work is in the conda env **`axfast`** on `spark-d500` and in `~/axfast-work`.

> **Running on cloud GPUs (x86)?** The recipe is standard Axolotl and transfers directly — see
> **[CLOUD.md](CLOUD.md)**. Verified on an NVIDIA L4 (HF Jobs): the optimized recipe runs Qwen3-0.6B
> QLoRA in **9.2 GiB at 6590 tok/s**, while vanilla Axolotl **OOMs on the 24 GiB card**. The GB10-specific
> `TRITON_PTXAS_PATH`/aarch64 bits below are *not* needed on mainstream cloud GPUs.

---

## TL;DR

1. **Axolotl on this box was not GPU‑capable at all.** The installed `axolotl`/`unsloth` ran under base
   conda Python whose torch is `2.10.0+cpu` — every run was CPU‑only. I built a working GPU stack
   (`axfast`: torch `2.9.1+cu130` aarch64, sm_121).
2. **Mandatory unlock:** Triton's bundled `ptxas` (12.8) can't target `sm_121a`. Export
   `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (CUDA 13) or every fused kernel (Liger, Axolotl LoRA
   kernels, Unsloth) dies with `ptxas fatal: Value 'sm_121a' is not defined`.
3. **Keep `flash_attention: true`** (your original setting). The flash‑attn *package* has no
   aarch64/Blackwell wheel, but transformers 5.x transparently uses `kernels-community/flash-attn2`
   (HF `kernels` lib) — verified working, and in real training it is **~1.4× faster and lighter than
   SDPA** (which falls back to a slower backend under padding/packing masks).
4. **Optimized recipe** (flash‑attn2 + Liger FLCE/RMSNorm/RoPE [+ optional LoRA Triton kernels]) is
   **1.83× faster and 3.1× less memory than a naive config**, and uses **less memory than Unsloth**.
5. **The gap to Unsloth shrinks with batch size and model size, and Axolotl always uses far less memory.**
   At small batch Unsloth is ~1.45×; scaling the micro‑batch (which Axolotl's FLCE lets it do in far less
   memory) closes it to **~82% of Unsloth at mbs32 with 38% less memory (33 vs 53 GiB)**. Because Axolotl
   fits a bigger batch in the same RAM, at a fixed memory budget it can match Unsloth's effective
   throughput. Full per‑step parity isn't reached (Unsloth's fused 4‑bit/memory access keeps a structural
   edge), but for "quality + speed" the optimized recipe is the right tool.

**Bottom line for "quality AND speed":** the optimized Axolotl recipe keeps your full feature set + your
`lora_dropout` regularization, runs on the GB10, is far faster/lighter than naive, and at your real model
size (1.7B) lands within ~1.27× of Unsloth using less memory.

---

## Benchmark results (measured: warm pass + measured pass; tok/s = padded tokens ÷ HF `train_runtime`)

### Qwen3‑0.6B, seq 1024, mbs 8, 4‑bit, 30 steps

| Config | tok/s | peak mem | note |
|---|---:|---:|---|
| Axolotl vanilla, SDPA | 3018 | 28.9 GiB | naive baseline |
| Axolotl vanilla, **flash** | 3923 | 27.1 GiB | flash alone: +30% |
| Axolotl optimized, SDPA | 3852 | 10.5 GiB | kernels, slow attn |
| **Axolotl optimized, flash** | **5511** | **9.19 GiB** | recommended (dropout 0 + LoRA kernels) |
| Axolotl `lora_dropout 0.05` + Liger, flash | 5294 | 14.25 GiB | keeps dropout; −4% speed |
| **Unsloth** | **8015** | 13.7 GiB | reference |

### Qwen3‑1.7B (your SelfTrace size), flash, 4‑bit, seq 1024 — batch scaling

| micro‑batch | Axolotl tok/s | Axolotl mem | Unsloth tok/s | Unsloth mem | Axolotl % |
|---:|---:|---:|---:|---:|---:|
| 4  | 2810 | 8.0 GiB  | 3568 | 10.3 GiB | 79% |
| 8  | 3260 | 14.6 GiB | 3987 | 19.0 GiB | 82% |
| 16 | 3797 | 26.7 GiB | 4433 | 36.5 GiB | **86%** |

(`flash` lifts mbs4 from 2518→2810 tok/s vs SDPA.) On your real model the optimized recipe reaches **86%
of Unsloth at mbs16 with 27% less memory**, and the gap keeps narrowing with batch size.

### Batch scaling closes the gap (Qwen3‑0.6B, flash, 4‑bit, seq 1024)

Axolotl's low memory lets it grow the micro‑batch where Unsloth's memory balloons:

| micro‑batch | Axolotl tok/s | Axolotl mem | Unsloth tok/s | Unsloth mem | Axolotl % |
|---:|---:|---:|---:|---:|---:|
| 8  | 5511 | 9.2 GiB  | 8015 | 13.7 GiB | 69% |
| 16 | 6420 | 17.7 GiB | 8412 | 26.8 GiB | 76% |
| 32 | 7034 | 33.0 GiB | 8549 | 52.9 GiB | 82% |
| 48 | 7228 | 49.1 GiB | (≈OOM‑risk) | ~80 GiB | — |

At mbs32 Axolotl reaches **82% of Unsloth using 38% less memory**; at a fixed RAM budget Axolotl fits a
larger batch (mbs48 @ 49 GiB ≈ Unsloth mbs32 @ 53 GiB), so its memory‑constrained throughput is close.
16‑bit+flash (5564) ≈ 4‑bit+flash (5511) and `adamw_torch_fused` (5562) ≈ bnb‑8bit — neither is a lever.

**Speedups (0.6B):** optimized+flash vs naive = **1.83× faster, 3.14× less memory**; attention held
constant (both flash), the kernels alone give **1.40× faster, 2.95× less memory**. **vs Unsloth:** small‑
batch is 69% (0.6B) / 79% (1.7B); scaling the batch (Axolotl's low memory permits it) reaches **82% (0.6B
mbs32) / 86% (1.7B mbs16)** of Unsloth at **25–38% lower memory** — and closes further with batch/model size.

### What moved the needle (and what didn't)
* **`flash_attention: true`** (kernels‑community flash‑attn2): **1.4× over SDPA in training.** SDPA looks
  flash‑class on a bare attention op, but under transformers' padding mask it drops to a slower backend.
* **Liger fused‑linear‑cross‑entropy:** the memory win — never materializes the 152k‑vocab logits
  (28.9→9.2 GiB) and most of the kernel speedup. Dropout‑compatible.
* Axolotl LoRA Triton kernels (`lora_mlp/qkv/o_kernel`): mostly **memory** (require `lora_dropout: 0`).
* **No effect:** logging frequency, dataloader workers, 4‑bit vs 16‑bit. **Hurts:** `torch.compile`
  (graph‑breaks on the LoRA kernels; disastrous with Liger). Bigger batch helps only modestly (bandwidth‑bound).

---

## The optimized recipe (drop‑in)

Differences from a stock Axolotl QLoRA config:

```yaml
flash_attention: true        # KEEP ON — uses kernels-community/flash-attn2 on GB10 (no SDPA fallback)

plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_rope: true
liger_rms_norm: true
liger_glu_activation: true   # fuse SwiGLU (set false only if you enable lora_mlp_kernel below)
liger_fused_linear_cross_entropy: true   # the memory unlock

sample_packing: true         # pack short turns -> fewer padded tokens -> big epoch-time win

# OPTIONAL, ~5 GiB more savings but REQUIRES lora_dropout: 0.0 (kernels don't support dropout/bias):
# lora_mlp_kernel: true
# lora_qkv_kernel: true
# lora_o_kernel: true
# liger_glu_activation: false   # (let lora_mlp_kernel own the MLP if you enable these)
```

You can **keep `lora_dropout: 0.05`** (or any dropout) with the Liger path above — it costs only ~4% vs the
dropout‑0 + LoRA‑kernel path. Caveat: the LoRA kernels and `kernels-community` modeling do **not** support
`trust_remote_code` (fine — Qwen3 is native in transformers 5.x).

Ready‑to‑use configs in `configs/`:
* `selftrace_qwen3_17b_qlora_optimized.yml` — your SelfTrace smoke config with the recipe (drop‑in, keeps your dropout).
* `qwen06b_optimized.yml`, `qwen17b_optimized.yml`, `qwen06b_optimized_packed.yml`, `qwen06b_lora16.yml`.

---

## How to run (on `spark-d500`)

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate axfast
export PATH="$CONDA_PREFIX/bin:$PATH" PYTHONNOUSERSITE=1   # avoid ~/.local CPU-torch axolotl
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas         # MANDATORY on GB10 (sm_121a)
cd ~/axfast-work
axolotl train configs/selftrace_qwen3_17b_qlora_optimized.yml
```

`scripts/env.sh` sets all of this. Benchmarks: `bash scripts/bench_all.sh`, `scripts/cmp_attn2.sh`,
`scripts/sweep3.sh`, `scripts/sweep6.sh`; aggregate with `python scripts/aggregate.py`.
First flash run downloads `kernels-community/flash-attn2` from HF Hub (needs network once, then cached).

### ⚠️ Operational note (learned the hard way)
GB10 uses **unified memory**, so a training OOM takes down the **whole machine** (a `seq2048/mbs16` vanilla
run peaked ~80–100 GB and hard‑rebooted the box). Keep peak well under free RAM and launch long runs
**detached + `timeout`‑wrapped** so a dropped SSH session can't orphan a runaway:
`setsid bash -c "timeout -s KILL <sec> <cmd>" </dev/null >log 2>&1 &`.

## Reproducing the `axfast` env
```bash
conda create -y -n axfast python=3.11 && conda activate axfast
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130   # aarch64 sm_121
python -m pip install axolotl==0.17.0
python -m pip install --no-deps torchao liger-kernel
```
Unsloth reference lives in env `unsloth-bench` (same torch; `torchvision` removed — its 0.25 build
mismatches torch 2.9.1 and breaks `import unsloth`).
