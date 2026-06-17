# Running this on cloud GPUs (x86) — verified on an NVIDIA L4

The recipe is **standard Axolotl** (flash-attn2 + Liger + LoRA kernels + packing), so it transfers to any
mainstream CUDA GPU. The GB10-specific bits in the main README (aarch64 torch, the `TRITON_PTXAS_PATH`
ptxas hack, the unified-memory-OOM-reboots-the-box warning) are **not needed** on cloud x86 GPUs.

## Verified result (Hugging Face Jobs, 1× L4, sm_89, 23 GiB, $0.80/hr)

Qwen3-0.6B QLoRA, seq 1024, mbs 8, 24 measured steps. Reproduce with one command (below).

| Config | tok/s | peak mem | result |
|---|---:|---:|---|
| Axolotl vanilla (SDPA, no kernels) | — | — | **CUDA OOM** (needs ~29 GiB > 23 GiB) |
| Axolotl optimized (flash + Liger) | 6433 | 11.8 GiB | ✓ |
| **Axolotl optimized + LoRA kernels** | **6590** | **9.2 GiB** | ✓ |

Two takeaways:
1. **The optimized recipe is what makes Qwen3 QLoRA *fit* on a commodity 24 GiB cloud GPU** — vanilla
   Axolotl OOMs (no fused cross-entropy → it materializes the 152k-vocab logits); the recipe runs in 9.2 GiB.
2. **It's not GB10-only.** The exact same config is actually *faster* on the L4 (6590 tok/s) than on the
   GB10 (5511) at identical memory — the L4's higher memory bandwidth helps. The recipe is portable and fast.

## What changes vs the GB10 setup

| | GB10 (DGX Spark) | x86 cloud GPU |
|---|---|---|
| torch | `2.9.1+cu130` aarch64 | `2.9.1+cu128` (or any cu12x) x86 |
| `TRITON_PTXAS_PATH` | **required** (sm_121a) | **not needed** (bundled ptxas handles sm_80/86/89/90) |
| flash attention | `kernels-community/flash-attn2` (no aarch64 wheel) | real `flash-attn` wheel **or** kernels-community |
| OOM behavior | reboots the whole box (unified memory) | just crashes the process |
| recipe / configs | identical | identical |

## One-command reproduction (HF Jobs)

Needs an HF token with the **Jobs** permission (`job.write`) and pay-as-you-go billing enabled.

```bash
hf jobs run --flavor l4x1 --timeout 50m python:3.11 \
  bash -c "git clone --depth 1 https://github.com/DJLougen/axoFast /w && cd /w && bash scripts/cloud_setup_and_bench.sh"
```

`scripts/cloud_setup_and_bench.sh` installs the stack on a clean base and benchmarks vanilla vs optimized.
Swap `--flavor` for `a10g-small`, `l40sx1`, `a100-large`, `h200`, etc. (`hf jobs hardware` lists them).

## Bare cloud box (RunPod / Lambda / your own `docker run --gpus all`)

The official `axolotlai/axolotl` images were broken at time of writing (cu130 builds: missing flash-attn,
bitsandbytes can't find `libnvJitLink.so.13`, pydantic 1.x). The reliable path is a clean install (what
`cloud_setup_and_bench.sh` does):

```bash
pip install "axolotl==0.17.0"
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # pin AFTER axolotl (it pulls cu130)
pip install --no-deps torchao liger-kernel
pip install kernels                                                            # flash-attn2 via kernels-community
# let bitsandbytes find torch's bundled CUDA-12 libs:
export LD_LIBRARY_PATH="$(python -c "import os,glob,nvidia;b=os.path.dirname(nvidia.__file__);print(':'.join(sorted({os.path.dirname(p) for p in glob.glob(b+'/**/*.so*',recursive=True)})))")":$LD_LIBRARY_PATH
axolotl train configs/qwen06b_optimized.yml   # flash_attention:true + Liger + LoRA kernels
```

If you have a normal CUDA-12 base image with system CUDA libs (e.g. `nvidia/cuda:12.4`), the
`LD_LIBRARY_PATH` line is unnecessary, and you can `pip install flash-attn` for the native package
(prebuilt wheels exist for sm_80/86/89/90) instead of the `kernels` fallback.
