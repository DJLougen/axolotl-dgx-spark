#!/usr/bin/env bash
# Self-contained cloud benchmark on a CLEAN base image (e.g. python:3.11) — installs the exact stack
# that runs on the GB10 Spark, but x86 + CUDA 12.8 (no ptxas hack needed). Proves the recipe transfers.
set -u
echo "================ INSTALL (clean stack, mirrors GB10 axfast) ================"
pip install -q "axolotl==0.17.0" 2>&1 | tail -2
# axolotl pulls torch 2.12+cu130, where bitsandbytes 0.49 can't find libnvJitLink -> 4-bit breaks.
# Pin torch back to 2.9.1+cu128 (mature; bnb works), matching the GB10 Spark stack.
pip install -q --force-reinstall --no-deps torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
pip install -q --no-deps torchao liger-kernel 2>&1 | tail -1
pip install -q kernels 2>&1 | tail -1   # flash-attn2 via kernels-community (no flash-attn build)
# belt-and-suspenders: ensure bnb can find CUDA libnvJitLink from torch's bundled nvidia libs
NVDIR=$(dirname "$(find /usr/local/lib -name 'libnvJitLink.so*' 2>/dev/null | head -1)" 2>/dev/null)
[ -n "$NVDIR" ] && export LD_LIBRARY_PATH="$NVDIR:${LD_LIBRARY_PATH:-}"
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1
python -c "import bitsandbytes; print('bnb loads OK')" 2>&1 | tail -1

echo "================ ENV ================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
python - <<'PY'
import torch
print("torch",torch.__version__,"cuda",torch.version.cuda,"cap",torch.cuda.get_device_capability(0),torch.cuda.get_device_name(0))
for m in ["axolotl","liger_kernel","bitsandbytes","transformers","peft","pydantic","kernels"]:
    try: mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e: print(m,"ERR",type(e).__name__)
PY

mkdir -p data results
python scripts/gen_data.py --n 600 --out data/bench_chat.jsonl --sentences 22
COMMON="--warm-steps 4 --measure-steps 24 --mbs 8 --seq 1024 --timeout 900"

echo "================ VANILLA (SDPA, no kernels) ================"
python scripts/bench.py --config configs/qwen06b_vanilla.yml --tag cloud_vanilla $COMMON \
  --out results/cloud_vanilla.json --extra="--logging-steps=50" || echo "VANILLA FAILED"

echo "================ OPTIMIZED (flash-attn2 via kernels-community + Liger) ================"
python scripts/bench.py --config configs/qwen06b_optimized.yml --tag cloud_opt $COMMON \
  --out results/cloud_opt.json \
  --extra="--flash-attention=true --sdp-attention=false --lora-mlp-kernel=false --lora-qkv-kernel=false --lora-o-kernel=false --logging-steps=50" || echo "OPT FAILED"

echo "================ OPTIMIZED + LoRA Triton kernels ================"
python scripts/bench.py --config configs/qwen06b_optimized.yml --tag cloud_opt_kernels $COMMON \
  --out results/cloud_opt_kernels.json \
  --extra="--flash-attention=true --sdp-attention=false --logging-steps=50" || echo "OPT+KERNELS FAILED"

echo "================ CLOUD RESULTS ================"
for f in cloud_vanilla cloud_opt cloud_opt_kernels; do
  [ -f results/$f.json ] && echo "$f: $(grep -oE '\"total_tok_per_s\": [0-9.]+|\"peak_reserved_gib\": [0-9.]+|\"final_loss\": [0-9.]+' results/$f.json | tr '\n' ' ')"
done
echo "================ DONE ================"
