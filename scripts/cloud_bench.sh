#!/usr/bin/env bash
# Cloud-GPU portability benchmark. Runs inside the official axolotl Docker image on any x86 CUDA GPU
# (HF Jobs / RunPod / Lambda / local docker). Proves the optimized recipe transfers off the GB10 and
# measures vanilla vs optimized (flash + Liger). NO TRITON_PTXAS_PATH needed on mainstream archs.
set -u
echo "================ ENV ================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))
for m in ["axolotl","flash_attn","liger_kernel","bitsandbytes","transformers","peft"]:
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
PY
python -m pip install -q liger-kernel 2>/dev/null || true   # ensure Liger present

mkdir -p data results
python scripts/gen_data.py --n 600 --out data/bench_chat.jsonl --sentences 22

COMMON="--warm-steps 4 --measure-steps 24 --mbs 8 --seq 1024 --timeout 900"
echo "================ VANILLA (SDPA, no kernels) ================"
python scripts/bench.py --config configs/qwen06b_vanilla.yml --tag cloud_vanilla $COMMON \
  --out results/cloud_vanilla.json --extra="--logging-steps=50" || echo "VANILLA FAILED"

echo "================ OPTIMIZED (flash-attn2 + Liger FLCE/RMSNorm/RoPE) ================"
python scripts/bench.py --config configs/qwen06b_optimized.yml --tag cloud_opt $COMMON \
  --out results/cloud_opt.json \
  --extra="--flash-attention=true --sdp-attention=false --lora-mlp-kernel=false --lora-qkv-kernel=false --lora-o-kernel=false --logging-steps=50" || echo "OPT FAILED"

echo "================ OPTIMIZED + LoRA Triton kernels (axolotl-native, if supported) ================"
python scripts/bench.py --config configs/qwen06b_optimized.yml --tag cloud_opt_kernels $COMMON \
  --out results/cloud_opt_kernels.json \
  --extra="--flash-attention=true --sdp-attention=false --logging-steps=50" || echo "OPT+KERNELS FAILED (kernels may be version-specific)"

echo "================ CLOUD RESULTS ================"
for f in cloud_vanilla cloud_opt cloud_opt_kernels; do
  [ -f results/$f.json ] && echo "$f: $(grep -oE '\"total_tok_per_s\": [0-9.]+|\"peak_reserved_gib\": [0-9.]+|\"final_loss\": [0-9.]+' results/$f.json | tr '\n' ' ')"
done
echo "================ DONE ================"
