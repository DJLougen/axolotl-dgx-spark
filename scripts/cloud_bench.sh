#!/usr/bin/env bash
# Cloud-GPU portability benchmark: proves the optimized Axolotl recipe transfers off the GB10.
# Designed for the official axolotl image (pin a CUDA-12.8 tag for stable flash-attn/bnb), runnable
# on any x86 CUDA GPU (HF Jobs / RunPod / Lambda / local docker). NO TRITON_PTXAS_PATH on mainstream archs.
set -u

# Defensive: some axolotl images ship a torchvision whose ABI mismatches torch (crashes
# `import transformers` via torchvision::nms). LLM training doesn't need vision.
python -c "import torchvision" 2>/dev/null || pip uninstall -y torchvision >/dev/null 2>&1 || true
# Defensive: if bitsandbytes can't find CUDA libnvJitLink, point it at torch's bundled copy.
python -c "import bitsandbytes" 2>/dev/null || {
  LNV=$(find /root /usr /opt -name "libnvJitLink.so*" 2>/dev/null | head -1)
  [ -n "$LNV" ] && export LD_LIBRARY_PATH="$(dirname "$LNV"):${LD_LIBRARY_PATH:-}" && echo "bnb fix: + $(dirname "$LNV")"
}
# Ensure Liger present WITHOUT touching other deps (a full install can downgrade pydantic and break axolotl).
python -c "import liger_kernel" 2>/dev/null || pip install -q --no-deps liger-kernel >/dev/null 2>&1 || true

echo "================ ENV ================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))
for m in ["axolotl","flash_attn","liger_kernel","bitsandbytes","transformers","peft","pydantic"]:
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
PY

# Pick the fastest available attention: real flash-attn package if present, else SDPA.
if python -c "import flash_attn" 2>/dev/null; then ATTN="--flash-attention=true --sdp-attention=false"; echo "ATTN=flash-attn2 (package)"; else ATTN="--flash-attention=false --sdp-attention=true"; echo "ATTN=sdpa (no flash-attn pkg)"; fi

mkdir -p data results
python scripts/gen_data.py --n 600 --out data/bench_chat.jsonl --sentences 22
COMMON="--warm-steps 4 --measure-steps 24 --mbs 8 --seq 1024 --timeout 900"

echo "================ VANILLA (SDPA, no kernels) ================"
python scripts/bench.py --config configs/qwen06b_vanilla.yml --tag cloud_vanilla $COMMON \
  --out results/cloud_vanilla.json --extra="--logging-steps=50" || echo "VANILLA FAILED"

echo "================ OPTIMIZED (Liger FLCE/RMSNorm/RoPE + best attention) ================"
python scripts/bench.py --config configs/qwen06b_optimized.yml --tag cloud_opt $COMMON \
  --out results/cloud_opt.json \
  --extra="$ATTN --lora-mlp-kernel=false --lora-qkv-kernel=false --lora-o-kernel=false --logging-steps=50" || echo "OPT FAILED"

echo "================ OPTIMIZED + LoRA Triton kernels ================"
python scripts/bench.py --config configs/qwen06b_optimized.yml --tag cloud_opt_kernels $COMMON \
  --out results/cloud_opt_kernels.json --extra="$ATTN --logging-steps=50" || echo "OPT+KERNELS FAILED"

echo "================ CLOUD RESULTS ================"
for f in cloud_vanilla cloud_opt cloud_opt_kernels; do
  [ -f results/$f.json ] && echo "$f: $(grep -oE '\"total_tok_per_s\": [0-9.]+|\"peak_reserved_gib\": [0-9.]+|\"final_loss\": [0-9.]+' results/$f.json | tr '\n' ' ')"
done
echo "================ DONE ================"
