#!/usr/bin/env bash
# Sweep #5: can torch.compile close the gap when NOT blocked by custom-autograd LoRA kernels?
# (compile graph-breaks on lora_*_kernel; test compile on eager + liger-only paths.) 4-bit, mbs8.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p results
C="--warm-steps 5 --measure-steps 30 --mbs 8 --seq 1024 --timeout 1600"
echo "==== sweep5 start $(date) ===="

# A) pure torch.compile on the eager HF model (no axolotl kernels at all)
"$AXPY" scripts/bench.py --config configs/qwen06b_vanilla.yml --tag axo_compile_eager $C \
  --out results/axo_compile_eager.json --extra="--torch-compile=true --logging-steps=50"
free -g | head -2

# B) Liger kernels + torch.compile, LoRA Triton kernels OFF (so no graph breaks)
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag axo_liger_compile $C \
  --out results/axo_liger_compile.json \
  --extra="--lora-mlp-kernel=false --lora-qkv-kernel=false --lora-o-kernel=false --torch-compile=true --logging-steps=50"
free -g | head -2

echo "==== sweep5 done $(date) ===="
