#!/usr/bin/env bash
# Corrected best-Axolotl with flash_attention:true (kernels-community flash-attn2 on GB10).
# 30-step warm+measure, Qwen3-0.6B mbs8/seq1024.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
echo "==== cmp_attn2 start $(date) ===="

# Full optimized (Liger + LoRA kernels, dropout 0) + flash-attn2
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag opt_flash \
  --warm-steps 5 --measure-steps 30 --mbs 8 --seq 1024 --timeout 900 --out results/opt_flash.json \
  --extra="--flash-attention=true --sdp-attention=false --logging-steps=50"

# Dropout-preserving recipe (Liger incl. SwiGLU, dropout 0.05, NO lora kernels) + flash-attn2
# == exactly the shipped selftrace recipe; also verifies liger_glu_activation:true works.
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag drop_flash \
  --warm-steps 5 --measure-steps 30 --mbs 8 --seq 1024 --timeout 900 --out results/drop_flash.json \
  --extra="--flash-attention=true --sdp-attention=false --lora-dropout=0.05 --lora-mlp-kernel=false --lora-qkv-kernel=false --lora-o-kernel=false --liger-glu-activation=true --logging-steps=50"

echo "==== cmp_attn2 done $(date) ===="
