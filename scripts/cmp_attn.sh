#!/usr/bin/env bash
# Compare attention backends at identical config (Liger, dropout 0, NO lora kernels): flash-attn2
# (kernels-community) vs SDPA. 30-step warm+measure, Qwen3-0.6B mbs8/seq1024.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
NOK="--lora-mlp-kernel=false --lora-qkv-kernel=false --lora-o-kernel=false --logging-steps=50"
echo "==== cmp_attn start $(date) ===="

"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag attn_flash \
  --warm-steps 5 --measure-steps 30 --mbs 8 --seq 1024 --timeout 900 --out results/attn_flash.json \
  --extra="--flash-attention=true --sdp-attention=false $NOK"

"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag attn_sdpa \
  --warm-steps 5 --measure-steps 30 --mbs 8 --seq 1024 --timeout 900 --out results/attn_sdpa.json \
  --extra="--flash-attention=false --sdp-attention=true $NOK"

echo "==== cmp_attn done $(date) ===="
