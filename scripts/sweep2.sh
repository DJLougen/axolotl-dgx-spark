#!/usr/bin/env bash
# Diagnostic sweep #2 (fixed --extra= form). 30-step warm+measure, mbs8/seq1024.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p results
W=5; M=30; COMMON="--warm-steps $W --measure-steps $M --mbs 8 --seq 1024 --timeout 1200"
echo "==== sweep2 start $(date) ===="

# A) isolate per-step instrumentation overhead (axolotl logs ppl+mem+tokens each step)
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag opt_log50 $COMMON \
  --out results/opt_log50.json --extra="--logging-steps=50"
free -g | head -2

# B) liger only, NO lora kernels -- are the LoRA Triton kernels helping at 0.6B?
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag liger_only $COMMON \
  --out results/liger_only.json \
  --extra="--lora-mlp-kernel=false --lora-qkv-kernel=false --lora-o-kernel=false --logging-steps=50"
free -g | head -2

# C) optimized + sample packing (removes padding waste -> real-token throughput)
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized_packed.yml --tag opt_packed $COMMON \
  --out results/opt_packed.json --extra="--logging-steps=50"
free -g | head -2

echo "==== sweep2 done $(date) ===="
