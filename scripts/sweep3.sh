#!/usr/bin/env bash
# 16-bit LoRA comparison: Axolotl vs Unsloth (no 4-bit). 30-step warm+measure, mbs8/seq1024.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p results
UPY=/home/djl/.conda/envs/unsloth-bench/bin/python
echo "==== sweep3 (16-bit LoRA) start $(date) ===="

# Axolotl 16-bit LoRA + fused kernels
"$AXPY" scripts/bench.py --config configs/qwen06b_lora16.yml --tag axo_lora16 \
  --warm-steps 5 --measure-steps 30 --mbs 8 --seq 1024 --timeout 1200 \
  --out results/axo_lora16.json --extra="--logging-steps=50"
free -g | head -2

# Unsloth 16-bit LoRA (warm then measure)
"$UPY" scripts/bench_unsloth.py --max-steps 5  --grad-ckpt false --load-4bit false \
  --out results/_uwarm16.json --tag unsloth16 > results/unsloth16_warm.log 2>&1
"$UPY" scripts/bench_unsloth.py --max-steps 30 --grad-ckpt false --load-4bit false \
  --out results/unsloth16.json --tag unsloth16 > results/unsloth16_meas.log 2>&1
cat results/unsloth16.json
free -g | head -2

echo "==== sweep3 done $(date) ===="
