#!/usr/bin/env bash
# Sweep #6: the user's real model size (Qwen3-1.7B QLoRA). axolotl-optimized vs unsloth.
# 20-step warm+measure, mbs4/seq1024, 4-bit.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p results
UPY=/home/djl/.conda/envs/unsloth-bench/bin/python
echo "==== sweep6 (Qwen3-1.7B) start $(date) ===="

"$AXPY" scripts/bench.py --config configs/qwen17b_optimized.yml --tag axo17b \
  --warm-steps 3 --measure-steps 20 --mbs 4 --seq 1024 --timeout 1400 --out results/axo17b.json \
  --extra="--logging-steps=50"
free -g | head -2

"$UPY" scripts/bench_unsloth.py --model Qwen/Qwen3-1.7B --max-steps 3  --mbs 4 --grad-ckpt false \
  --load-4bit true --out results/_uw17.json --tag unsloth17b > results/unsloth17_warm.log 2>&1
"$UPY" scripts/bench_unsloth.py --model Qwen/Qwen3-1.7B --max-steps 20 --mbs 4 --grad-ckpt false \
  --load-4bit true --out results/unsloth17b.json --tag unsloth17b > results/unsloth17_meas.log 2>&1
cat results/unsloth17b.json
free -g | head -2

echo "==== sweep6 done $(date) ===="
