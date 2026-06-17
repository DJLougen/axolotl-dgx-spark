#!/usr/bin/env bash
# Sweep #4: is the gap fixed-overhead bound? Test dataloader workers + larger batch.
# 4-bit QLoRA (matches user's workflow). 30-step warm+measure, seq1024.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p results
UPY=/home/djl/.conda/envs/unsloth-bench/bin/python
echo "==== sweep4 start $(date) ===="

# 1) mbs8 + dataloader workers/pin (isolate data-pipeline stalls)
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag axo_dl4 \
  --warm-steps 5 --measure-steps 30 --mbs 8 --seq 1024 --timeout 1200 --out results/axo_dl4.json \
  --extra="--dataloader-num-workers=4 --dataloader-pin-memory=true --logging-steps=50"
free -g | head -2

# 2) axolotl mbs32 (amortize fixed per-step overhead)
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag axo_mbs32 \
  --warm-steps 5 --measure-steps 30 --mbs 32 --seq 1024 --timeout 1400 --out results/axo_mbs32.json \
  --extra="--micro-batch-size=32 --dataloader-num-workers=4 --logging-steps=50"
free -g | head -2

# 3) unsloth mbs32 (warm + measure)
"$UPY" scripts/bench_unsloth.py --max-steps 5  --mbs 32 --grad-ckpt false --load-4bit true \
  --out results/_uwarm32.json --tag unsloth_mbs32 > results/unsloth_mbs32_warm.log 2>&1
"$UPY" scripts/bench_unsloth.py --max-steps 30 --mbs 32 --grad-ckpt false --load-4bit true \
  --out results/unsloth_mbs32.json --tag unsloth_mbs32 > results/unsloth_mbs32_meas.log 2>&1
cat results/unsloth_mbs32.json
free -g | head -2

echo "==== sweep4 done $(date) ===="
