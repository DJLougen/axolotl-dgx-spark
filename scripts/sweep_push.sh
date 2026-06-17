#!/usr/bin/env bash
# Push sweep: combine flash with the levers not yet tested under flash. Qwen3-0.6B, seq1024.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
UPY=/home/djl/.conda/envs/unsloth-bench/bin/python
FL="--flash-attention=true --sdp-attention=false --logging-steps=50"
echo "==== sweep_push start $(date) ===="

# 1) 16-bit LoRA + flash + Liger + LoRA kernels (no 4-bit dequant overhead), mbs8
"$AXPY" scripts/bench.py --config configs/qwen06b_lora16.yml --tag opt16_flash \
  --warm-steps 5 --measure-steps 30 --mbs 8 --seq 1024 --timeout 900 --out results/opt16_flash.json --extra="$FL"
free -g | head -2

# 2) 4-bit + flash, mbs16 (batch scaling under flash)
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag opt_flash_mbs16 \
  --warm-steps 5 --measure-steps 30 --mbs 16 --seq 1024 --timeout 1100 --out results/opt_flash_mbs16.json \
  --extra="--micro-batch-size=16 $FL"
free -g | head -2

# 3) 4-bit + flash + fused torch optimizer (vs bnb 8bit)
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag opt_flash_adamwfused \
  --warm-steps 5 --measure-steps 30 --mbs 8 --seq 1024 --timeout 900 --out results/opt_flash_adamwfused.json \
  --extra="--optimizer=adamw_torch_fused $FL"
free -g | head -2

# 4) unsloth mbs16 reference (warm+measure)
"$UPY" scripts/bench_unsloth.py --max-steps 5  --mbs 16 --grad-ckpt false --load-4bit true \
  --out results/_uw16.json --tag unsloth_mbs16 > results/unsloth_mbs16_warm.log 2>&1
"$UPY" scripts/bench_unsloth.py --max-steps 30 --mbs 16 --grad-ckpt false --load-4bit true \
  --out results/unsloth_mbs16.json --tag unsloth_mbs16 > results/unsloth_mbs16_meas.log 2>&1
cat results/unsloth_mbs16.json

echo "==== sweep_push done $(date) ===="
