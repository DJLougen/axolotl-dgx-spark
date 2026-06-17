#!/usr/bin/env bash
# Confirm batch scaling on the user's real model (Qwen3-1.7B). flash, 4-bit, seq1024.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
UPY=/home/djl/.conda/envs/unsloth-bench/bin/python
FL="--flash-attention=true --sdp-attention=false --logging-steps=50"
echo "==== sweep_17b start $(date) ===="
for MBS in 8 16; do
  echo "---- axolotl 1.7B flash mbs$MBS ----"; free -g | awk 'NR==2{print "free:",$4}'
  "$AXPY" scripts/bench.py --config configs/qwen17b_optimized.yml --tag axo17b_flash_mbs$MBS \
    --warm-steps 3 --measure-steps 20 --mbs $MBS --seq 1024 --timeout 1200 \
    --out results/axo17b_flash_mbs$MBS.json --extra="--micro-batch-size=$MBS $FL"
  "$UPY" scripts/bench_unsloth.py --model Qwen/Qwen3-1.7B --max-steps 3  --mbs $MBS --grad-ckpt false \
    --load-4bit true --out results/_uw17_$MBS.json --tag u17_$MBS > results/u17_${MBS}_w.log 2>&1
  "$UPY" scripts/bench_unsloth.py --model Qwen/Qwen3-1.7B --max-steps 20 --mbs $MBS --grad-ckpt false \
    --load-4bit true --out results/unsloth17b_mbs$MBS.json --tag u17_$MBS > results/u17_${MBS}_m.log 2>&1
  free -g | awk 'NR==2{print "free after:",$4}'
done
echo "==== sweep_17b done $(date) ===="
