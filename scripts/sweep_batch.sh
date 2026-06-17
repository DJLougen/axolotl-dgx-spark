#!/usr/bin/env bash
# Batch scaling under flash: leverage Axolotl's lower memory (FLCE) to run larger batches than
# Unsloth can, chasing its throughput ceiling. Qwen3-0.6B 4-bit seq1024. Memory-logged between runs.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
FL="--flash-attention=true --sdp-attention=false --logging-steps=50"
echo "==== sweep_batch start $(date) ===="
for MBS in 32 48; do
  echo "---- axolotl opt+flash mbs$MBS ----"; free -g | awk 'NR==2{print "free GiB before:",$4}'
  "$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag opt_flash_mbs$MBS \
    --warm-steps 4 --measure-steps 24 --mbs $MBS --seq 1024 --timeout 1400 \
    --out results/opt_flash_mbs$MBS.json --extra="--micro-batch-size=$MBS $FL"
  free -g | awk 'NR==2{print "free GiB after:",$4}'
done
echo "==== sweep_batch done $(date) ===="
