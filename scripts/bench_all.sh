#!/usr/bin/env bash
# Run the Axolotl benchmark suite sequentially. Memory-safe shape (seq1024/mbs8).
# Launch DETACHED with: setsid timeout -s KILL <N> bash scripts/bench_all.sh
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh

WARM=${WARM:-5}
MEAS=${MEAS:-30}
mkdir -p results

echo "==== bench_all start $(date) | warm=$WARM measure=$MEAS ===="
free -g | head -2

for SPEC in "vanilla:configs/qwen06b_vanilla.yml" "optimized:configs/qwen06b_optimized.yml"; do
  TAG="${SPEC%%:*}"; CFG="${SPEC##*:}"
  echo "---- running $TAG ($CFG) ----"
  "$AXPY" scripts/bench.py --config "$CFG" --tag "$TAG" \
      --warm-steps "$WARM" --measure-steps "$MEAS" --out "results/${TAG}.json" --timeout 1400
done

echo "==== bench_all done $(date) ===="
