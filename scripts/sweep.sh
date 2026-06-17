#!/usr/bin/env bash
# Sweep Axolotl-optimized variants to close the gap to Unsloth. All 30-step warm+measure.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p results
WARM=5; MEAS=30
echo "==== sweep start $(date) ===="

# 1) isolate per-step logging/instrumentation overhead (logging_steps 1 -> 50)
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag opt_log50 \
  --warm-steps $WARM --measure-steps $MEAS --out results/opt_log50.json --timeout 1200 \
  --extra "--logging-steps=50"
free -g | head -2

# 2) + torch.compile (docs: net win on Blackwell sm_120; GB10 is sm_121)
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized.yml --tag opt_compile \
  --warm-steps $WARM --measure-steps $MEAS --out results/opt_compile.json --timeout 1600 \
  --extra "--torch-compile=true --logging-steps=50"
free -g | head -2

# 3) + sample packing (removes padding waste; real-token throughput)
"$AXPY" scripts/bench.py --config configs/qwen06b_optimized_packed.yml --tag opt_packed \
  --warm-steps $WARM --measure-steps $MEAS --out results/opt_packed.json --timeout 1200 \
  --extra "--logging-steps=50"
free -g | head -2

echo "==== sweep done $(date) ===="
