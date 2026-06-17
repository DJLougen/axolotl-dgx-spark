#!/usr/bin/env bash
# Shared environment for all Axolotl runs on the GB10 Spark.
source /opt/conda/etc/profile.d/conda.sh
conda activate axfast
export PATH="$CONDA_PREFIX/bin:$PATH"     # env bins first (avoid ~/.local CPU-torch axolotl)
export PYTHONNOUSERSITE=1                  # ignore ~/.local/lib (py3.13 base, CPU torch)
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas   # CUDA13 ptxas: supports sm_121a (GB10). MANDATORY.
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=0
export AXPY=/home/djl/.conda/envs/axfast/bin/python   # explicit interpreter (never rely on PATH)
