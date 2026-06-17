#!/usr/bin/env bash
# Build a separate env for the Unsloth reference benchmark (Unsloth needs transformers 4.x,
# which conflicts with Axolotl 0.17's transformers 5.x). Same torch 2.9.1+cu130 for fairness.
set -u
source /opt/conda/etc/profile.d/conda.sh
echo "==== build unsloth-bench $(date) ===="
conda create -y -n unsloth-bench python=3.11 2>&1 | tail -2
PY=/home/djl/.conda/envs/unsloth-bench/bin/python
echo "--- torch cu130 ---"
$PY -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -3
echo "--- unsloth ---"
$PY -m pip install --no-cache-dir unsloth unsloth_zoo 2>&1 | tail -5
echo "--- ensure cu130 torch retained ---"
$PY -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2
echo "--- versions ---"
$PY -c "import torch,transformers,trl; print('torch',torch.__version__,'avail',torch.cuda.is_available()); print('transformers',transformers.__version__,'trl',trl.__version__)" 2>&1 | tail -4
echo "==== build done $(date) ===="
