#!/bin/bash
# GPU launcher for LingBot-VLA debug evaluation
set -e

CONDA_DIR=/mnt/shared-storage-user/p1-shared/yujiale/conda
ROBOTWIN_ROOT=/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin

# Init conda
source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate lingbot

# Install ffmpeg if needed
which ffmpeg > /dev/null 2>&1 || apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1

cd ${ROBOTWIN_ROOT}/policy/LingBotVLA

echo "=== Starting debug eval ==="
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run debug eval with ckpt 5000 (first checkpoint)
bash scripts/debug_eval.sh 5000 0
