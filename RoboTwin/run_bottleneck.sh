#!/bin/bash
set -e
source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate robotwin
cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda,cpu
export OPENPI_DATA_HOME=/mnt/shared-storage-user/p1-shared/yujiale/code/VLA-Benchs/.cache_runtime/openpi
export PYTHONUNBUFFERED=1

echo "=== Bottleneck Analysis ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python -u policy/pi05/verify_grounding_bottleneck.py 2>&1 | tee reports/attention_analysis/bottleneck_run.log

echo "=== Done: $(date) ==="
ls -la reports/attention_analysis/bottleneck*
