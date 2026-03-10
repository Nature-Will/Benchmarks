#!/bin/bash
# Run Pi0.5 attention visualization on GPU node
set -e

# Init conda
source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate robotwin

cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin

echo "=== Pi0.5 Attention Visualization ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Python: $(python --version)"
echo "Start: $(date)"

# Use single GPU
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda,cpu

# Point openpi cache to shared storage (GPU node has no network)
export OPENPI_DATA_HOME=/mnt/shared-storage-user/p1-shared/yujiale/code/VLA-Benchs/.cache_runtime/openpi

# Run the visualization script (unbuffered output)
export PYTHONUNBUFFERED=1
python -u policy/pi05/visualize_attention.py 2>&1 | tee reports/attention_analysis/run.log

echo ""
echo "=== Done: $(date) ==="
echo "Results in: reports/attention_analysis/"
ls -la reports/attention_analysis/
