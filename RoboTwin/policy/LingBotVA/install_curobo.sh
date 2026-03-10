#!/bin/bash
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_HOME=/usr/local/cuda

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot-va

CUROBO_DIR=/mnt/shared-storage-user/p1-shared/yujiale/code/VLA-Benchs/third_party/benchmarks/RoboTwin/envs/curobo

echo "=== Installing curobo from source ==="
cd "$CUROBO_DIR"
pip install -e . --no-build-isolation 2>&1 | tail -10
echo "=== Verifying ==="
python -c "from curobo.types.math import Pose; print('curobo OK')"
echo "=== Done ==="
