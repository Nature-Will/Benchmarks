#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda/bin:/opt/conda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TMPDIR=/mnt/shared-storage-user/p1-shared/yujiale/.tmp
mkdir -p $TMPDIR

# CuRobo CUDA kernel compilation settings for H200
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=8

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot

echo "Python: $(which python)"
echo "CUDA: $(nvcc --version | tail -1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -2

echo "=== Installing curobo from source ==="
cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/envs/curobo
pip install -e . --no-build-isolation 2>&1 | tail -40

echo "=== Verifying curobo ==="
python -c "from curobo.types.math import Pose; print('curobo import OK')"

echo "=== Done ==="
