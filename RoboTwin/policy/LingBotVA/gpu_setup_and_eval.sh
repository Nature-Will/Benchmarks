#!/bin/bash
set -e

# ============ GPU Node Environment Init ============
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Proxy for pip
export http_proxy="http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128"
export https_proxy="$http_proxy"

# Conda
source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot-va

echo "=== GPU Check ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

echo "=== Python/Torch Check ==="
python -c "import torch; print(f'torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}')"

echo "=== Install flash-attn ==="
pip install flash-attn --no-build-isolation 2>&1 | tail -5

echo "=== Setup Complete ==="
