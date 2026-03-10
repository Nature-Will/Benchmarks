#!/bin/bash
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_HOME=/usr/local/cuda

export http_proxy="http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128"
export https_proxy="$http_proxy"

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot-va

echo "=== Installing flash-attn from PyPI (not mirror) ==="
pip install flash-attn --no-build-isolation --index-url https://pypi.org/simple/ 2>&1 | tail -10
echo "=== Verifying ==="
python -c "import flash_attn; print(f'flash_attn={flash_attn.__version__}')"
echo "=== Done ==="
