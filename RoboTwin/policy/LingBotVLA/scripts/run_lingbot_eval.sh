#!/bin/bash
set -o pipefail

# ============ GPU node environment init ============
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda/bin:/opt/conda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Vulkan ICD for SAPIEN rendering
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
if [ ! -f "$VK_ICD_FILENAMES" ]; then
    # Try to create nvidia_icd.json if missing
    mkdir -p /usr/share/vulkan/icd.d/
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null || true
fi
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Proxy
export http_proxy="http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128"
export https_proxy="$http_proxy"
export no_proxy="10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,127.0.0.1,localhost,.pjlab.org.cn,.h.pjlab.org.cn"

# Cache dirs
export XDG_CACHE_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache
export TMPDIR=/mnt/shared-storage-user/p1-shared/yujiale/.tmp
mkdir -p $TMPDIR

# Disable wandb
export WANDB_MODE=disabled

# Install ffmpeg if missing (needed for video recording)
if ! command -v ffmpeg &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1 || true
fi

echo "=== GPU Environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -2

# ============ Activate conda env ============
source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo "CuRobo: $(python -c 'from curobo.types.math import Pose; print("OK")' 2>&1)"

# ============ Run eval ============
echo ""
echo "=== Starting LingBotVLA Evaluation ==="
echo "Task: beat_block_hammer"
echo "Config: demo_clean"

cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVLA

export CUDA_VISIBLE_DEVICES=0
export QWEN25_PATH=/mnt/shared-storage-user/p1-shared/yujiale/models/Qwen2.5-VL-3B-Instruct
export PYTHONUNBUFFERED=1

mkdir -p /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVLA/logs

cd ../..
python -u -X faulthandler script/eval_policy.py --config policy/LingBotVLA/deploy_policy.yml \
    --overrides \
    --task_name beat_block_hammer \
    --task_config demo_clean \
    --ckpt_setting 0 \
    --seed 0 \
    --policy_name LingBotVLA \
    2>&1 | tee /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVLA/logs/lingbot_eval.log
PYTHON_EXIT=${PIPESTATUS[0]}

echo ""
echo "=== Python exit code: $PYTHON_EXIT ==="
echo "=== Evaluation Complete ==="
echo "Log saved to: /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVLA/logs/lingbot_eval.log"
