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
    mkdir -p /usr/share/vulkan/icd.d/
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null || true
fi
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Cache dirs
export XDG_CACHE_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache
export TMPDIR=/mnt/shared-storage-user/p1-shared/yujiale/.tmp
mkdir -p $TMPDIR

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# Install ffmpeg if missing
if ! command -v ffmpeg &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1 || true
fi

echo "=== GPU Environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -8

# ============ Activate conda env ============
source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo "CuRobo: $(python -c 'from curobo.types.math import Pose; print("OK")' 2>&1)"

# ============ Evaluate all checkpoints in parallel ============
POLICY_DIR=/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVLA
LOG_DIR=$POLICY_DIR/logs/eval_all_ckpts
mkdir -p "$LOG_DIR"

TASK=beat_block_hammer
CONFIG=demo_randomized

echo ""
echo "=== Evaluating LingBot-VLA fine-tuned checkpoints ==="
echo "Task: $TASK, Config: $CONFIG"

# Evaluate 4 checkpoints on 4 GPUs in parallel
for i in 0 1 2 3; do
    CKPT_STEP=$(( (i + 1) * 5000 ))  # 5000, 10000, 15000, 20000
    echo "Starting eval: ckpt_${CKPT_STEP} on GPU $i"
    cd "$POLICY_DIR"
    bash scripts/eval_finetuned.sh "$TASK" "$CONFIG" "$CKPT_STEP" 0 "$i" \
        > "$LOG_DIR/ckpt_${CKPT_STEP}.log" 2>&1 &
done

echo "All 4 evals launched. Waiting..."
wait

echo ""
echo "=== All evaluations complete ==="
echo "Logs at: $LOG_DIR/"
for i in 0 1 2 3; do
    CKPT_STEP=$(( (i + 1) * 5000 ))
    echo "--- ckpt_${CKPT_STEP} ---"
    tail -5 "$LOG_DIR/ckpt_${CKPT_STEP}.log"
done
