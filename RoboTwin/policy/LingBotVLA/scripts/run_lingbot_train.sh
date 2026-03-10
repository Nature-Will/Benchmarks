#!/bin/bash
set -o pipefail

# ============ GPU node environment init ============
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda/bin:/opt/conda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Cache dirs
export XDG_CACHE_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache
export TMPDIR=/mnt/shared-storage-user/p1-shared/yujiale/.tmp
mkdir -p $TMPDIR

# Disable wandb
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export QWEN25_PATH=/mnt/shared-storage-user/p1-shared/yujiale/models/Qwen2.5-VL-3B-Instruct
export HF_LEROBOT_HOME=/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVLA

echo "=== GPU Environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -8

# ============ Activate conda env ============
source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())')"

# ============ Run training ============
echo ""
echo "=== Starting LingBot-VLA Training ==="
echo "Task: beat_block_hammer (500 randomized episodes)"

POLICY_DIR=/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVLA
LINGBOT_REPO="$POLICY_DIR/lingbot-vla"
CONFIG_FILE="$POLICY_DIR/configs/beat_block_hammer_randomized500.yaml"
mkdir -p "$POLICY_DIR/logs"

NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "Using $NPROC_PER_NODE GPUs"
echo "Config: $CONFIG_FILE"

cd "$LINGBOT_REPO"
torchrun \
    --nnodes=1 \
    --nproc-per-node=$NPROC_PER_NODE \
    --node-rank=0 \
    --master-addr=0.0.0.0 \
    --master-port=62500 \
    tasks/vla/train_lingbotvla.py \
    "$CONFIG_FILE" \
    2>&1 | tee "$POLICY_DIR/logs/train_beat_block_hammer_randomized500.log"
TRAIN_EXIT=${PIPESTATUS[0]}

echo ""
echo "=== Training exit code: $TRAIN_EXIT ==="
echo "=== Training Complete ==="
echo "Log: $POLICY_DIR/logs/train_beat_block_hammer_randomized500.log"
echo "Checkpoints: $POLICY_DIR/checkpoints/beat_block_hammer_randomized500/"
