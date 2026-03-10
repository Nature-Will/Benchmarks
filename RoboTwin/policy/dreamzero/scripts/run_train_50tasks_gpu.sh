#!/bin/bash
# DreamZero 50-Task Training Launcher (GPU node)
#
# Run on GPU node (e.g. worker-9f8hv, 8×H200):
#   bash scripts/run_train_50tasks_gpu.sh
#
# If OOM with bs=8, retry with:
#   BATCH_SIZE=4 GRAD_ACCUM=2 bash scripts/run_train_50tasks_gpu.sh

# ============ GPU Node Environment Init ============
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda/bin:/opt/conda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export XDG_CACHE_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache
export TMPDIR=/mnt/shared-storage-user/p1-shared/yujiale/.tmp
export HF_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache/huggingface
mkdir -p $TMPDIR $HF_HOME

# No proxy needed on GPU node (all data already downloaded)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO

# ============ Conda ============
source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/dreamzero

# ============ Config ============
DREAMZERO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR=$DREAMZERO_DIR/logs
mkdir -p $LOG_DIR

cd $DREAMZERO_DIR

# Pass through env vars for training script
export BATCH_SIZE=${BATCH_SIZE:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-1}
export MAX_STEPS=${MAX_STEPS:-50000}
export SAVE_STEPS=${SAVE_STEPS:-5000}

EFFECTIVE_BS=$((BATCH_SIZE * GRAD_ACCUM * 8))

echo "============================================"
echo "DreamZero 50-Task Training"
echo "============================================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Batch size: ${BATCH_SIZE}/GPU × ${GRAD_ACCUM} accum × 8 GPUs = ${EFFECTIVE_BS} effective"
echo "Max steps: $MAX_STEPS"
echo "Log: $LOG_DIR/train_50tasks.log"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | head -8
echo "============================================"

# Count datasets
NUM_DATASETS=$(ls -d data/robotwin_* 2>/dev/null | wc -l)
echo "GEAR datasets found: $NUM_DATASETS"

if [ "$NUM_DATASETS" -eq 0 ]; then
    echo "ERROR: No data/robotwin_* datasets found"
    echo "Run scripts/convert_all_tasks.sh on CPU node first"
    exit 1
fi

echo ""
echo "Starting training..."
echo ""

bash scripts/robotwin_all_training.sh 2>&1 | tee $LOG_DIR/train_50tasks.log

echo ""
echo "============================================"
echo "Training finished at $(date)"
echo "Log: $LOG_DIR/train_50tasks.log"
echo "Checkpoints: $DREAMZERO_DIR/checkpoints/dreamzero_robotwin_50tasks_lora/"
echo "============================================"
