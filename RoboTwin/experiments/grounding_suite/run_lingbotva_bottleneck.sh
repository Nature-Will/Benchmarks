#!/bin/bash
set -e

EXP_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${EXP_DIR}/../.." && pwd)
OUT_DIR="${EXP_DIR}/outputs/lingbotva_bottleneck"

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot-va
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export JAX_PLATFORMS=cuda,cpu
export PYTHONUNBUFFERED=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29661}
export RANK=${RANK:-0}
export LOCAL_RANK=${LOCAL_RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

mkdir -p "${OUT_DIR}"

echo "=== LingBot-VA Bottleneck Analysis ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python -u "${EXP_DIR}/verify_lingbotva_bottleneck.py" 2>&1 | tee "${OUT_DIR}/run.log"

echo "=== Done: $(date) ==="
ls -la "${OUT_DIR}"
