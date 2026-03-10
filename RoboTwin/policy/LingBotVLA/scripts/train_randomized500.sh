#!/bin/bash
set -e

# ============================================================
# LingBot-VLA Training Pipeline — 500 randomized episodes
# ============================================================
# Usage: bash train_randomized500.sh
#
# Steps 1-3 (data processing) run on CPU node.
# Step 4 (training) runs on GPU node via rjob.
# This script runs ALL steps; set SKIP_DATA=1 to skip data processing.

TASK_NAME=beat_block_hammer
NUM_EPISODES=500

POLICY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROBOTWIN_ROOT="$(cd "$POLICY_DIR/../.." && pwd)"
LINGBOT_REPO="$POLICY_DIR/lingbot-vla"

RAW_DATA_PATH="/mnt/shared-storage-user/p1-shared/yujiale/code/VLA-Benchs/data/original/robotwin/beat_block_hammer/aloha-agilex_randomized_500/data"
INTERMEDIATE_DIR="$POLICY_DIR/data/$TASK_NAME/randomized-${NUM_EPISODES}"
LEROBOT_DIR="$POLICY_DIR/lerobot_data/${TASK_NAME}_randomized500"
NORM_STATS="$POLICY_DIR/norm_stats/${TASK_NAME}_randomized500.json"
CONFIG_FILE="$POLICY_DIR/configs/${TASK_NAME}_randomized500.yaml"

export QWEN25_PATH=/mnt/shared-storage-user/p1-shared/yujiale/models/Qwen2.5-VL-3B-Instruct
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
mkdir -p "$POLICY_DIR/logs"

echo "============================================="
echo "LingBot-VLA Training — 500 Randomized Episodes"
echo "  Task:     $TASK_NAME"
echo "  Episodes: $NUM_EPISODES"
echo "  Raw data: $RAW_DATA_PATH"
echo "============================================="

if [ "${SKIP_DATA}" != "1" ]; then

# ----------------------------------------------------------
# Step 1: Convert RoboTwin HDF5 to intermediate format
# ----------------------------------------------------------
echo ""
echo "[Step 1/4] Converting RoboTwin HDF5 to intermediate format..."
cd "$POLICY_DIR"
python tools/process_data.py "$TASK_NAME" demo_randomized "$NUM_EPISODES" \
    --data_path "$RAW_DATA_PATH" \
    --save_path "$INTERMEDIATE_DIR"

# ----------------------------------------------------------
# Step 2: Compute normalization stats
# ----------------------------------------------------------
echo ""
echo "[Step 2/4] Computing normalization stats..."
mkdir -p "$POLICY_DIR/norm_stats"
python tools/compute_norm_stats.py \
    --data_dir "$INTERMEDIATE_DIR" \
    --output "$NORM_STATS"

# ----------------------------------------------------------
# Step 3: Convert to LeRobot format
# ----------------------------------------------------------
echo ""
echo "[Step 3/4] Converting to LeRobot dataset format..."
python tools/convert_to_lerobot.py \
    --data_dir "$INTERMEDIATE_DIR" \
    --output_dir "$LEROBOT_DIR" \
    --task_name "$TASK_NAME" \
    --fps 30

echo ""
echo "Data processing complete!"
echo "  Intermediate: $INTERMEDIATE_DIR"
echo "  LeRobot:      $LEROBOT_DIR"
echo "  Norm stats:   $NORM_STATS"

fi  # SKIP_DATA

# ----------------------------------------------------------
# Step 4: Launch distributed training
# ----------------------------------------------------------
echo ""
echo "[Step 4/4] Launching training..."

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Using $NPROC_PER_NODE GPUs"

cd "$LINGBOT_REPO"
torchrun \
    --nnodes=1 \
    --nproc-per-node=$NPROC_PER_NODE \
    --node-rank=0 \
    --master-addr=0.0.0.0 \
    --master-port=62500 \
    tasks/vla/train_lingbotvla.py \
    "$CONFIG_FILE" \
    2>&1 | tee "$POLICY_DIR/logs/train_${TASK_NAME}_randomized500.log"

echo ""
echo "============================================="
echo "Training complete!"
echo "Checkpoints at: $POLICY_DIR/checkpoints/${TASK_NAME}_randomized500"
echo "============================================="
