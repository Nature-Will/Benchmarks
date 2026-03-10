#!/bin/bash
set -e

# ============================================================
# LingBot-VLA Training Pipeline for RoboTwin
# ============================================================
# Usage: bash train.sh [task_name] [setting] [num_episodes]
# Example: bash train.sh beat_block_hammer demo_clean 7

TASK_NAME=${1:-beat_block_hammer}
SETTING=${2:-demo_clean}
NUM_EPISODES=${3:-7}

POLICY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROBOTWIN_ROOT="$(cd "$POLICY_DIR/../.." && pwd)"
LINGBOT_REPO="$POLICY_DIR/lingbot-vla"

export QWEN25_PATH=/mnt/shared-storage-user/p1-shared/yujiale/models/Qwen2.5-VL-3B-Instruct
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
mkdir -p "$POLICY_DIR/logs"

echo "============================================="
echo "LingBot-VLA Training Pipeline"
echo "  Task:     $TASK_NAME"
echo "  Setting:  $SETTING"
echo "  Episodes: $NUM_EPISODES"
echo "============================================="

# ----------------------------------------------------------
# Step 1: Convert RoboTwin HDF5 to intermediate format
# ----------------------------------------------------------
echo ""
echo "[Step 1/4] Converting RoboTwin HDF5 to intermediate format..."
cd "$POLICY_DIR"
python tools/process_data.py "$TASK_NAME" "$SETTING" "$NUM_EPISODES"
INTERMEDIATE_DIR="$POLICY_DIR/data/$TASK_NAME/${SETTING}-${NUM_EPISODES}"

# ----------------------------------------------------------
# Step 2: Compute normalization stats
# ----------------------------------------------------------
echo ""
echo "[Step 2/4] Computing normalization stats..."
mkdir -p "$POLICY_DIR/norm_stats"
python tools/compute_norm_stats.py \
    --data_dir "$INTERMEDIATE_DIR" \
    --output "$POLICY_DIR/norm_stats/${TASK_NAME}.json"

# ----------------------------------------------------------
# Step 3: Convert to LeRobot format
# ----------------------------------------------------------
echo ""
echo "[Step 3/4] Converting to LeRobot dataset format..."
python tools/convert_to_lerobot.py \
    --data_dir "$INTERMEDIATE_DIR" \
    --output_dir "$POLICY_DIR/lerobot_data/$TASK_NAME" \
    --task_name "$TASK_NAME" \
    --fps 30

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

CONFIG_FILE="$POLICY_DIR/configs/${TASK_NAME}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

cd "$LINGBOT_REPO"
torchrun \
    --nnodes=1 \
    --nproc-per-node=$NPROC_PER_NODE \
    --node-rank=0 \
    --master-addr=0.0.0.0 \
    --master-port=62500 \
    tasks/vla/train_lingbotvla.py \
    "$CONFIG_FILE" \
    2>&1 | tee "$POLICY_DIR/logs/train_${TASK_NAME}.log"

echo ""
echo "============================================="
echo "Training complete!"
echo "Checkpoints at: $POLICY_DIR/checkpoints/$TASK_NAME"
echo "============================================="
