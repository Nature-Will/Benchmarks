#!/bin/bash
# DreamZero RoboTwin Fine-tuning Script (LoRA)
#
# Matches Pi0.5 training report settings where applicable:
#   - 500 trajectories from demo_randomized
#   - Effective batch size 64 (via gradient accumulation)
#   - 5K optimizer steps (LoRA converges faster than full fine-tuning)
#   - Save checkpoints every 1K steps
#
# Usage:
#   # 1. First convert data:
#   python scripts/convert_robotwin_to_gear.py --task beat_block_hammer --setting demo_randomized --output-dir ./data/robotwin_beat_block_hammer_rand500
#   python scripts/data/convert_lerobot_to_gear.py \
#       --dataset-path ./data/robotwin_beat_block_hammer_rand500 \
#       --embodiment-tag xdof \
#       --state-keys '{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}' \
#       --action-keys '{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}' \
#       --relative-action-keys left_joint_pos right_joint_pos \
#       --task-key annotation.task --force
#
#   # 2. Then train:
#   bash scripts/robotwin_training.sh

set -e

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
MODELS_DIR=${MODELS_DIR:-"/mnt/shared-storage-user/p1-shared/yujiale/models"}
ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT:-"./data/robotwin_beat_block_hammer_rand500"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_robotwin_lora"}
NUM_GPUS=${NUM_GPUS:-8}

# Base model paths
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"${MODELS_DIR}/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"${MODELS_DIR}/umt5-xxl"}

# Pre-trained DreamZero-AgiBot checkpoint (resume from)
PRETRAINED_CKPT=${PRETRAINED_CKPT:-"${MODELS_DIR}/DreamZero-AgiBot"}
# =============================================

# Validate paths
if [ ! -d "$ROBOTWIN_DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $ROBOTWIN_DATA_ROOT"
    echo "Run the data conversion script first."
    exit 1
fi

if [ ! -d "$WAN_CKPT_DIR" ]; then
    echo "ERROR: Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR"
    echo "Download: huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir $WAN_CKPT_DIR"
    exit 1
fi

if [ ! -d "$TOKENIZER_DIR" ]; then
    echo "ERROR: umt5-xxl tokenizer not found at $TOKENIZER_DIR"
    echo "Download: huggingface-cli download google/umt5-xxl --local-dir $TOKENIZER_DIR"
    exit 1
fi

echo "=== DreamZero RoboTwin LoRA Fine-tuning ==="
echo "Data: $ROBOTWIN_DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Resume from: $PRETRAINED_CKPT"
echo "============================================"

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/robotwin_relative \
    wandb_project=dreamzero_robotwin \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=5000 \
    weight_decay=1e-5 \
    save_total_limit=6 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    robotwin_data_root=$ROBOTWIN_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_CKPT
