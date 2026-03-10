#!/bin/bash
# DreamZero 50-Task RoboTwin Training Script (LoRA)
#
# Based on YAM official settings (100K steps, bs=4 per GPU on H100).
# Adapted for H200 (143GB VRAM): bs=8 per GPU, 50K steps first.
#
# Key fixes from previous single-task run:
#   1. skip_component_loading=true  (proper pretrained weight loading)
#   2. defer_lora_injection=true    (correct LoRA init order)
#   3. bs=8 per GPU (vs 1)         (effective bs=64 on 8 GPUs)
#   4. 50K steps (vs 5K)           (new embodiment needs more training)
#   5. 50 tasks × 550 episodes     (vs 1 task × 500 episodes)
#
# Usage:
#   bash scripts/robotwin_all_training.sh
#
# If OOM with bs=8, set:
#   BATCH_SIZE=4 GRAD_ACCUM=2 bash scripts/robotwin_all_training.sh

set -e

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
MODELS_DIR=${MODELS_DIR:-"/mnt/shared-storage-user/p1-shared/yujiale/models"}
ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT:-"./data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_robotwin_50tasks_lora"}

# Training hyperparameters (override via env vars)
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_STEPS=${MAX_STEPS:-50000}
SAVE_STEPS=${SAVE_STEPS:-5000}
LR=${LR:-1e-5}

# GPU config
if [ -z "${NUM_GPUS}" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

# Base model paths
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"${MODELS_DIR}/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"${MODELS_DIR}/umt5-xxl"}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-"${MODELS_DIR}/DreamZero-AgiBot"}
# =============================================

# Validate paths
for dir_name in ROBOTWIN_DATA_ROOT WAN_CKPT_DIR TOKENIZER_DIR PRETRAINED_CKPT; do
    dir_val="${!dir_name}"
    if [ ! -d "$dir_val" ]; then
        echo "ERROR: $dir_name not found at $dir_val"
        exit 1
    fi
done

# Count datasets
NUM_DATASETS=$(ls -d "$ROBOTWIN_DATA_ROOT"/robotwin_* 2>/dev/null | wc -l)
if [ "$NUM_DATASETS" -eq 0 ]; then
    echo "ERROR: No robotwin_* datasets found in $ROBOTWIN_DATA_ROOT"
    echo "Run scripts/convert_all_tasks.sh first."
    exit 1
fi

EFFECTIVE_BS=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

echo "=== DreamZero 50-Task RoboTwin Training ==="
echo "Data root: $ROBOTWIN_DATA_ROOT"
echo "Datasets found: $NUM_DATASETS"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE/GPU × $GRAD_ACCUM accum × $NUM_GPUS GPUs = $EFFECTIVE_BS effective"
echo "Max steps: $MAX_STEPS"
echo "Save every: $SAVE_STEPS steps"
echo "Pretrained: $PRETRAINED_CKPT"
echo "skip_component_loading=true, defer_lora_injection=true"
echo "============================================"

# Build gradient accumulation arg
GRAD_ACCUM_ARG=""
if [ "$GRAD_ACCUM" -gt 1 ]; then
    GRAD_ACCUM_ARG="training_args.gradient_accumulation_steps=$GRAD_ACCUM"
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/robotwin_all_relative \
    wandb_project=dreamzero_robotwin_50tasks \
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
    training_args.learning_rate=$LR \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=11 \
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
    pretrained_model_path=$PRETRAINED_CKPT \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    $GRAD_ACCUM_ARG
