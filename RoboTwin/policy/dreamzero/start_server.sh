#!/bin/bash
# Start DreamZero inference server for RoboTwin evaluation.
#
# This launches a distributed inference server on 2+ GPUs.
# Must be started BEFORE running eval.sh.
#
# Usage:
#   bash start_server.sh [model_path] [port] [num_gpus] [gpu_ids]
#
# Examples:
#   # Default: 2 GPUs on devices 0,1, port 5000
#   bash start_server.sh ./checkpoints/dreamzero_robotwin_50tasks_lora
#
#   # Custom: 4 GPUs, port 6000
#   bash start_server.sh ./checkpoints/dreamzero_robotwin_50tasks_lora 6000 4 0,1,2,3
#
#   # Use DreamZero-AgiBot pretrained (no fine-tuning, for testing)
#   bash start_server.sh /mnt/shared-storage-user/p1-shared/yujiale/models/DreamZero-AgiBot

set -e

MODEL_PATH=${1:-"./checkpoints/dreamzero_robotwin_50tasks_lora"}
PORT=${2:-5000}
NUM_GPUS=${3:-2}
GPU_IDS=${4:-"0,1"}

echo "=== DreamZero RoboTwin Inference Server ==="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPUs: $NUM_GPUS (devices: $GPU_IDS)"
echo "============================================"

export CUDA_VISIBLE_DEVICES=$GPU_IDS

torchrun --nproc_per_node=$NUM_GPUS --standalone \
    socket_test_robotwin.py \
    --port $PORT \
    --enable-dit-cache \
    --model-path $MODEL_PATH
