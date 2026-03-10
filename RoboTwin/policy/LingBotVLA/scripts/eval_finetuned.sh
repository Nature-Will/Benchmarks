#!/bin/bash

# Evaluate a fine-tuned LingBot-VLA checkpoint
# Usage: bash scripts/eval_finetuned.sh [task] [config] [ckpt_step] [seed] [gpu_id]
# Example: bash scripts/eval_finetuned.sh beat_block_hammer demo_randomized 15000 0 0

policy_name=LingBotVLA
task_name=${1:-beat_block_hammer}
task_config=${2:-demo_randomized}
ckpt_step=${3:-20000}
seed=${4:-0}
gpu_id=${5:-0}

POLICY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="$POLICY_DIR/checkpoints/beat_block_hammer_randomized500/checkpoints/global_step_${ckpt_step}/hf_ckpt"
NORM_STATS="$POLICY_DIR/norm_stats/beat_block_hammer_randomized500.json"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_DIR"
    echo "Available checkpoints:"
    ls "$POLICY_DIR/checkpoints/beat_block_hammer_randomized500/checkpoints/" 2>/dev/null || echo "  (none)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export QWEN25_PATH=/mnt/shared-storage-user/p1-shared/yujiale/models/Qwen2.5-VL-3B-Instruct

echo -e "\033[33mgpu id: ${gpu_id}\033[0m"
echo -e "\033[33mtask: ${task_name}, config: ${task_config}, ckpt: ${ckpt_step}, seed: ${seed}\033[0m"
echo -e "\033[33mmodel: ${CKPT_DIR}\033[0m"

cd ../.. # move to RoboTwin root

python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting "ckpt${ckpt_step}" \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --model_path "${CKPT_DIR}" \
    --norm_stats_file "${NORM_STATS}"
