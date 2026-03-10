#!/bin/bash

policy_name=LingBotVLA
task_name=${1:-beat_block_hammer}
task_config=${2:-demo_clean}
ckpt_setting=${3:-0}
seed=${4:-0}
gpu_id=${5:-0}
POLICY_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export CUDA_VISIBLE_DEVICES=${gpu_id}
export QWEN25_PATH=/mnt/shared-storage-user/p1-shared/yujiale/models/Qwen2.5-VL-3B-Instruct

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mtask: ${task_name}, config: ${task_config}, seed: ${seed}\033[0m"

cd "$POLICY_DIR/../.." # move to RoboTwin root

python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}
