#!/bin/bash
# DreamZero RoboTwin Evaluation Script
#
# Prerequisites:
#   1. Start the DreamZero inference server first (see start_server.sh)
#   2. The server must be running before starting evaluation
#
# Usage:
#   bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> [server_port]
#
# Example:
#   bash eval.sh beat_block_hammer demo_clean dreamzero_robotwin_lora 0 2 5000

policy_name=dreamzero
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
server_port=${6:-5000}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mserver port: ${server_port}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --server_port ${server_port}
