#!/bin/bash
# LingBot-VA Client on GPU node (runs RoboTwin simulation)
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_HOME=/usr/local/cuda
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot-va

cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVA/lingbot-va

save_root=${1:-'results/'}
task_name=${2:-'beat_block_hammer'}
test_num=${3:-100}

policy_name=LingBotVA
task_config=demo_randomized
PORT=29056

echo "=== Starting LingBot-VA Client: task=$task_name, test_num=$test_num ==="

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python -m evaluation.robotwin.eval_polict_client_openpi \
    --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name 0 \
    --model_name 0 \
    --ckpt_setting 0 \
    --seed 0 \
    --policy_name ${policy_name} \
    --save_root ${save_root} \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --test_num ${test_num} \
    --port ${PORT}
