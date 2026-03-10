#!/bin/bash
# Dim-I Instruction Following Probe — LingBot-VA
# Requires: LingBot-VA server already running on this node (start_server.sh)
set -e

export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_HOME=/usr/local/cuda
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot-va

cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVA/lingbot-va

SAVE_ROOT="results_instruction_probe/"
TASK_NAME="beat_block_hammer"
TEST_NUM=3
POLICY_NAME="LingBotVA"
TASK_CONFIG="demo_clean"
PORT=29056

INSTRUCTION_TYPES=("seen" "unseen" "abstract" "minimal" "do_nothing" "wrong_grasp" "partial" "wrong_action")

echo "============================================"
echo "  Dim-I Instruction Following Probe"
echo "  Model: LingBot-VA (posttrain-robotwin)"
echo "  Task: ${TASK_NAME}"
echo "  Config: ${TASK_CONFIG}"
echo "  Episodes per type: ${TEST_NUM}"
echo "============================================"

for ITYPE in "${INSTRUCTION_TYPES[@]}"; do
    echo ""
    echo ">>> Running instruction_type=${ITYPE} (${TEST_NUM} episodes) ..."
    echo ""

    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python -m evaluation.robotwin.eval_polict_client_openpi \
        --config policy/${POLICY_NAME}/deploy_policy.yml \
        --overrides \
        --task_name ${TASK_NAME} \
        --task_config ${TASK_CONFIG} \
        --train_config_name 0 \
        --model_name 0 \
        --ckpt_setting "probe_${ITYPE}" \
        --seed 0 \
        --policy_name ${POLICY_NAME} \
        --save_root ${SAVE_ROOT} \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --test_num ${TEST_NUM} \
        --port ${PORT} \
        --instruction_type ${ITYPE}

    echo ">>> Done: ${ITYPE}"
done

echo ""
echo "============================================"
echo "  All instruction types complete!"
echo "============================================"
