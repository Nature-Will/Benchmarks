#!/bin/bash
# Dim-I Instruction Following Probe — Pi0.5 (15K checkpoint)
set -e

export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_HOME=/usr/local/cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/robotwin

cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin

POLICY_NAME="pi05"
TASK_NAME="beat_block_hammer"
TASK_CONFIG="demo_clean"
TRAIN_CONFIG="pi05_aloha_full_base"
MODEL_NAME="beat_block_hammer"
SEED=0
GPU_ID=${1:-0}
CHECKPOINT_ID=15000
TEST_NUM=3

export CUDA_VISIBLE_DEVICES=${GPU_ID}

INSTRUCTION_TYPES=("seen" "unseen" "abstract" "minimal" "do_nothing" "wrong_grasp" "partial" "wrong_action")

echo "============================================"
echo "  Dim-I Instruction Following Probe"
echo "  Model: Pi0.5 (ckpt ${CHECKPOINT_ID})"
echo "  Task: ${TASK_NAME}"
echo "  Config: ${TASK_CONFIG}"
echo "  Episodes per type: ${TEST_NUM}"
echo "  GPU: ${GPU_ID}"
echo "============================================"

for ITYPE in "${INSTRUCTION_TYPES[@]}"; do
    echo ""
    echo ">>> Running instruction_type=${ITYPE} (${TEST_NUM} episodes) ..."
    echo ""

    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py \
        --config policy/${POLICY_NAME}/deploy_policy.yml \
        --overrides \
        --task_name ${TASK_NAME} \
        --task_config ${TASK_CONFIG} \
        --train_config_name ${TRAIN_CONFIG} \
        --model_name ${MODEL_NAME} \
        --ckpt_setting "probe_${ITYPE}_ckpt${CHECKPOINT_ID}" \
        --seed ${SEED} \
        --policy_name ${POLICY_NAME} \
        --checkpoint_id ${CHECKPOINT_ID} \
        --instruction_type ${ITYPE} \
        --test_num ${TEST_NUM}

    echo ">>> Done: ${ITYPE}"
done

echo ""
echo "============================================"
echo "  All instruction types complete!"
echo "============================================"
