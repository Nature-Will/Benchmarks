#!/bin/bash
# Dim-I Instruction Probe — Pi0.5 PARALLEL (one instruction type per GPU)
set -e

export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/robotwin

cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin

POLICY_NAME="pi05"
TASK_NAME="beat_block_hammer"
TASK_CONFIG="demo_clean"
TRAIN_CONFIG="pi05_aloha_full_base"
MODEL_NAME="beat_block_hammer"
SEED=0
CHECKPOINT_ID=15000
TEST_NUM=3

# Remaining types to run (seen already done, unseen running on GPU 1)
TYPES=("abstract" "minimal" "do_nothing" "wrong_grasp" "partial" "wrong_action")
GPUS=(2 3 4 5 6 7)

LOGDIR=/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/pi05

echo "============================================"
echo "  Pi0.5 Parallel Instruction Probe"
echo "  Launching ${#TYPES[@]} types on GPUs ${GPUS[*]}"
echo "============================================"

for i in "${!TYPES[@]}"; do
    ITYPE=${TYPES[$i]}
    GPU=${GPUS[$i]}
    LOG=${LOGDIR}/instruction_probe_${ITYPE}.log

    echo "Launching ${ITYPE} on GPU ${GPU} -> ${LOG}"

    CUDA_VISIBLE_DEVICES=${GPU} \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
    nohup python script/eval_policy.py \
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
        --test_num ${TEST_NUM} \
        > ${LOG} 2>&1 &
done

echo ""
echo "All ${#TYPES[@]} jobs launched. Monitor with:"
echo "  grep -E 'Success|Fail|Done' ${LOGDIR}/instruction_probe_*.log"
echo ""
echo "Waiting for all jobs..."
wait
echo "ALL PARALLEL JOBS COMPLETE"

echo ""
echo "========== Results Summary =========="
for ITYPE in "${TYPES[@]}"; do
    echo "--- ${ITYPE} ---"
    grep -E 'Success rate.*3' ${LOGDIR}/instruction_probe_${ITYPE}.log 2>/dev/null || echo "(not complete)"
done
