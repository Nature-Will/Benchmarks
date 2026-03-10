#!/bin/bash
# Full evaluation of all LingBot-VLA fine-tuned checkpoints (100 episodes each)
# NO set -e: continue evaluating remaining checkpoints even if one fails

CONDA_DIR=/mnt/shared-storage-user/p1-shared/yujiale/conda
ROBOTWIN_ROOT=/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin
POLICY_DIR=${ROBOTWIN_ROOT}/policy/LingBotVLA
LOG_DIR=${POLICY_DIR}/logs/eval_all

source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate lingbot

which ffmpeg > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1)

mkdir -p ${LOG_DIR}

export QWEN25_PATH=/mnt/shared-storage-user/p1-shared/yujiale/models/Qwen2.5-VL-3B-Instruct
export CUDA_VISIBLE_DEVICES=0

echo "=== LingBot-VLA Full Evaluation (with TOPP fallback fix) ==="
echo "Start time: $(date)"

for CKPT_STEP in 5000 10000 15000 20000; do
    CKPT_DIR="${POLICY_DIR}/checkpoints/beat_block_hammer_randomized500/checkpoints/global_step_${CKPT_STEP}/hf_ckpt"
    NORM_STATS="${POLICY_DIR}/norm_stats/beat_block_hammer_randomized500.json"

    if [ ! -d "${CKPT_DIR}" ]; then
        echo "SKIP: checkpoint ${CKPT_STEP} not found"
        continue
    fi

    # Skip if already completed (log has final success rate)
    EXISTING_LOG="${LOG_DIR}/ckpt_${CKPT_STEP}.log"
    if [ -f "${EXISTING_LOG}" ] && grep -q "Success rate.*100" "${EXISTING_LOG}" 2>/dev/null; then
        echo "SKIP: checkpoint ${CKPT_STEP} already evaluated (found 100-episode result in log)"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating checkpoint: ${CKPT_STEP}"
    echo "Time: $(date)"
    echo "=========================================="

    cd ${ROBOTWIN_ROOT}

    python script/eval_policy.py --config policy/LingBotVLA/deploy_policy.yml \
        --overrides \
        --task_name beat_block_hammer \
        --task_config demo_randomized \
        --ckpt_setting "ckpt${CKPT_STEP}" \
        --seed 0 \
        --policy_name LingBotVLA \
        --model_path "${CKPT_DIR}" \
        --norm_stats_file "${NORM_STATS}" \
        2>&1 | tee "${LOG_DIR}/ckpt_${CKPT_STEP}.log"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "WARNING: checkpoint ${CKPT_STEP} eval exited with code ${EXIT_CODE}"
    fi

    echo ""
    echo "Checkpoint ${CKPT_STEP} evaluation complete (exit code: ${EXIT_CODE})."
    echo ""
done

echo ""
echo "=== All evaluations complete ==="
echo "End time: $(date)"
echo "Logs at: ${LOG_DIR}"

# Print summary
echo ""
echo "=== SUMMARY ==="
for CKPT_STEP in 5000 10000 15000 20000; do
    LOG="${LOG_DIR}/ckpt_${CKPT_STEP}.log"
    if [ -f "${LOG}" ]; then
        RESULT=$(grep "Success rate" "${LOG}" | tail -1)
        echo "ckpt ${CKPT_STEP}: ${RESULT}"
    fi
done
