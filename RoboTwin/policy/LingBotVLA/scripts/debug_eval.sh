#!/bin/bash
# Debug evaluation: runs 3 episodes with full action logging + TOPP error messages
# Usage: bash scripts/debug_eval.sh [ckpt_step] [gpu_id]

set -e

policy_name=LingBotVLA
task_name=beat_block_hammer
task_config=demo_randomized
ckpt_step=${1:-5000}
gpu_id=${2:-0}

POLICY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="$POLICY_DIR/checkpoints/beat_block_hammer_randomized500/checkpoints/global_step_${ckpt_step}/hf_ckpt"
NORM_STATS="$POLICY_DIR/norm_stats/beat_block_hammer_randomized500.json"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_DIR"
    ls "$POLICY_DIR/checkpoints/beat_block_hammer_randomized500/checkpoints/" 2>/dev/null || echo "  (none)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export QWEN25_PATH=/mnt/shared-storage-user/p1-shared/yujiale/models/Qwen2.5-VL-3B-Instruct
export LINGBOT_DEBUG=1

echo "============================================="
echo "DEBUG EVAL: LingBot-VLA"
echo "  ckpt: ${ckpt_step}"
echo "  model: ${CKPT_DIR}"
echo "  norm_stats: ${NORM_STATS}"
echo "  LINGBOT_DEBUG=1 (action logging enabled)"
echo "  TOPP errors: uncommented"
echo "============================================="

mkdir -p "$POLICY_DIR/logs"

cd "$POLICY_DIR/../.."

python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting "debug_ckpt${ckpt_step}" \
    --seed 0 \
    --policy_name ${policy_name} \
    --model_path "${CKPT_DIR}" \
    --norm_stats_file "${NORM_STATS}" \
    2>&1 | tee "$POLICY_DIR/logs/debug_eval_ckpt${ckpt_step}.log"
