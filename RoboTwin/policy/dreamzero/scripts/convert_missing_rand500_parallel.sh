#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DREAMZERO_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/p1-shared/yujiale/conda/envs/dreamzero/bin/python}"
VLA_BENCHS_PATH="${VLA_BENCHS_PATH:-/mnt/shared-storage-user/p1-shared/yujiale/code/VLA-Benchs/data/original/robotwin}"
LOG_DIR="${LOG_DIR:-$DREAMZERO_DIR/logs/rand500_parallel}"
JOBS="${JOBS:-$(nproc)}"

STATE_KEYS='{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}'
ACTION_KEYS='{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}'

mkdir -p "$LOG_DIR"
cd "$DREAMZERO_DIR"

job_file="$(mktemp)"
trap 'rm -f "$job_file"' EXIT

for d in ./data/robotwin_*_rand500; do
    if [ ! -f "$d/meta/episodes.jsonl" ]; then
        task="$(basename "$d" | sed 's/^robotwin_//; s/_rand500$//')"
        src="$VLA_BENCHS_PATH/$task/aloha-agilex_randomized_500/data"
        printf "%s\t%s\t%s\n" "$task" "$src" "$d" >> "$job_file"
    fi
done

job_count="$(wc -l < "$job_file")"
echo "Missing rand500 tasks: $job_count"
echo "Parallel jobs: $JOBS"
echo "Logs: $LOG_DIR"

if [ "$job_count" -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

convert_one() {
    local task="$1"
    local src="$2"
    local out="$3"

    "$PYTHON_BIN" scripts/convert_robotwin_to_gear.py \
        --task "$task" \
        --data-dir "$src" \
        --output-dir "$out" \
        > "$LOG_DIR/$task.convert.log" 2>&1

    "$PYTHON_BIN" scripts/data/convert_lerobot_to_gear.py \
        --dataset-path "$out" \
        --embodiment-tag xdof \
        --state-keys "$STATE_KEYS" \
        --action-keys "$ACTION_KEYS" \
        --relative-action-keys left_joint_pos right_joint_pos \
        --task-key annotation.task \
        --force \
        > "$LOG_DIR/$task.gear.log" 2>&1
}

export PYTHON_BIN LOG_DIR STATE_KEYS ACTION_KEYS
export -f convert_one

parallel -j "$JOBS" --colsep '\t' convert_one {1} {2} {3} :::: "$job_file"
