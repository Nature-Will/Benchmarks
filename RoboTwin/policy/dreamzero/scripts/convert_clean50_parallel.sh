#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DREAMZERO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DREAMZERO_DIR"

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/p1-shared/yujiale/conda/envs/dreamzero/bin/python}"
ROBOTWIN_DATA_DIR="${ROBOTWIN_DATA_DIR:-/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/data}"
OUTPUT_BASE="${OUTPUT_BASE:-./data}"
LOG_DIR="${LOG_DIR:-$DREAMZERO_DIR/logs/clean50_parallel}"
JOBS="${JOBS:-$(nproc)}"

ALL_TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle
    shake_bottle_horizontally
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)

STATE_KEYS='{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}'
ACTION_KEYS='{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}'

mkdir -p "$LOG_DIR"

job_file="$(mktemp)"
trap 'rm -f "$job_file"' EXIT

for task in "${ALL_TASKS[@]}"; do
    src="$ROBOTWIN_DATA_DIR/$task/demo_clean/data"
    out="$OUTPUT_BASE/robotwin_${task}_clean50"
    printf "%s\t%s\t%s\n" "$task" "$src" "$out" >> "$job_file"
done

convert_one() {
    local task="$1"
    local src="$2"
    local out="$3"

    if [ -f "$out/meta/relative_stats_dreamzero.json" ]; then
        echo "[SKIP] ${task}_clean50"
        return 0
    fi

    if [ ! -d "$src" ] || ! ls "$src"/episode*.hdf5 1>/dev/null 2>&1; then
        echo "[MISS] ${task}_clean50 - no HDF5 data at $src"
        return 1
    fi

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

export PYTHON_BIN ROBOTWIN_DATA_DIR OUTPUT_BASE LOG_DIR STATE_KEYS ACTION_KEYS
export -f convert_one

echo "Clean50 jobs: $(wc -l < "$job_file")"
echo "Parallel jobs: $JOBS"
echo "Logs: $LOG_DIR"

parallel -j "$JOBS" --colsep '\t' convert_one {1} {2} {3} :::: "$job_file"
