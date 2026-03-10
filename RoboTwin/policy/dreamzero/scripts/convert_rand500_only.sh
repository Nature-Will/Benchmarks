#!/bin/bash
# Convert only demo_randomized (500 eps) for all 50 tasks to GEAR format
# Run this while demo_clean is downloading
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DREAMZERO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DREAMZERO_DIR"

VLA_BENCHS_PATH=${VLA_BENCHS_PATH:-"/mnt/shared-storage-user/p1-shared/yujiale/code/VLA-Benchs/data/original/robotwin"}
OUTPUT_BASE="./data"

STATE_KEYS='{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}'
ACTION_KEYS='{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}'

ALL_TASKS=(
    adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
    click_alarmclock click_bell dump_bin_bigbin grab_roller
    handover_block handover_mic hanging_mug lift_pot
    move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad
    open_laptop open_microwave pick_diverse_bottles pick_dual_bottles
    place_a2b_left place_a2b_right place_bread_basket place_bread_skillet
    place_burger_fries place_can_basket place_cans_plasticbox place_container_plate
    place_dual_shoes place_empty_cup place_fan place_mouse_pad
    place_object_basket place_object_scale place_object_stand place_phone_stand
    place_shoe press_stapler put_bottles_dustbin put_object_cabinet
    rotate_qrcode scan_object shake_bottle shake_bottle_horizontally
    stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two
    stamp_seal turn_switch
)

echo "=== Convert demo_randomized (500 eps) for ${#ALL_TASKS[@]} tasks ==="
echo ""

DONE=0
SKIP=0
FAIL=0

for task in "${ALL_TASKS[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/robotwin_${task}_rand500"

    # Skip if already fully converted
    if [ -f "$OUTPUT_DIR/meta/relative_stats_dreamzero.json" ]; then
        echo "[SKIP] $task"
        SKIP=$((SKIP + 1))
        continue
    fi

    DATA_DIR="$VLA_BENCHS_PATH/$task/aloha-agilex_randomized_500/data"
    if [ ! -d "$DATA_DIR" ]; then
        echo "[MISS] $task - no data"
        FAIL=$((FAIL + 1))
        continue
    fi

    echo "[CONV] $task (rand500) ..."

    # Step 1: HDF5 → LeRobot
    if [ ! -f "$OUTPUT_DIR/meta/info.json" ]; then
        python scripts/convert_robotwin_to_gear.py \
            --task "$task" \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" 2>&1 | tail -2
    fi

    # Step 2: GEAR metadata
    python scripts/data/convert_lerobot_to_gear.py \
        --dataset-path "$OUTPUT_DIR" \
        --embodiment-tag xdof \
        --state-keys "$STATE_KEYS" \
        --action-keys "$ACTION_KEYS" \
        --relative-action-keys left_joint_pos right_joint_pos \
        --task-key annotation.task \
        --force 2>&1 | tail -2

    DONE=$((DONE + 1))
    echo "[DONE] $task ($DONE done)"
done

echo ""
echo "=== Summary ==="
echo "Converted: $DONE"
echo "Skipped: $SKIP"
echo "Failed: $FAIL"
