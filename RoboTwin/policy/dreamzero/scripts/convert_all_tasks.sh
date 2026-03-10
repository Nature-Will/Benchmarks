#!/bin/bash
# Batch convert all 50 RoboTwin tasks to GEAR format for DreamZero training
#
# Converts both demo_randomized (500 eps) and demo_clean (50 eps) for each task.
# Uses GNU parallel for 8-way parallelism.
#
# Usage:
#   bash scripts/convert_all_tasks.sh
#
# Prerequisites:
#   - demo_randomized data at VLA_BENCHS_PATH/{task}/aloha-agilex_randomized_500/data/
#   - demo_clean data at ROBOTWIN_DATA_DIR/{task}/demo_clean/data/
#     (run scripts/download_demo_clean.sh first)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DREAMZERO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DREAMZERO_DIR"

VLA_BENCHS_PATH=${VLA_BENCHS_PATH:-"/mnt/shared-storage-user/p1-shared/yujiale/code/VLA-Benchs/data/original/robotwin"}
ROBOTWIN_DATA_DIR=${ROBOTWIN_DATA_DIR:-"/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/data"}
OUTPUT_BASE="./data"
NUM_WORKERS=${NUM_WORKERS:-8}

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

convert_one_dataset() {
    local task="$1"
    local data_dir="$2"
    local output_dir="$3"
    local label="$4"

    # Skip if already converted and has GEAR metadata
    if [ -f "$output_dir/meta/relative_stats_dreamzero.json" ]; then
        echo "[SKIP] $label - already converted"
        return 0
    fi

    # Check source data exists
    if [ ! -d "$data_dir" ] || ! ls "$data_dir"/episode*.hdf5 1>/dev/null 2>&1; then
        echo "[MISS] $label - no HDF5 data at $data_dir"
        return 1
    fi

    echo "[CONV] $label ..."

    # Step 1: Convert HDF5 to LeRobot format
    if [ ! -f "$output_dir/meta/info.json" ]; then
        python scripts/convert_robotwin_to_gear.py \
            --task "$task" \
            --data-dir "$data_dir" \
            --output-dir "$output_dir" 2>&1 | tail -3
    fi

    # Step 2: Generate GEAR metadata
    python scripts/data/convert_lerobot_to_gear.py \
        --dataset-path "$output_dir" \
        --embodiment-tag xdof \
        --state-keys "$STATE_KEYS" \
        --action-keys "$ACTION_KEYS" \
        --relative-action-keys left_joint_pos right_joint_pos \
        --task-key annotation.task \
        --force 2>&1 | tail -3

    echo "[DONE] $label"
    return 0
}

export -f convert_one_dataset
export VLA_BENCHS_PATH ROBOTWIN_DATA_DIR OUTPUT_BASE STATE_KEYS ACTION_KEYS

echo "=== Batch Convert All Tasks to GEAR Format ==="
echo "Tasks: ${#ALL_TASKS[@]}"
echo "Workers: $NUM_WORKERS"
echo "Output: $OUTPUT_BASE/robotwin_{task}_{rand500,clean50}/"
echo ""

# Build job list
JOB_FILE=$(mktemp)
TOTAL_JOBS=0

for task in "${ALL_TASKS[@]}"; do
    # demo_randomized (500 episodes)
    RAND_DATA="$VLA_BENCHS_PATH/$task/aloha-agilex_randomized_500/data"
    RAND_OUT="$OUTPUT_BASE/robotwin_${task}_rand500"
    echo "$task|$RAND_DATA|$RAND_OUT|${task}_rand500" >> "$JOB_FILE"
    TOTAL_JOBS=$((TOTAL_JOBS + 1))

    # demo_clean (50 episodes)
    CLEAN_DATA="$ROBOTWIN_DATA_DIR/$task/demo_clean/data"
    CLEAN_OUT="$OUTPUT_BASE/robotwin_${task}_clean50"
    echo "$task|$CLEAN_DATA|$CLEAN_OUT|${task}_clean50" >> "$JOB_FILE"
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
done

echo "Total conversion jobs: $TOTAL_JOBS"
echo ""

# Check if GNU parallel is available
if command -v parallel &>/dev/null; then
    echo "Using GNU parallel with $NUM_WORKERS workers..."
    cat "$JOB_FILE" | parallel -j "$NUM_WORKERS" --colsep '\|' \
        "cd $DREAMZERO_DIR && convert_one_dataset {1} {2} {3} {4}"
else
    echo "GNU parallel not found, running sequentially..."
    while IFS='|' read -r task data_dir output_dir label; do
        convert_one_dataset "$task" "$data_dir" "$output_dir" "$label" || true
    done < "$JOB_FILE"
fi

rm -f "$JOB_FILE"

echo ""
echo "=== Conversion Complete ==="

# Validate results
echo ""
echo "=== Validation ==="
TOTAL_DATASETS=0
TOTAL_EPISODES=0
MISSING=0

for task in "${ALL_TASKS[@]}"; do
    for suffix in rand500 clean50; do
        DIR="$OUTPUT_BASE/robotwin_${task}_${suffix}"
        if [ -f "$DIR/meta/relative_stats_dreamzero.json" ]; then
            TOTAL_DATASETS=$((TOTAL_DATASETS + 1))
            # Count episodes from episodes.jsonl
            if [ -f "$DIR/meta/episodes.jsonl" ]; then
                EPS=$(wc -l < "$DIR/meta/episodes.jsonl")
                TOTAL_EPISODES=$((TOTAL_EPISODES + EPS))
            fi
        else
            echo "  [MISSING] robotwin_${task}_${suffix}"
            MISSING=$((MISSING + 1))
        fi
    done
done

echo ""
echo "Valid GEAR datasets: $TOTAL_DATASETS / $TOTAL_JOBS"
echo "Total episodes: $TOTAL_EPISODES"
echo "Missing: $MISSING"
