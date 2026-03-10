#!/bin/bash
# Generate missing demo_clean HDF5 files from downloaded seed.txt + _traj_data/*.pkl.
#
# Usage:
#   bash scripts/generate_demo_clean_hdf5.sh
#   TASKS="adjust_bottle beat_block_hammer" bash scripts/generate_demo_clean_hdf5.sh

set -euo pipefail

ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-"/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin"}
PYTHON_BIN=${PYTHON_BIN:-"/mnt/shared-storage-user/p1-shared/yujiale/conda/envs/robotwin/bin/python"}
LOG_DIR=${LOG_DIR:-"$ROBOTWIN_ROOT/policy/dreamzero/logs/demo_clean_hdf5"}

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

if [ -n "${TASKS:-}" ]; then
    read -r -a TASK_LIST <<< "$TASKS"
else
    TASK_LIST=("${ALL_TASKS[@]}")
fi

mkdir -p "$LOG_DIR"

echo "=== Generate demo_clean HDF5 ==="
echo "RoboTwin root: $ROBOTWIN_ROOT"
echo "Python: $PYTHON_BIN"
echo "Tasks: ${#TASK_LIST[@]}"
echo "Logs: $LOG_DIR"
echo ""

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python not found or not executable: $PYTHON_BIN"
    exit 1
fi

cd "$ROBOTWIN_ROOT"

DONE=0
SKIP=0
MISS=0
FAIL=0

for task in "${TASK_LIST[@]}"; do
    task_dir="$ROBOTWIN_ROOT/data/$task/demo_clean"
    traj_dir="$task_dir/_traj_data"
    data_dir="$task_dir/data"
    seed_file="$task_dir/seed.txt"
    log_file="$LOG_DIR/${task}.log"

    hdf5_count=$(find "$data_dir" -maxdepth 1 -type f -name 'episode*.hdf5' 2>/dev/null | wc -l || true)
    pkl_count=$(find "$traj_dir" -maxdepth 1 -type f -name 'episode*.pkl' 2>/dev/null | wc -l || true)

    if [ "$hdf5_count" -ge 50 ]; then
        echo "[SKIP] $task - already has 50 hdf5"
        SKIP=$((SKIP + 1))
        continue
    fi

    if [ "$pkl_count" -lt 50 ] || [ ! -f "$seed_file" ]; then
        echo "[MISS] $task - need 50 pkl + seed.txt first (pkl=$pkl_count)"
        MISS=$((MISS + 1))
        continue
    fi

    echo "[GEN ] $task - hdf5=$hdf5_count, pkl=$pkl_count"
    if "$PYTHON_BIN" script/collect_data.py "$task" demo_clean >"$log_file" 2>&1; then
        new_count=$(find "$data_dir" -maxdepth 1 -type f -name 'episode*.hdf5' 2>/dev/null | wc -l || true)
        if [ "$new_count" -ge 50 ]; then
            echo "[DONE] $task - hdf5=$new_count"
            DONE=$((DONE + 1))
        else
            echo "[FAIL] $task - collect_data.py exited 0 but hdf5=$new_count"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "[FAIL] $task - see $log_file"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Done: $DONE"
echo "Skipped: $SKIP"
echo "Missing pkl/seed: $MISS"
echo "Failed: $FAIL"
