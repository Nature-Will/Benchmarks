"""Parallel conversion of all 50 RoboTwin tasks to GEAR format.

Usage:
    python scripts/convert_all_parallel.py --workers 8
"""

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool, current_process

ALL_TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
    "handover_block", "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad",
    "open_laptop", "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_bread_skillet",
    "place_burger_fries", "place_can_basket", "place_cans_plasticbox", "place_container_plate",
    "place_dual_shoes", "place_empty_cup", "place_fan", "place_mouse_pad",
    "place_object_basket", "place_object_scale", "place_object_stand", "place_phone_stand",
    "place_shoe", "press_stapler", "put_bottles_dustbin", "put_object_cabinet",
    "rotate_qrcode", "scan_object", "shake_bottle", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two",
    "stamp_seal", "turn_switch",
]

VLA_BENCHS_PATH = "/mnt/shared-storage-user/p1-shared/yujiale/code/VLA-Benchs/data/original/robotwin"

STATE_KEYS = '{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}'
ACTION_KEYS = '{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}'


def convert_task(task: str) -> str:
    """Convert one task's demo_randomized data to GEAR format."""
    output_dir = f"./data/robotwin_{task}_rand500"
    data_dir = f"{VLA_BENCHS_PATH}/{task}/aloha-agilex_randomized_500/data"

    # Skip if already fully converted
    gear_marker = f"{output_dir}/meta/relative_stats_dreamzero.json"
    if os.path.exists(gear_marker):
        return f"[SKIP] {task}"

    # Check source data
    if not os.path.isdir(data_dir):
        return f"[MISS] {task} - no data at {data_dir}"

    t0 = time.time()

    try:
        # Step 1: HDF5 → LeRobot
        info_json = f"{output_dir}/meta/info.json"
        if not os.path.exists(info_json):
            result = subprocess.run(
                [
                    sys.executable, "scripts/convert_robotwin_to_gear.py",
                    "--task", task,
                    "--data-dir", data_dir,
                    "--output-dir", output_dir,
                ],
                capture_output=True, text=True, timeout=1800,
            )
            if result.returncode != 0:
                return f"[FAIL] {task} step1: {result.stderr[-200:]}"

        # Step 2: GEAR metadata
        result = subprocess.run(
            [
                sys.executable, "scripts/data/convert_lerobot_to_gear.py",
                "--dataset-path", output_dir,
                "--embodiment-tag", "xdof",
                "--state-keys", STATE_KEYS,
                "--action-keys", ACTION_KEYS,
                "--relative-action-keys", "left_joint_pos", "right_joint_pos",
                "--task-key", "annotation.task",
                "--force",
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return f"[FAIL] {task} step2: {result.stderr[-200:]}"

        elapsed = time.time() - t0
        return f"[DONE] {task} ({elapsed:.0f}s)"

    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] {task}"
    except Exception as e:
        return f"[ERROR] {task}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print(f"=== Parallel conversion: {len(ALL_TASKS)} tasks, {args.workers} workers ===")
    print()

    t0 = time.time()

    with Pool(args.workers) as pool:
        results = []
        for result in pool.imap_unordered(convert_task, ALL_TASKS):
            print(result, flush=True)
            results.append(result)

    elapsed = time.time() - t0

    # Summary
    done = sum(1 for r in results if r.startswith("[DONE]"))
    skip = sum(1 for r in results if r.startswith("[SKIP]"))
    fail = sum(1 for r in results if not r.startswith("[DONE]") and not r.startswith("[SKIP]"))

    print()
    print(f"=== Complete in {elapsed:.0f}s ===")
    print(f"Done: {done}, Skipped: {skip}, Failed: {fail}")

    # Count total episodes
    total_eps = 0
    total_datasets = 0
    for task in ALL_TASKS:
        ep_file = f"./data/robotwin_{task}_rand500/meta/episodes.jsonl"
        if os.path.exists(ep_file):
            with open(ep_file) as f:
                total_eps += sum(1 for _ in f)
            total_datasets += 1

    print(f"GEAR datasets: {total_datasets}")
    print(f"Total episodes: {total_eps}")


if __name__ == "__main__":
    main()
