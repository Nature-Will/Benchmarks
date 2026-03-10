"""
Compute percentile-based normalization stats (q01/q99) from intermediate HDF5 data.

Outputs JSON matching LingBot-VLA's Normalizer format with keys:
  - observation.state.arm.position (12-dim: left_arm(6) + right_arm(6))
  - observation.state.effector.position (2-dim: left_gripper + right_gripper)
  - action.arm.position (12-dim)
  - action.effector.position (2-dim)

Usage:
    python tools/compute_norm_stats.py \
        --data_dir data/beat_block_hammer/demo_clean-7 \
        --output norm_stats/beat_block_hammer.json
"""
import os
import sys
import json
import argparse
import h5py
import numpy as np
from pathlib import Path

LINGBOT_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lingbot-vla"))
if LINGBOT_REPO not in sys.path:
    sys.path.insert(0, LINGBOT_REPO)

from lingbotvla.utils.normalize import RunningStats, NormStats, serialize_json


def split_arm_effector(data_14d):
    """Split 14-dim [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    into arm(12) and effector(2)."""
    arm = np.concatenate([data_14d[:, :6], data_14d[:, 7:13]], axis=1)  # 12-dim
    effector = np.concatenate([data_14d[:, 6:7], data_14d[:, 13:14]], axis=1)  # 2-dim
    return arm, effector


def compute_stats_from_episodes(data_dir):
    episode_files = sorted(Path(data_dir).glob("episode_*.hdf5"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {data_dir}")

    stats = {
        "observation.state.arm.position": RunningStats(),
        "observation.state.effector.position": RunningStats(),
        "action.arm.position": RunningStats(),
        "action.effector.position": RunningStats(),
    }

    for ep_file in episode_files:
        with h5py.File(ep_file, "r") as f:
            qpos = f["observations/qpos"][()]
            actions = f["action"][()]

        state_arm, state_eff = split_arm_effector(qpos)
        action_arm, action_eff = split_arm_effector(actions)

        stats["observation.state.arm.position"].update(state_arm)
        stats["observation.state.effector.position"].update(state_eff)
        stats["action.arm.position"].update(action_arm)
        stats["action.effector.position"].update(action_eff)

    norm_stats = {}
    total_count = 0
    for key, rs in stats.items():
        ns = rs.get_statistics()
        norm_stats[key] = ns
        total_count = rs._count

    return norm_stats, total_count


def main():
    parser = argparse.ArgumentParser(description="Compute normalization stats for LingBot-VLA")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing intermediate episode_*.hdf5 files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    args = parser.parse_args()

    print(f"Computing norm stats from {args.data_dir}...")
    norm_stats, count = compute_stats_from_episodes(args.data_dir)

    # Print summary
    for key, ns in norm_stats.items():
        print(f"\n{key}:")
        print(f"  mean: {ns.mean}")
        print(f"  std:  {ns.std}")
        print(f"  q01:  {ns.q01}")
        print(f"  q99:  {ns.q99}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_str = serialize_json(norm_stats, count)
    output_path.write_text(json_str)
    print(f"\nNorm stats saved to {output_path}")


if __name__ == "__main__":
    main()
