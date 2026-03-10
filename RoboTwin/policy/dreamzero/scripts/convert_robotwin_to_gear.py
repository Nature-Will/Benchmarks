"""
Convert RoboTwin HDF5 demo data to LeRobot v2 / GEAR format for DreamZero training.

RoboTwin HDF5 structure (per episode):
  /joint_action/left_arm: (T, 6)
  /joint_action/left_gripper: (T,)
  /joint_action/right_arm: (T, 6)
  /joint_action/right_gripper: (T,)
  /observation/head_camera/rgb: (T,) JPEG-encoded bytes
  /observation/left_camera/rgb: (T,) JPEG-encoded bytes
  /observation/right_camera/rgb: (T,) JPEG-encoded bytes

Output: LeRobot v2 format:
  data/chunk-000/episode_NNNNNN.parquet  (state + action per timestep)
  videos/chunk-000/observation.images.{cam}/episode_NNNNNN.mp4
  meta/{info,modality,embodiment,stats,tasks,episodes}.json

Usage:
  python scripts/convert_robotwin_to_gear.py \
      --task beat_block_hammer \
      --setting demo_clean \
      --robotwin-root ../../ \
      --output-dir ./data/robotwin_beat_block_hammer

  # Then generate GEAR metadata:
  python scripts/data/convert_lerobot_to_gear.py \
      --dataset-path ./data/robotwin_beat_block_hammer \
      --embodiment-tag xdof \
      --state-keys '{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}' \
      --action-keys '{"left_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}' \
      --relative-action-keys left_joint_pos right_joint_pos \
      --task-key annotation.task \
      --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

CAMERA_NAMES = ["head_camera", "right_camera", "left_camera"]
FPS = 15


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to RGB image (H, W, 3)."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_hdf5_episode(hdf5_path: str) -> dict:
    """Load a single RoboTwin HDF5 episode."""
    with h5py.File(hdf5_path, "r") as f:
        left_arm = f["/joint_action/left_arm"][()]       # (T, 6)
        left_gripper = f["/joint_action/left_gripper"][()] # (T,)
        right_arm = f["/joint_action/right_arm"][()]     # (T, 6)
        right_gripper = f["/joint_action/right_gripper"][()] # (T,)

        images = {}
        for cam in CAMERA_NAMES:
            key = f"/observation/{cam}/rgb"
            if key in f:
                images[cam] = f[key][()]

    # Build state vector: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] = 14D
    T = left_arm.shape[0]
    state = np.zeros((T, 14), dtype=np.float64)
    state[:, 0:6] = left_arm
    state[:, 6:7] = left_gripper.reshape(-1, 1)
    state[:, 7:13] = right_arm
    state[:, 13:14] = right_gripper.reshape(-1, 1)

    return {"state": state, "images": images, "T": T}


def convert_dataset(
    task_name: str,
    setting: str,
    robotwin_root: str,
    output_dir: str,
    task_instruction: str | None = None,
    data_dir: str | None = None,
):
    """Convert RoboTwin task data to LeRobot v2 format."""
    if data_dir is None:
        data_dir = os.path.join(robotwin_root, "data", task_name, setting, "data")
    output_path = Path(output_dir)

    # Find all episodes
    episode_files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("episode") and f.endswith(".hdf5")]
    )
    num_episodes = len(episode_files)
    print(f"Found {num_episodes} episodes in {data_dir}")

    if num_episodes == 0:
        print("No episodes found!")
        sys.exit(1)

    # Default instruction
    if task_instruction is None:
        task_instruction = task_name.replace("_", " ")

    # Create output directories
    chunk_dir = output_path / "data" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for cam in CAMERA_NAMES:
        video_dir = output_path / "videos" / "chunk-000" / f"observation.images.{cam}"
        video_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    episode_lengths = []

    for ep_idx in tqdm(range(num_episodes), desc="Converting episodes"):
        hdf5_path = os.path.join(data_dir, f"episode{ep_idx}.hdf5")
        episode = load_hdf5_episode(hdf5_path)
        T = episode["T"]

        # Build state/action arrays
        # state[t] is the state at timestep t
        # action[t] is the action that transitions from state[t] to state[t+1]
        # We use state[t+1] as the action (next-state as action, common in imitation learning)
        usable_T = T - 1  # Last timestep has no next action

        states = episode["state"][:usable_T]  # (usable_T, 14)
        actions = episode["state"][1:]          # (usable_T, 14) - next state as action

        # Build parquet dataframe
        rows = []
        for t in range(usable_T):
            row = {
                "observation.state": states[t].tolist(),
                "action": actions[t].tolist(),
                "episode_index": ep_idx,
                "frame_index": t,
                "timestamp": t / FPS,
                "task_index": 0,
                "annotation.task": task_instruction,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        parquet_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)

        # Save camera videos as mp4
        for cam in CAMERA_NAMES:
            if cam not in episode["images"]:
                continue

            video_path = (
                output_path / "videos" / "chunk-000"
                / f"observation.images.{cam}" / f"episode_{ep_idx:06d}.mp4"
            )

            frames = []
            for t in range(usable_T):
                jpeg_data = episode["images"][cam][t]
                img = decode_jpeg(jpeg_data)
                frames.append(img)

            if frames:
                imageio.mimsave(str(video_path), frames, fps=FPS, codec="libx264")

        episode_lengths.append(usable_T)
        total_frames += usable_T

    # Write meta/info.json
    # Read one frame to get image dimensions
    sample_ep = load_hdf5_episode(os.path.join(data_dir, "episode0.hdf5"))
    sample_img = decode_jpeg(sample_ep["images"]["head_camera"][0])
    img_h, img_w = sample_img.shape[:2]

    info = {
        "codebase_version": "v2.0",
        "robot_type": "robotwin_dual_arm",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "fps": FPS,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float64",
                "shape": [14],
                "names": [
                    "left_joint_0", "left_joint_1", "left_joint_2",
                    "left_joint_3", "left_joint_4", "left_joint_5",
                    "left_gripper",
                    "right_joint_0", "right_joint_1", "right_joint_2",
                    "right_joint_3", "right_joint_4", "right_joint_5",
                    "right_gripper",
                ],
            },
            "action": {
                "dtype": "float64",
                "shape": [14],
                "names": [
                    "left_joint_0", "left_joint_1", "left_joint_2",
                    "left_joint_3", "left_joint_4", "left_joint_5",
                    "left_gripper",
                    "right_joint_0", "right_joint_1", "right_joint_2",
                    "right_joint_3", "right_joint_4", "right_joint_5",
                    "right_gripper",
                ],
            },
            "observation.images.head_camera": {
                "dtype": "video",
                "shape": [img_h, img_w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": FPS, "video.codec": "libx264"},
            },
            "observation.images.right_camera": {
                "dtype": "video",
                "shape": [img_h, img_w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": FPS, "video.codec": "libx264"},
            },
            "observation.images.left_camera": {
                "dtype": "video",
                "shape": [img_h, img_w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": FPS, "video.codec": "libx264"},
            },
            "annotation.task": {
                "dtype": "string",
                "shape": [1],
            },
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Write tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_instruction}) + "\n")

    # Write episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_idx in range(num_episodes):
            ep = {
                "episode_index": ep_idx,
                "tasks": [task_instruction],
                "length": episode_lengths[ep_idx],
            }
            f.write(json.dumps(ep) + "\n")

    print(f"\nConversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Image size: {img_w}x{img_h}")
    print(f"\nNext: run convert_lerobot_to_gear.py to generate GEAR metadata:")
    print(f"  python scripts/data/convert_lerobot_to_gear.py \\")
    print(f"      --dataset-path {output_path} \\")
    print(f"      --embodiment-tag xdof \\")
    print(f"      --state-keys '{{\"left_joint_pos\": [0, 6], \"left_gripper_pos\": [6, 7], \"right_joint_pos\": [7, 13], \"right_gripper_pos\": [13, 14]}}' \\")
    print(f"      --action-keys '{{\"left_joint_pos\": [0, 6], \"left_gripper_pos\": [6, 7], \"right_joint_pos\": [7, 13], \"right_gripper_pos\": [13, 14]}}' \\")
    print(f"      --relative-action-keys left_joint_pos right_joint_pos \\")
    print(f"      --task-key annotation.task \\")
    print(f"      --force")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboTwin HDF5 data to LeRobot v2 / GEAR format")
    parser.add_argument("--task", type=str, default="beat_block_hammer", help="Task name")
    parser.add_argument("--setting", type=str, default="demo_clean", help="Data setting")
    parser.add_argument("--robotwin-root", type=str, default="../../", help="Path to RoboTwin root")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: ./data/robotwin_{task})")
    parser.add_argument("--instruction", type=str, default=None, help="Task instruction text")
    parser.add_argument("--data-dir", type=str, default=None, help="Direct path to HDF5 data directory (overrides robotwin-root/task/setting path)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./data/robotwin_{args.task}"

    convert_dataset(
        task_name=args.task,
        setting=args.setting,
        robotwin_root=args.robotwin_root,
        output_dir=args.output_dir,
        task_instruction=args.instruction,
        data_dir=args.data_dir,
    )
