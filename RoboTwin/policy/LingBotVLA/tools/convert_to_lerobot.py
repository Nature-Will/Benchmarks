"""
Convert intermediate HDF5 episodes to LeRobot v3.0 dataset format (lerobot 0.4.4 compatible).

Uses embedded images in parquet (no individual PNG files).
Processes episodes in chunks to manage memory.

Usage:
    python tools/convert_to_lerobot.py \
        --data_dir data/beat_block_hammer/randomized-500 \
        --output_dir lerobot_data/beat_block_hammer_randomized500 \
        --task_name beat_block_hammer \
        --fps 30 --chunk_episodes 50
"""
import os
import sys
import json
import argparse
import gc
import h5py
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image
import time

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import datasets


TASK_INSTRUCTIONS = {
    "beat_block_hammer": "Use the hammer to beat the block.",
}


def encode_image_to_bytes(img_array_bgr):
    """Encode a BGR numpy array to JPEG bytes."""
    img_rgb = img_array_bgr[:, :, ::-1].copy()
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def get_hf_features():
    return datasets.Features({
        "episode_index": datasets.Value("int64"),
        "frame_index": datasets.Value("int64"),
        "index": datasets.Value("int64"),
        "timestamp": datasets.Value("float32"),
        "task_index": datasets.Value("int64"),
        "observation.state": datasets.Sequence(
            feature=datasets.Value("float32"), length=14
        ),
        "action": datasets.Sequence(
            feature=datasets.Value("float32"), length=14
        ),
        "action_is_pad": datasets.Value("bool"),
        "observation.images.cam_high": datasets.Image(),
        "observation.images.cam_left_wrist": datasets.Image(),
        "observation.images.cam_right_wrist": datasets.Image(),
    })


def process_episode_chunk(episode_files, ep_start_idx, global_idx_start, fps, task_instruction):
    """Process a chunk of episodes, return dict for HF Dataset + episodes metadata."""
    all_data = {
        "episode_index": [], "frame_index": [], "index": [], "timestamp": [],
        "task_index": [], "observation.state": [], "action": [], "action_is_pad": [],
        "observation.images.cam_high": [], "observation.images.cam_left_wrist": [],
        "observation.images.cam_right_wrist": [],
    }
    episodes_meta = []
    global_idx = global_idx_start
    all_states_raw = []
    all_actions_raw = []

    for local_idx, ep_file in enumerate(episode_files):
        ep_idx = ep_start_idx + local_idx
        with h5py.File(ep_file, "r") as f:
            actions = f["action"][()]
            qpos = f["observations/qpos"][()]
            cam_high = f["observations/images/cam_high"][()]
            cam_left = f["observations/images/cam_left_wrist"][()]
            cam_right = f["observations/images/cam_right_wrist"][()]

        num_frames = qpos.shape[0]
        ep_frame_start = global_idx

        for frame_idx in range(num_frames):
            all_data["episode_index"].append(ep_idx)
            all_data["frame_index"].append(frame_idx)
            all_data["index"].append(global_idx)
            all_data["timestamp"].append(np.float32(frame_idx / fps))
            all_data["task_index"].append(0)
            all_data["observation.state"].append(qpos[frame_idx].tolist())

            if frame_idx < actions.shape[0]:
                all_data["action"].append(actions[frame_idx].tolist())
                all_data["action_is_pad"].append(False)
                all_actions_raw.append(actions[frame_idx])
            else:
                all_data["action"].append(actions[-1].tolist())
                all_data["action_is_pad"].append(True)

            all_data["observation.images.cam_high"].append(
                {"path": None, "bytes": encode_image_to_bytes(cam_high[frame_idx])})
            all_data["observation.images.cam_left_wrist"].append(
                {"path": None, "bytes": encode_image_to_bytes(cam_left[frame_idx])})
            all_data["observation.images.cam_right_wrist"].append(
                {"path": None, "bytes": encode_image_to_bytes(cam_right[frame_idx])})

            all_states_raw.append(qpos[frame_idx])
            global_idx += 1

        episodes_meta.append({
            "episode_index": ep_idx,
            "length": num_frames,
            "dataset_from_index": ep_frame_start,
            "dataset_to_index": global_idx,
        })

    return all_data, episodes_meta, global_idx, all_states_raw, all_actions_raw


def convert_to_lerobot(data_dir, output_dir, task_name, fps=30, chunk_episodes=50):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    episode_files = sorted(data_dir.glob("episode_*.hdf5"),
                           key=lambda p: int(p.stem.split("_")[1]))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.hdf5 in {data_dir}")

    total_episodes = len(episode_files)
    print(f"Found {total_episodes} episodes in {data_dir}")

    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    task_instruction = TASK_INSTRUCTIONS.get(task_name, f"Complete the {task_name} task.")
    with h5py.File(episode_files[0], "r") as f:
        if "language_raw" in f:
            raw = f["language_raw"][()]
            if isinstance(raw, bytes):
                task_instruction = raw.decode("utf-8")

    hf_features = get_hf_features()

    # Process in chunks
    all_episodes_meta = []
    all_states = []
    all_actions = []
    global_idx = 0
    file_idx = 0
    total_start = time.time()

    for chunk_start in range(0, total_episodes, chunk_episodes):
        chunk_end = min(chunk_start + chunk_episodes, total_episodes)
        chunk_files = episode_files[chunk_start:chunk_end]

        print(f"\n--- Chunk: episodes {chunk_start}-{chunk_end-1} ({len(chunk_files)} episodes) ---")
        chunk_start_time = time.time()

        data, ep_metas, global_idx, states, actions = process_episode_chunk(
            chunk_files, chunk_start, global_idx, fps, task_instruction
        )
        all_episodes_meta.extend(ep_metas)
        all_states.extend(states)
        all_actions.extend(actions)

        # Write chunk to parquet
        ds = datasets.Dataset.from_dict(data, features=hf_features)
        parquet_path = output_dir / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet"
        ds.to_parquet(str(parquet_path))

        chunk_time = time.time() - chunk_start_time
        total_elapsed = time.time() - total_start
        eps_done = chunk_end
        eta = total_elapsed / eps_done * (total_episodes - eps_done)
        print(f"  Written {parquet_path.name} ({len(ds)} frames, {chunk_time:.1f}s, ETA: {eta/60:.1f}min)")

        # Update file index for episodes in this chunk
        for em in ep_metas:
            em["data/file_index"] = file_idx
        file_idx += 1

        # Free memory
        del data, ds
        gc.collect()

    # Write tasks.parquet
    print("\nWriting tasks.parquet...")
    tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index([task_instruction], name="task"))
    tasks_df.to_parquet(output_dir / "meta" / "tasks.parquet")

    # Write episodes parquet
    print("Writing episodes.parquet...")
    ep_table = pa.table({
        "episode_index": [e["episode_index"] for e in all_episodes_meta],
        "tasks": [[task_instruction] for _ in all_episodes_meta],
        "length": [e["length"] for e in all_episodes_meta],
        "task_index": [0 for _ in all_episodes_meta],
        "data/chunk_index": [0 for _ in all_episodes_meta],
        "data/file_index": [e.get("data/file_index", 0) for e in all_episodes_meta],
        "dataset_from_index": [e["dataset_from_index"] for e in all_episodes_meta],
        "dataset_to_index": [e["dataset_to_index"] for e in all_episodes_meta],
    })
    pq.write_table(ep_table, output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # Write info.json
    print("Writing info.json...")
    info = {
        "codebase_version": "v3.0",
        "robot_type": "dual_arm",
        "total_episodes": total_episodes,
        "total_frames": global_idx,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 500,
        "video_files_size_in_mb": 500,
        "fps": fps,
        "splits": {"train": f"0:{global_idx}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.state": {"dtype": "float32", "shape": [14], "names": None},
            "action": {"dtype": "float32", "shape": [14], "names": None},
            "action_is_pad": {"dtype": "bool", "shape": [1], "names": None},
            "observation.images.cam_high": {
                "dtype": "image", "shape": [240, 320, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam_left_wrist": {
                "dtype": "image", "shape": [240, 320, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam_right_wrist": {
                "dtype": "image", "shape": [240, 320, 3],
                "names": ["height", "width", "channels"],
            },
        },
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Write stats.json
    print("Writing stats.json...")
    np_states = np.array(all_states)
    np_actions = np.array(all_actions)
    stats = {
        "observation.state": {
            "mean": np_states.mean(axis=0).tolist(),
            "std": np_states.std(axis=0).tolist(),
            "min": np_states.min(axis=0).tolist(),
            "max": np_states.max(axis=0).tolist(),
        },
        "action": {
            "mean": np_actions.mean(axis=0).tolist(),
            "std": np_actions.std(axis=0).tolist(),
            "min": np_actions.min(axis=0).tolist(),
            "max": np_actions.max(axis=0).tolist(),
        },
    }
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    total_time = time.time() - total_start
    print(f"\nLeRobot dataset created at {output_dir}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Total frames: {global_idx}")
    print(f"  Parquet files: {file_idx}")
    print(f"  Task: {task_instruction}")
    print(f"  Total time: {total_time/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert intermediate HDF5 to LeRobot v3.0 format")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--chunk_episodes", type=int, default=50,
                        help="Number of episodes per parquet chunk (for memory management)")
    args = parser.parse_args()
    convert_to_lerobot(args.data_dir, args.output_dir, args.task_name, args.fps, args.chunk_episodes)
