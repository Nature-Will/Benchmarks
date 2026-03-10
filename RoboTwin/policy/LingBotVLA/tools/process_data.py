"""
Convert RoboTwin HDF5 episodes to intermediate HDF5 format for LingBot-VLA.

Usage:
    python tools/process_data.py beat_block_hammer demo_clean 7

Reads from: ../../data/{task_name}/{setting}/data/episode{i}.hdf5
Writes to:  data/{task_name}/{setting}-{num}/episode_{i}.hdf5
"""
import os
import sys
import h5py
import numpy as np
import cv2
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

TASK_INSTRUCTIONS = {
    "beat_block_hammer": "Use the hammer to beat the block.",
}


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        sys.exit(1)

    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        image_dict = {}
        for cam_name in root["/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def data_transform(path, episode_num, save_path, task_name):
    floders = os.listdir(path)
    assert episode_num <= len(floders), f"data num not enough: {len(floders)} < {episode_num}"

    os.makedirs(save_path, exist_ok=True)

    instruction = TASK_INSTRUCTIONS.get(task_name, f"Complete the {task_name} task.")

    for i in range(episode_num):
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_hdf5(
            os.path.join(path, f"episode{i}.hdf5")
        )

        qpos_list = []
        actions_list = []
        cam_high_list = []
        cam_right_wrist_list = []
        cam_left_wrist_list = []

        num_steps = left_gripper_all.shape[0]

        for j in range(num_steps):
            left_gripper = left_gripper_all[j]
            left_arm = left_arm_all[j]
            right_gripper = right_gripper_all[j]
            right_arm = right_arm_all[j]

            state = np.concatenate(
                (left_arm, [left_gripper], right_arm, [right_gripper]), axis=0
            ).astype(np.float32)

            if j < num_steps - 1:
                qpos_list.append(state)

                camera_high = cv2.imdecode(
                    np.frombuffer(image_dict["head_camera"][j], np.uint8), cv2.IMREAD_COLOR
                )
                cam_high_list.append(camera_high)

                camera_right_wrist = cv2.imdecode(
                    np.frombuffer(image_dict["right_camera"][j], np.uint8), cv2.IMREAD_COLOR
                )
                cam_right_wrist_list.append(camera_right_wrist)

                camera_left_wrist = cv2.imdecode(
                    np.frombuffer(image_dict["left_camera"][j], np.uint8), cv2.IMREAD_COLOR
                )
                cam_left_wrist_list.append(camera_left_wrist)

            if j > 0:
                actions_list.append(state)

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions_list))
            f.create_dataset("language_raw", data=np.array(instruction.encode("utf-8")))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos_list))
            image = obs.create_group("images")
            image.create_dataset("cam_high", data=np.stack(cam_high_list), dtype=np.uint8)
            image.create_dataset(
                "cam_right_wrist", data=np.stack(cam_right_wrist_list), dtype=np.uint8
            )
            image.create_dataset(
                "cam_left_wrist", data=np.stack(cam_left_wrist_list), dtype=np.uint8
            )

        print(f"Processed episode {i} -> {hdf5path} "
              f"(actions: {np.array(actions_list).shape}, "
              f"images: {np.stack(cam_high_list).shape})")

    return episode_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboTwin HDF5 to intermediate format")
    parser.add_argument("task_name", type=str, help="Task name (e.g., beat_block_hammer)")
    parser.add_argument("setting", type=str, help="Setting (e.g., demo_clean)")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Custom path to raw HDF5 data dir (default: ../../data/{task}/{setting}/data)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Custom output path (default: data/{task}/{setting}-{num})")
    args = parser.parse_args()

    data_path = args.data_path or os.path.join("../../data/", args.task_name, args.setting, "data")
    save_path = args.save_path or f"data/{args.task_name}/{args.setting}-{args.expert_data_num}"

    num_processed = data_transform(data_path, args.expert_data_num, save_path, args.task_name)
    print(f"\nDone! Processed {num_processed} episodes.")
