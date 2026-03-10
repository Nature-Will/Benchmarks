import os
import sys
import json
import numpy as np
import cv2
import torch
from PIL import Image

# Add lingbot-vla repo to path
LINGBOT_REPO = os.path.join(os.path.dirname(__file__), "lingbot-vla")
if LINGBOT_REPO not in sys.path:
    sys.path.insert(0, LINGBOT_REPO)

# Debug logging control
DEBUG_LOG = os.environ.get("LINGBOT_DEBUG", "0") == "1"
_debug_step_count = 0


def _debug_print(*args, **kwargs):
    if DEBUG_LOG:
        print("[DEBUG]", *args, **kwargs, flush=True)


class LingBotVLAWrapper:
    def __init__(self, server):
        self.server = server

    def get_action(self, obs):
        result = self.server.infer(obs)
        return result["action"]

    def reset(self):
        self.server.global_step = 0
        self.server.last_action_chunk = None


def encode_obs(observation):
    obs = observation
    cam_high = obs["observation"]["head_camera"]["rgb"]
    cam_left = obs["observation"]["left_camera"]["rgb"]
    cam_right = obs["observation"]["right_camera"]["rgb"]

    # BGRA -> RGB
    cam_high = cv2.cvtColor(cam_high, cv2.COLOR_BGRA2RGB)
    cam_left = cv2.cvtColor(cam_left, cv2.COLOR_BGRA2RGB)
    cam_right = cv2.cvtColor(cam_right, cv2.COLOR_BGRA2RGB)

    # Build 14-dim qpos: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    qpos = (observation["joint_action"]["left_arm"] +
            [observation["joint_action"]["left_gripper"]] +
            observation["joint_action"]["right_arm"] +
            [observation["joint_action"]["right_gripper"]])
    qpos = np.array(qpos, dtype=np.float32)

    return {
        "observation.images.cam_high": cam_high,
        "observation.images.cam_left_wrist": cam_left,
        "observation.images.cam_right_wrist": cam_right,
        "observation.state": qpos,
    }


def get_model(usr_args):
    # Set QWEN25_PATH env var before importing lingbot modules
    os.environ["QWEN25_PATH"] = usr_args["qwen25_path"]

    from deploy.lingbot_robotwin_policy import QwenPiServer

    model_path = usr_args["model_path"]
    use_length = usr_args.get("use_length", 50)
    chunk_ret = usr_args.get("chunk_ret", True)
    use_bf16 = usr_args.get("use_bf16", True)

    server = QwenPiServer(
        path_to_pi_model=model_path,
        use_length=use_length,
        chunk_ret=chunk_ret,
        use_bf16=use_bf16,
    )

    # Override normalizer with task-specific norm stats if provided
    norm_stats_file = usr_args.get("norm_stats_file")
    if norm_stats_file and os.path.exists(norm_stats_file):
        from lingbotvla.data.vla_data.transform import Normalizer
        with open(norm_stats_file) as f:
            norm_stats = json.load(f)
        server.vla.normalizer = Normalizer(
            norm_stats=norm_stats["norm_stats"],
            from_file=True,
            data_type="robotwin",
            norm_type={
                "observation.images.cam_high": "identity",
                "observation.images.cam_left_wrist": "identity",
                "observation.images.cam_right_wrist": "identity",
                "observation.state": server.data_config.norm_type,
                "action": server.data_config.norm_type,
            },
        )

    # Debug: print normalizer stats
    if DEBUG_LOG:
        ns = server.vla.normalizer.norm_stats
        _debug_print(f"norm_type: {server.vla.normalizer.norm_type}")
        for k in ns:
            if isinstance(ns[k], dict):
                for sk in ns[k]:
                    val = ns[k][sk]
                    if hasattr(val, 'shape'):
                        _debug_print(f"  norm_stats[{k}][{sk}]: shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}]")
                    elif isinstance(val, (list, np.ndarray)):
                        arr = np.array(val)
                        _debug_print(f"  norm_stats[{k}][{sk}]: len={len(arr)}, range=[{arr.min():.4f}, {arr.max():.4f}]")

    model = LingBotVLAWrapper(server)
    return model


def eval(TASK_ENV, model, observation):
    global _debug_step_count
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()
    obs["task"] = str(instruction)

    # Debug: log input state
    if DEBUG_LOG and _debug_step_count < 3:
        _debug_print(f"\n=== Inference step {_debug_step_count} ===")
        _debug_print(f"instruction: {instruction}")
        state = obs["observation.state"]
        _debug_print(f"input state (14D): [{', '.join(f'{v:.4f}' for v in state)}]")

    actions = model.get_action(obs)  # shape: (use_length, 14)

    # Debug: log output actions
    if DEBUG_LOG and _debug_step_count < 3:
        _debug_print(f"actions shape: {actions.shape}")
        _debug_print(f"actions range: min={actions.min():.4f}, max={actions.max():.4f}")
        _debug_print(f"actions[0] (first step): [{', '.join(f'{v:.4f}' for v in actions[0])}]")
        _debug_print(f"actions[24] (mid step):  [{', '.join(f'{v:.4f}' for v in actions[24])}]")
        _debug_print(f"actions[49] (last step): [{', '.join(f'{v:.4f}' for v in actions[49])}]")
        # Check per-dim ranges
        for i in range(14):
            dim_vals = actions[:, i]
            _debug_print(f"  dim {i}: min={dim_vals.min():.4f}, max={dim_vals.max():.4f}, "
                        f"mean={dim_vals.mean():.4f}, std={dim_vals.std():.4f}")
        _debug_step_count += 1

    for action in actions[:50]:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()


def reset_model(model):
    global _debug_step_count
    _debug_step_count = 0
    model.reset()
