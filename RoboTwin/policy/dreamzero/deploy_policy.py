"""
DreamZero deploy_policy.py for RoboTwin evaluation.

Connects to a running DreamZero inference server via WebSocket.
Follows the standard RoboTwin policy interface: get_model, eval, reset_model.

The server must be started separately (see start_server.sh).
"""

import numpy as np
import cv2
import os
import sys
import logging
import time
import subprocess
from pathlib import Path
from typing import Optional

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from eval_utils.policy_client import WebsocketClientPolicy

logger = logging.getLogger(__name__)


# ============ 3-view comparison video utilities ============

def add_title_bar(img, text, font_scale=0.7, thickness=2):
    """Add a black title bar with text above the image."""
    h, w = img.shape[:2]
    bar_height = 32
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = (w - tw) // 2
    ty = (bar_height + th) // 2 - 3
    cv2.putText(title_bar, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([title_bar, img])


def letterbox(src, target_h, target_w):
    """Resize keeping aspect ratio, pad with black to target size."""
    sh, sw = src.shape[:2]
    scale = min(target_w / sw, target_h / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resized = cv2.resize(src, (nw, nh))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def save_comparison_video(real_obs_list, pred_video_path, save_path, fps=5):
    """Save a comparison video: real 3-view obs (top) vs predicted 2x2 split (bottom).

    Args:
        real_obs_list: list of dicts with keys 'head', 'right', 'left', each (H,W,3) uint8
        pred_video_path: path to DreamZero's predicted 2x2 tiled video (or None)
        save_path: output video path
        fps: output fps
    """
    if not real_obs_list:
        return

    n_real = len(real_obs_list)
    base_h = real_obs_list[0]['head'].shape[0]  # 240

    # Load predicted video frames
    pred_frames = []
    if pred_video_path and os.path.exists(pred_video_path):
        cap = cv2.VideoCapture(pred_video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pred_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    n_pred = len(pred_frames)

    logger.info(f"Saving comparison: {n_real} real frames, {n_pred} predicted frames -> {save_path}")

    # Use ffmpeg pipe for H.264 output
    view_w = real_obs_list[0]['head'].shape[1]  # 320
    grid_w = view_w * 3
    grid_h = (base_h + 32) * 2  # 2 rows with title bars

    proc = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{grid_w}x{grid_h}",
        "-framerate", str(fps),
        "-i", "-",
        "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20",
        save_path
    ], stdin=subprocess.PIPE)

    for i in range(n_real):
        obs = real_obs_list[i]
        # Row 1: Real observation 3 views
        row_real = np.hstack([obs['head'], obs['right'], obs['left']])
        row_real = add_title_bar(row_real, "Real Observation (Head / Right / Left)")

        # Row 2: Predicted views (split from 2x2 tiled frame)
        if i < n_pred:
            pf = pred_frames[i]
            ph, pw = pf.shape[:2]
            mid_h, mid_w = ph // 2, pw // 2
            pred_head = pf[:mid_h, :mid_w]
            pred_right = pf[:mid_h, mid_w:]
            pred_left = pf[mid_h:, :mid_w]
            row_pred = np.hstack([
                letterbox(pred_head, base_h, view_w),
                letterbox(pred_right, base_h, view_w),
                letterbox(pred_left, base_h, view_w),
            ])
        else:
            row_pred = np.zeros((base_h, grid_w, 3), dtype=np.uint8)
            cv2.putText(row_pred, "No prediction", (grid_w // 2 - 80, base_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        row_pred = add_title_bar(row_pred, "DreamZero Predicted (Head / Right / Left)")

        full = np.vstack([row_real, row_pred])
        proc.stdin.write(full.tobytes())

    proc.stdin.close()
    proc.wait()
    logger.info(f"Comparison video saved: {save_path}")

# Action chunk size: DreamZero outputs 24 action steps per inference
ACTION_HORIZON = 24
# Image resolution expected by DreamZero (RoboTwin D435 native: 240x320)
TARGET_IMG_H = 240
TARGET_IMG_W = 320


class DreamZeroModel:
    """WebSocket client wrapper for DreamZero inference server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        logger.info(f"Connecting to DreamZero server at {host}:{port}...")
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.client.get_server_metadata()
        logger.info(f"Connected. Server metadata: {self.metadata}")

        self.observation_window = None
        self.instruction = None
        self.session_id = f"robotwin_{int(time.time())}"
        self._call_count = 0

        # For 3-view comparison video
        self.full_obs_list = []  # list of {'head': img, 'right': img, 'left': img}
        self.pred_video_path = None  # set by eval to latest predicted video

    def set_language(self, instruction: str):
        self.instruction = instruction
        logger.info(f"Set instruction: {instruction}")

    def get_action(self, images, state):
        """Send observation to server and get action chunk.

        Args:
            images: list of 3 images [head_camera, right_camera, left_camera],
                    each (H, W, 3) uint8
            state: (14,) float array [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]

        Returns:
            actions: (N, 14) float32 action chunk
        """
        # Resize images to target resolution
        resized_images = []
        for img in images:
            if img.shape[0] != TARGET_IMG_H or img.shape[1] != TARGET_IMG_W:
                img = cv2.resize(img, (TARGET_IMG_W, TARGET_IMG_H))
            resized_images.append(img)

        # Split state into dual-arm components
        left_joint_pos = state[0:6].astype(np.float64)
        left_gripper_pos = state[6:7].astype(np.float64)
        right_joint_pos = state[7:13].astype(np.float64)
        right_gripper_pos = state[13:14].astype(np.float64)

        obs = {
            "observation/head_camera": resized_images[0],
            "observation/right_camera": resized_images[1],
            "observation/left_camera": resized_images[2],
            "observation/left_joint_pos": left_joint_pos,
            "observation/left_gripper_pos": left_gripper_pos,
            "observation/right_joint_pos": right_joint_pos,
            "observation/right_gripper_pos": right_gripper_pos,
            "prompt": self.instruction or "",
            "session_id": self.session_id,
            "endpoint": "infer",
        }

        result = self.client.infer(obs)
        self._call_count += 1

        # Result is (N, 14) numpy array
        if isinstance(result, np.ndarray):
            return result
        elif isinstance(result, dict):
            # Fallback: reconstruct from dict
            parts = []
            for key in ["action.left_joint_pos", "action.left_gripper_pos",
                        "action.right_joint_pos", "action.right_gripper_pos"]:
                if key in result:
                    v = result[key]
                    if v.ndim == 1:
                        v = v.reshape(-1, 1) if "gripper" in key else v.reshape(1, -1)
                    parts.append(v)
            if parts:
                return np.concatenate(parts, axis=-1)
            return np.zeros((ACTION_HORIZON, 14), dtype=np.float32)
        else:
            return np.zeros((ACTION_HORIZON, 14), dtype=np.float32)

    def reset(self):
        """Reset model state for new episode."""
        self.observation_window = None
        self.instruction = None
        self.session_id = f"robotwin_{int(time.time())}"
        self._call_count = 0
        self.full_obs_list = []
        self.pred_video_path = None
        try:
            self.client.reset({})
        except Exception as e:
            logger.warning(f"Reset error (non-fatal): {e}")


def encode_obs(observation):
    """Extract images and state from RoboTwin observation dict."""
    images = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    state = observation["joint_action"]["vector"]
    return images, state


def get_model(usr_args):
    """Initialize DreamZero model (WebSocket client)."""
    host = usr_args.get("server_host", "0.0.0.0")
    port = usr_args.get("server_port", 5000)
    return DreamZeroModel(host=host, port=port)


def eval(TASK_ENV, model, observation):
    """Run one evaluation step: get action chunk and execute."""
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)
        model.observation_window = True  # Mark as initialized

    images, state = encode_obs(observation)

    # Collect initial observation only on first call (avoid duplicates at chunk boundaries)
    if len(model.full_obs_list) == 0:
        model.full_obs_list.append({
            'head': images[0].copy(),
            'right': images[1].copy(),
            'left': images[2].copy(),
        })

    actions = model.get_action(images, state)

    # Execute action chunk
    num_actions = min(len(actions), ACTION_HORIZON)
    for i in range(num_actions):
        action = actions[i]
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()

        # Collect 3-view observation after each action step
        obs_imgs, _ = encode_obs(observation)
        model.full_obs_list.append({
            'head': obs_imgs[0].copy(),
            'right': obs_imgs[1].copy(),
            'left': obs_imgs[2].copy(),
        })

        if TASK_ENV.eval_success:
            break


def _find_pred_video(ep_id):
    """Find the predicted video saved by DreamZero server for this episode.
    Server saves to: checkpoints/dreamzero_robotwin_lora/robotwin_eval_*/checkpoint-*/{seq:06d}_*.mp4
    """
    import glob
    pattern = os.path.join(parent_directory,
                           "checkpoints", "dreamzero_robotwin_lora",
                           "robotwin_eval_*", "checkpoint-*",
                           f"{ep_id:06d}_*.mp4")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    return None


def save_episode_comparison(TASK_ENV, model):
    """Save 3-view comparison video for the completed episode.
    Call this after the episode loop ends (success or step_lim reached).
    Saves first 10 episodes + all successful episodes to limit disk usage.
    """
    if not model.full_obs_list:
        return

    ep_id = getattr(TASK_ENV, 'test_num', 0)
    succ = getattr(TASK_ENV, 'eval_success', False)

    # Only save first 10 episodes + all successes to limit disk
    if ep_id >= 10 and not succ:
        logger.info(f"Skipping comparison video for episode {ep_id} (fail, ep>=10)")
        return

    # Auto-discover predicted video from server output
    pred_video = _find_pred_video(ep_id)
    if pred_video:
        logger.info(f"Found predicted video: {pred_video}")
    else:
        logger.warning(f"No predicted video found for episode {ep_id}")

    # Determine save path
    vis_dir = os.path.join(parent_directory, "comparison_videos")
    os.makedirs(vis_dir, exist_ok=True)
    tag = "succ" if succ else "fail"
    save_path = os.path.join(vis_dir, f"episode{ep_id}_{tag}.mp4")

    try:
        save_comparison_video(
            real_obs_list=model.full_obs_list,
            pred_video_path=pred_video,
            save_path=save_path,
            fps=10,
        )
    except Exception as e:
        logger.warning(f"Failed to save comparison video: {e}")


def reset_model(model):
    """Reset model between episodes."""
    model.reset()
