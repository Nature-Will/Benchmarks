"""
DreamZero inference server adapted for RoboTwin dual-arm evaluation.

Handles:
- Dual-arm state mapping (14D → left/right joint_pos + gripper_pos)
- 3 cameras (head, right, left)
- 14D action output

Usage:
    torchrun --nproc_per_node=2 --standalone socket_test_robotwin.py \
        --port 5000 --enable-dit-cache \
        --model-path /path/to/finetuned/checkpoint
"""

import dataclasses
import logging
import socket
import asyncio
import os
import datetime
import time
import traceback

import torch
import tyro
import numpy as np
from einops import rearrange
import imageio

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from tianshou.data import Batch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    port: int = 5000
    timeout_seconds: int = 50000
    model_path: str = "./checkpoints/dreamzero_robotwin_lora"
    enable_dit_cache: bool = False
    index: int = 0
    max_chunk_size: int | None = None


class RoboTwinPolicy:
    """Wrapper for DreamZero inference with RoboTwin dual-arm observations.

    Input format (from deploy_policy.py client):
        observation/head_camera: (H, W, 3) or (T, H, W, 3) uint8
        observation/right_camera: (H, W, 3) or (T, H, W, 3) uint8
        observation/left_camera: (H, W, 3) or (T, H, W, 3) uint8
        observation/left_joint_pos: (6,) float64
        observation/left_gripper_pos: (1,) float64
        observation/right_joint_pos: (6,) float64
        observation/right_gripper_pos: (1,) float64
        prompt: str
        session_id: str

    Output format:
        action: (N, 14) float32 [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
    """

    FRAMES_PER_CHUNK = 4

    def __init__(self, groot_policy, signal_group, output_dir=None):
        self._policy = groot_policy
        self._signal_group = signal_group
        self._output_dir = output_dir

        self._frame_buffers = {
            "video.head_camera": [],
            "video.right_camera": [],
            "video.left_camera": [],
        }
        self._call_count = 0
        self._is_first_call = True
        self._current_session_id = None
        self.video_across_time = []
        self._msg_index = 0

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _convert_observation(self, obs):
        converted = {}

        image_key_mapping = {
            "observation/head_camera": "video.head_camera",
            "observation/right_camera": "video.right_camera",
            "observation/left_camera": "video.left_camera",
        }

        for src_key, dst_key in image_key_mapping.items():
            if src_key in obs:
                data = obs[src_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        self._frame_buffers[dst_key].extend(list(data))
                    else:
                        self._frame_buffers[dst_key].append(data)

        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK

        for dst_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    frames_to_use = buffer[-num_frames:]
                else:
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        frames_to_use.insert(0, buffer[0])
                converted[dst_key] = np.stack(frames_to_use, axis=0)

        # State: dual-arm split into 4 modality keys
        if "observation/left_joint_pos" in obs:
            val = obs["observation/left_joint_pos"]
            converted["state.left_joint_pos"] = np.array(val, dtype=np.float64).reshape(1, -1)
        else:
            converted["state.left_joint_pos"] = np.zeros((1, 6), dtype=np.float64)

        if "observation/left_gripper_pos" in obs:
            val = obs["observation/left_gripper_pos"]
            converted["state.left_gripper_pos"] = np.array(val, dtype=np.float64).reshape(1, -1)
        else:
            converted["state.left_gripper_pos"] = np.zeros((1, 1), dtype=np.float64)

        if "observation/right_joint_pos" in obs:
            val = obs["observation/right_joint_pos"]
            converted["state.right_joint_pos"] = np.array(val, dtype=np.float64).reshape(1, -1)
        else:
            converted["state.right_joint_pos"] = np.zeros((1, 6), dtype=np.float64)

        if "observation/right_gripper_pos" in obs:
            val = obs["observation/right_gripper_pos"]
            converted["state.right_gripper_pos"] = np.array(val, dtype=np.float64).reshape(1, -1)
        else:
            converted["state.right_gripper_pos"] = np.zeros((1, 1), dtype=np.float64)

        if "prompt" in obs:
            converted["annotation.language.action_text"] = obs["prompt"]
        else:
            converted["annotation.language.action_text"] = ""

        return converted

    def _convert_action(self, action_dict):
        """Convert model action dict to (N, 14) array.

        Model outputs:
            action.left_joint_pos: (N, 6)
            action.left_gripper_pos: (N, 1)
            action.right_joint_pos: (N, 6)
            action.right_gripper_pos: (N, 1)

        Output: (N, 14) = [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
        """
        parts = {}
        for key, value in action_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            if value.ndim == 1:
                value = value.reshape(-1, 1)  # (N,) → (N, 1) for gripper
            parts[key] = value

        # Find max horizon N (joint positions typically have full horizon)
        N = max((v.shape[0] for v in parts.values()), default=1)
        if N == 0:
            return np.zeros((1, 14), dtype=np.float32)

        left_joint = parts.get("action.left_joint_pos", np.zeros((N, 6)))
        left_gripper = parts.get("action.left_gripper_pos", np.zeros((N, 1)))
        right_joint = parts.get("action.right_joint_pos", np.zeros((N, 6)))
        right_gripper = parts.get("action.right_gripper_pos", np.zeros((N, 1)))

        if left_gripper.ndim == 1:
            left_gripper = left_gripper.reshape(-1, 1)
        if right_gripper.ndim == 1:
            right_gripper = right_gripper.reshape(-1, 1)

        # Broadcast grippers to match joint horizon if needed
        if left_gripper.shape[0] < N:
            left_gripper = np.broadcast_to(left_gripper[-1:], (N, left_gripper.shape[1])).copy()
        if right_gripper.shape[0] < N:
            right_gripper = np.broadcast_to(right_gripper[-1:], (N, right_gripper.shape[1])).copy()

        action = np.concatenate([left_joint, left_gripper, right_joint, right_gripper], axis=-1)
        return action.astype(np.float32)

    def _broadcast_batch_to_workers(self, obs):
        import pickle
        serialized = pickle.dumps(obs)
        size_tensor = torch.tensor([len(serialized)], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    def infer(self, obs):
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info(f"Session changed, resetting state")
                self._reset_state()
            self._current_session_id = session_id

        self._msg_index += 1
        self._call_count += 1

        converted_obs = self._convert_observation(obs)

        # Signal workers to continue
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

        self._broadcast_batch_to_workers(converted_obs)

        batch = Batch(obs=converted_obs)

        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        self.video_across_time.append(video_pred)

        action_chunk = result_batch.act
        action_dict = {}

        # Extract action keys from Tianshou Batch (nested structure)
        if isinstance(action_chunk, Batch):
            # Tianshou Batch with dotted keys creates nested Batches
            # e.g. {"action.left_joint_pos": arr} → batch.action.left_joint_pos
            if hasattr(action_chunk, 'action'):
                action_sub = action_chunk.action
                if isinstance(action_sub, Batch):
                    for k in action_sub.keys():
                        val = action_sub[k]
                        if isinstance(val, (np.ndarray, torch.Tensor)):
                            action_dict[f"action.{k}"] = val
                elif isinstance(action_sub, (np.ndarray, torch.Tensor)):
                    # Single concatenated action tensor — split manually
                    action_dict["action"] = action_sub
            # Fallback: try __getstate__ for flattened keys
            if not action_dict:
                try:
                    state_dict = action_chunk.__getstate__()
                    for k, v in state_dict.items():
                        if k.startswith("action") and isinstance(v, (np.ndarray, torch.Tensor)):
                            action_dict[k] = v
                except Exception:
                    pass

        if self._call_count <= 2:
            logger.info(f"Action dict keys: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in action_dict.items()]}")

        action = self._convert_action(action_dict)

        if self._is_first_call:
            self._is_first_call = False

        return action

    def _reset_state(self, save_video=True):
        if save_video and len(self.video_across_time) > 0 and self._output_dir:
            try:
                video_cat = torch.cat(self.video_across_time, dim=2)
                frames = self._policy.trained_model.action_head.vae.decode(
                    video_cat,
                    tiled=self._policy.trained_model.action_head.tiled,
                    tile_size=(self._policy.trained_model.action_head.tile_size_height,
                               self._policy.trained_model.action_head.tile_size_width),
                    tile_stride=(self._policy.trained_model.action_head.tile_stride_height,
                                  self._policy.trained_model.action_head.tile_stride_width),
                )
                frames = rearrange(frames, "B C T H W -> B T H W C")[0]
                frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)

                save_dir = self._output_dir
                os.makedirs(save_dir, exist_ok=True)
                all_mp4 = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                ts = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                path = os.path.join(save_dir, f"{len(all_mp4):06}_{ts}.mp4")
                imageio.mimsave(path, list(frames), fps=5, codec='libx264')
                logger.info(f"Saved video: {path}")
            except Exception as e:
                logger.warning(f"Failed to save video: {e}")

        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []

    def reset(self, reset_info):
        self._reset_state(save_video=True)


class RoboTwinServer:
    """WebSocket server for RoboTwin DreamZero inference."""

    def __init__(self, policy, host="0.0.0.0", port=5000, metadata=None,
                 output_dir=None, signal_group=None):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_dir = output_dir
        self._signal_group = signal_group
        self.video_across_time = []

    def serve_forever(self, rank=0):
        asyncio.run(self.run(rank))

    async def run(self, rank=0):
        if rank == 0:
            async with _server.serve(
                self._handler, self._host, self._port,
                compression=None, max_size=None, ping_interval=None,
            ) as server:
                await server.serve_forever()
        else:
            await self._worker_loop()

    async def _worker_loop(self):
        logger.info(f"Worker loop started for rank {dist.get_rank()}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        while True:
            try:
                dist.broadcast(signal_tensor, src=0, group=self._signal_group)
                signal = signal_tensor.item()
                if signal == 1:
                    break
                elif signal == 2:
                    continue

                batch = self._receive_batch()
                dist.barrier()
                with torch.no_grad():
                    self._policy._policy.lazy_joint_forward_causal(batch)
                dist.barrier()
            except Exception as e:
                logger.error(f"Worker error: {e}")
                traceback.print_exc()
                break

    def _receive_batch(self):
        import pickle
        size_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device='cuda')
        dist.broadcast(data_tensor, src=0)
        obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address}")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        try:
            while True:
                try:
                    data = await websocket.recv()
                    obs = msgpack_numpy.unpackb(data)

                    endpoint = obs.pop("endpoint", "infer")
                    if endpoint == "reset":
                        self._policy.reset(obs)
                        await websocket.send("reset successful")
                        continue

                    t0 = time.perf_counter()
                    action = self._policy.infer(obs)
                    dt = time.perf_counter() - t0
                    logger.info(f"Inference time: {dt:.2f}s, action shape: {action.shape}")

                    await websocket.send(packer.pack(action))
                except websockets.ConnectionClosed:
                    logger.info("Connection closed")
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error.",
                    )
                    raise
        finally:
            signal_tensor.fill_(2)
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)


def main(args: Args):
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = "xdof"
    model_path = args.model_path

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ip",))

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda",
        device_mesh=device_mesh,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        parent_dir = os.path.dirname(model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        ckpt_name = os.path.basename(model_path)
        output_dir = os.path.join(parent_dir, f"robotwin_eval_{date_suffix}_{args.index}", ckpt_name)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Server on {hostname} ({local_ip}), videos → {output_dir}")
    else:
        output_dir = None

    wrapper = RoboTwinPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
    )

    metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero_robotwin",
        "model_path": model_path,
    }

    server = RoboTwinServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
        output_dir=output_dir,
        signal_group=signal_group,
    )

    if rank == 0:
        server.serve_forever(rank=0)
    else:
        asyncio.run(server._worker_loop())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)
