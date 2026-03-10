"""
Verify Language Grounding Bottleneck for LingBot-VA
===================================================

Reproduces the Pi0.5-style bottleneck analysis for LingBot-VA by comparing:
1. Text encoder hidden states across instructions.
2. Action-side transformer block outputs during the first action denoising step.

This script does not patch source files on disk. It monkey-patches the
transformer block forward in memory to capture per-layer action representations.
"""

import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
LINGBOT_ROOT = ROOT / "policy/LingBotVA/lingbot-va"
SAVE_DIR = EXP_DIR / "outputs/lingbotva_bottleneck"

sys.path.insert(0, str(LINGBOT_ROOT / "wan_va"))
sys.path.insert(0, str(LINGBOT_ROOT))

from configs import VA_CONFIGS
from distributed.util import init_distributed
from einops import rearrange
import wan_va_server as server_module
from wan_va_server import VA_Server
import modules.model as model_module


_CAPTURE = {
    "enabled": False,
    "outputs": [],
}
_ORIG_BLOCK_FORWARD = model_module.WanTransformerBlock.forward


def _capture_block_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    rotary_emb,
    update_cache=0,
    cache_name="pos",
):
    out = _ORIG_BLOCK_FORWARD(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=update_cache,
        cache_name=cache_name,
    )
    if _CAPTURE["enabled"]:
        _CAPTURE["outputs"].append(out.detach().float().cpu().numpy())
    return out


model_module.WanTransformerBlock.forward = _capture_block_forward


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dist_env():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29661")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def cosine_sim(a, b):
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-10
    return float(np.dot(a_flat, b_flat) / denom)


def l2_dist(a, b):
    return float(np.linalg.norm(a.flatten().astype(np.float32) - b.flatten().astype(np.float32)))


def load_model():
    ensure_dist_env()
    init_distributed(
        int(os.environ["WORLD_SIZE"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["RANK"]),
    )
    config = VA_CONFIGS["robotwin"]
    config.rank = int(os.environ["RANK"])
    config.local_rank = int(os.environ["LOCAL_RANK"])
    config.world_size = int(os.environ["WORLD_SIZE"])
    config.save_root = str(SAVE_DIR / "raw")
    return VA_Server(config)


def create_observation(job_config):
    example_dir = LINGBOT_ROOT / "example/robotwin"
    frame = {}
    for key in job_config.obs_cam_keys:
        img_path = example_dir / f"{key}.png"
        frame[key] = np.array(Image.open(img_path).convert("RGB"))
    return {"obs": [frame]}


def extract_text_hidden_states(server, prompt, max_sequence_length=512):
    prompt = [prompt]
    text_inputs = server.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    seq_len = int(mask[0].sum().item())
    text_encoder_device = next(server.text_encoder.parameters()).device
    outputs = server.text_encoder(
        input_ids.to(text_encoder_device),
        mask.to(text_encoder_device),
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states
    layers = []
    for hs in hidden_states:
        layers.append(hs[0, :seq_len].detach().float().cpu().numpy())
    return layers


def capture_action_block_outputs(server, obs):
    frame_chunk_size = server.job_config.frame_chunk_size
    server.frame_st_id = 0
    init_latent = server._encode_obs(obs)
    server.init_latent = init_latent

    latents = torch.randn(
        1,
        48,
        frame_chunk_size,
        server.latent_height,
        server.latent_width,
        device=server.device,
        dtype=server.dtype,
    )
    actions = torch.randn(
        1,
        server.job_config.action_dim,
        frame_chunk_size,
        server.action_per_frame,
        1,
        device=server.device,
        dtype=server.dtype,
    )

    server.scheduler.set_timesteps(server.job_config.num_inference_steps)
    server.action_scheduler.set_timesteps(server.job_config.action_num_inference_steps)

    timesteps = torch.nn.functional.pad(server.scheduler.timesteps, (0, 1), mode="constant", value=0)
    action_timesteps = torch.nn.functional.pad(
        server.action_scheduler.timesteps,
        (0, 1),
        mode="constant",
        value=0,
    )

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            latent_cond = init_latent[:, :, 0:1].to(server.dtype)
            input_dict = server._prepare_latent_input(
                latents,
                None,
                t,
                t,
                latent_cond,
                None,
                frame_st_id=0,
            )
            video_noise_pred = server.transformer(
                server._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=server.cache_name,
                action_mode=False,
            )
            if not last_step or server.job_config.video_exec_step != -1:
                video_noise_pred = server_module.data_seq_to_patch(
                    server.job_config.patch_size,
                    video_noise_pred,
                    frame_chunk_size,
                    server.latent_height,
                    server.latent_width,
                    batch_size=2 if server.use_cfg else 1,
                )
                if server.job_config.guidance_scale > 1:
                    video_noise_pred = video_noise_pred[1:] + server.job_config.guidance_scale * (
                        video_noise_pred[:1] - video_noise_pred[1:]
                    )
                else:
                    video_noise_pred = video_noise_pred[:1]
                latents = server.scheduler.step(video_noise_pred, t, latents, return_dict=False)
            latents[:, :, 0:1] = latent_cond

        captured_layers = None
        for i, t in enumerate(action_timesteps):
            last_step = i == len(action_timesteps) - 1
            action_cond = torch.zeros(
                [1, server.job_config.action_dim, 1, server.action_per_frame, 1],
                device=server.device,
                dtype=server.dtype,
            )
            input_dict = server._prepare_latent_input(
                None,
                actions,
                t,
                t,
                None,
                action_cond,
                frame_st_id=0,
            )
            _CAPTURE["enabled"] = i == 0
            if i == 0:
                _CAPTURE["outputs"].clear()
            action_noise_pred = server.transformer(
                server._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=server.cache_name,
                action_mode=True,
            )
            _CAPTURE["enabled"] = False
            if i == 0:
                captured_layers = [x[0] for x in _CAPTURE["outputs"]]

            if not last_step:
                action_noise_pred = rearrange(action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size)
                if server.job_config.action_guidance_scale > 1:
                    action_noise_pred = action_noise_pred[1:] + server.job_config.action_guidance_scale * (
                        action_noise_pred[:1] - action_noise_pred[1:]
                    )
                else:
                    action_noise_pred = action_noise_pred[:1]
                actions = server.action_scheduler.step(action_noise_pred, t, actions, return_dict=False)
            actions[:, :, 0:1] = action_cond

    if captured_layers is None:
        raise RuntimeError("Failed to capture LingBot-VA action block outputs.")
    return captured_layers


def analyze_pairwise(all_layers, instructions, title, key_prefix):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    n_layers = len(next(iter(all_layers.values())))
    names = list(instructions.keys())
    results = {}

    for layer_idx in range(n_layers):
        layer_results = {}
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if j <= i:
                    continue
                xi = all_layers[name_i][layer_idx]
                xj = all_layers[name_j][layer_idx]
                if xi.shape != xj.shape:
                    min_len = min(xi.shape[0], xj.shape[0])
                    xi = xi[:min_len]
                    xj = xj[:min_len]
                pair_name = f"{name_i}_vs_{name_j}"
                layer_results[pair_name] = {
                    f"{key_prefix}_cosine": cosine_sim(xi, xj),
                    f"{key_prefix}_l2": l2_dist(xi, xj),
                }
        results[layer_idx] = layer_results

    pair_names = list(results[0].keys())
    for pair_name in pair_names:
        print(f"\n  Pair: {pair_name}")
        print(f"  {'Layer':>5s}  {'Cosine':>10s}  {'L2':>12s}")
        for layer_idx in range(n_layers):
            r = results[layer_idx][pair_name]
            print(f"  {layer_idx:5d}  {r[f'{key_prefix}_cosine']:10.6f}  {r[f'{key_prefix}_l2']:12.4f}")
    return results


def linear_probe(all_layers, instructions, title, result_key):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    n_layers = len(next(iter(all_layers.values())))
    names = list(instructions.keys())
    n_classes = len(names)
    results = {}

    for layer_idx in range(n_layers):
        X, y = [], []
        for cls_idx, name in enumerate(names):
            features = all_layers[name][layer_idx]
            if features.ndim == 3:
                features = features[0]
            X.append(features.reshape(features.shape[0], -1).astype(np.float32))
            y.extend([cls_idx] * features.shape[0])
        X = np.concatenate(X, axis=0)
        y = np.array(y)

        try:
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
            acc = accuracy_score(y, clf.predict(X))
        except Exception:
            acc = 1.0 / n_classes
        results[layer_idx] = {result_key: float(acc)}

    print(f"\n  {'Layer':>5s}  {result_key:>18s}")
    for layer_idx in range(n_layers):
        print(f"  {layer_idx:5d}  {results[layer_idx][result_key]:18.4f}")
    return results


def plot_results(text_results, action_results, text_probe_results, action_probe_results):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    text_layers = len(text_probe_results)
    action_layers = len(action_probe_results)
    text_pairs = list(text_results[0].keys())
    action_pairs = list(action_results[0].keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("LingBot-VA Grounding Bottleneck Analysis", fontsize=14, fontweight="bold")

    ax = axes[0][0]
    for pair in text_pairs:
        vals = [text_results[l][pair]["text_cosine"] for l in range(text_layers)]
        ax.plot(range(text_layers), vals, "o-", label=pair, markersize=3)
    ax.set_title("Text Encoder Hidden State Similarity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.legend(fontsize=7)
    ax.set_ylim(0.0, 1.02)

    ax = axes[0][1]
    for pair in action_pairs:
        vals = [action_results[l][pair]["action_cosine"] for l in range(action_layers)]
        ax.plot(range(action_layers), vals, "o-", label=pair, markersize=3)
    ax.set_title("Action Transformer Block Output Similarity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.legend(fontsize=7)
    ax.set_ylim(0.0, 1.02)

    ax = axes[1][0]
    vals = [text_probe_results[l]["text_probe_acc"] for l in range(text_layers)]
    ax.plot(range(text_layers), vals, "o-", color="blue", label="Text probe")
    ax.axhline(y=1.0 / 3, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_title("Text Encoder Linear Probe")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, 1.05)

    ax = axes[1][1]
    vals = [action_probe_results[l]["action_probe_acc"] for l in range(action_layers)]
    ax.plot(range(action_layers), vals, "o-", color="darkorange", label="Action probe")
    ax.axhline(y=1.0 / 3, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_title("Action Transformer Linear Probe")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    fig.savefig(SAVE_DIR / "lingbotva_bottleneck_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.chdir(ROOT)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading LingBot-VA...")
    server = load_model()
    obs = create_observation(server.job_config)

    instructions = {
        "seen": "Catch the plastic handle metal hammer and use it on the block.",
        "do_nothing": "Stay still. Do not move the arms.",
        "wrong_action": "Place the hammer gently next to the block.",
    }

    all_text_layers = {}
    all_action_layers = {}

    for name, prompt in instructions.items():
        print(f"\n{'=' * 60}")
        print(f"Running: {name} — \"{prompt}\"")
        print(f"{'=' * 60}")

        set_seed(0)
        server._reset(prompt=prompt)
        text_layers = extract_text_hidden_states(server, prompt)
        action_layers = capture_action_block_outputs(server, obs)
        all_text_layers[name] = text_layers
        all_action_layers[name] = action_layers

        print(f"  Text layers captured:   {len(text_layers)}")
        print(f"  Action layers captured: {len(action_layers)}")
        print(f"  Text token shape:       {text_layers[0].shape}")
        print(f"  Action token shape:     {action_layers[0].shape}")

    text_results = analyze_pairwise(
        all_text_layers,
        instructions,
        title="EXPERIMENT A: Text Encoder Representation Analysis",
        key_prefix="text",
    )
    action_results = analyze_pairwise(
        all_action_layers,
        instructions,
        title="EXPERIMENT B: Action Transformer Output Analysis",
        key_prefix="action",
    )
    text_probe_results = linear_probe(
        all_text_layers,
        instructions,
        title="EXPERIMENT C: Linear Probe on Text Encoder Hidden States",
        result_key="text_probe_acc",
    )
    action_probe_results = linear_probe(
        all_action_layers,
        instructions,
        title="EXPERIMENT D: Linear Probe on Action Transformer Outputs",
        result_key="action_probe_acc",
    )

    plot_results(text_results, action_results, text_probe_results, action_probe_results)

    summary = {
        "experiment": "LingBot-VA Language Grounding Bottleneck Verification",
        "instructions": instructions,
        "text_representation_similarity": {},
        "action_output_similarity": {},
        "text_probe_accuracy": {},
        "action_probe_accuracy": {},
    }

    for layer_idx in range(len(text_results)):
        summary["text_representation_similarity"][str(layer_idx)] = text_results[layer_idx]
        summary["text_probe_accuracy"][str(layer_idx)] = text_probe_results[layer_idx]
    for layer_idx in range(len(action_results)):
        summary["action_output_similarity"][str(layer_idx)] = action_results[layer_idx]
        summary["action_probe_accuracy"][str(layer_idx)] = action_probe_results[layer_idx]

    avg_text_sim = np.mean([
        text_results[l][p]["text_cosine"]
        for l in range(len(text_results))
        for p in text_results[l]
    ])
    avg_action_sim = np.mean([
        action_results[l][p]["action_cosine"]
        for l in range(len(action_results))
        for p in action_results[l]
    ])
    avg_text_probe = np.mean([text_probe_results[l]["text_probe_acc"] for l in range(len(text_probe_results))])
    avg_action_probe = np.mean([action_probe_results[l]["action_probe_acc"] for l in range(len(action_probe_results))])

    summary["verdict"] = {
        "avg_text_similarity": float(avg_text_sim),
        "avg_action_similarity": float(avg_action_sim),
        "avg_text_probe_accuracy": float(avg_text_probe),
        "avg_action_probe_accuracy": float(avg_action_probe),
        "text_differentiates": bool(avg_text_probe > 0.5),
        "action_collapses": bool(avg_action_sim > avg_text_sim),
        "bottleneck": "action_transformer" if (avg_text_probe > 0.5 and avg_action_sim > avg_text_sim) else "text_or_both",
    }

    with open(SAVE_DIR / "lingbotva_bottleneck_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print(f"  Text similarity (avg):        {avg_text_sim:.6f}")
    print(f"  Action similarity (avg):      {avg_action_sim:.6f}")
    print(f"  Text probe accuracy (avg):    {avg_text_probe:.4f}")
    print(f"  Action probe accuracy (avg):  {avg_action_probe:.4f}")
    print(f"\n  BOTTLENECK: {summary['verdict']['bottleneck'].upper()}")


if __name__ == "__main__":
    main()
