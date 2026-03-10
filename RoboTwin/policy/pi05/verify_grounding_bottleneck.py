"""
Verify Language Grounding Bottleneck: VLM vs Action Expert
==========================================================
Tests whether the VLM (PaliGemma) correctly encodes instruction semantics,
and whether the action expert fails to utilize them.

Experiment A: Extract prefix KV cache for different instructions,
              compare VLM's language token representations.
Experiment B: Extract action expert's attention output (context vector),
              compare across instructions.

If VLM representations differ but action expert context vectors are the same,
the bottleneck is confirmed to be in the action expert.

Usage:
    conda activate robotwin
    cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin
    CUDA_VISIBLE_DEVICES=0 python policy/pi05/verify_grounding_bottleneck.py
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = "/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "policy/pi05"))
sys.path.insert(0, os.path.join(ROOT, "policy/pi05/src"))

GEMMA_PATH = os.path.join(ROOT, "policy/pi05/src/openpi/models/gemma.py")
BACKUP_PATH = GEMMA_PATH + ".bottleneck_bak"

# ── Patch gemma.py to capture KV cache AND attention-weighted output ─────

def patch_gemma():
    """Add captures for both KV cache values and attention output."""
    shutil.copy2(GEMMA_PATH, BACKUP_PATH)

    with open(GEMMA_PATH, "r") as f:
        content = f.read()

    # Module-level capture code
    capture_header = '''
# ═══ BOTTLENECK ANALYSIS CAPTURE (injected by verify_grounding_bottleneck.py) ═══
import numpy as _capture_np
_kv_cache_captures = []       # Captures from prefix encoding (VLM representations)
_attn_output_captures = []    # Captures from denoising (action expert attention output)
_capture_enabled = True

def _kv_cache_capture_fn(k, v):
    """Capture K,V from prefix encoding (VLM expert representations)."""
    _kv_cache_captures.append({
        "k": _capture_np.array(k),  # [B, T_prefix, num_kv_heads, head_dim]
        "v": _capture_np.array(v),
    })

def _attn_output_capture_fn(attn_out):
    """Capture attention-weighted output for action expert during denoising."""
    _attn_output_captures.append({
        "attn_out": _capture_np.array(attn_out),  # [B, T_action, embed_dim]
    })
# ═══ END CAPTURE HEADER ═══
'''

    # Capture 1: KV cache during prefix encoding (kv_cache is None)
    # After: return out, (k, v)
    # We capture k, v when kv_cache is None (prefix encoding)
    kv_capture_call = '''
        # ═══ KV CACHE CAPTURE (prefix encoding) ═══
        if _capture_enabled and kv_cache is None:
            jax.debug.callback(_kv_cache_capture_fn, k, v)
        # ═══ END KV CACHE CAPTURE ═══'''

    # Capture 2: Attention-weighted output during denoising
    # After computing: encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
    # And after: out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
    # We want to capture the action expert's output (the second expert, index 1)
    attn_output_capture_call = '''
        # ═══ ATTN OUTPUT CAPTURE (denoising) ═══
        if _capture_enabled and kv_cache is not None and len(out) == 2 and out[1] is not None:
            jax.debug.callback(_attn_output_capture_fn, out[1])
        # ═══ END ATTN OUTPUT CAPTURE ═══'''

    # Insert capture header before PALIGEMMA_VOCAB_SIZE
    marker = "PALIGEMMA_VOCAB_SIZE"
    if marker not in content:
        raise RuntimeError(f"Cannot find '{marker}' in gemma.py")
    idx = content.index(marker)
    line_start = content.rfind("\n", 0, idx) + 1
    content = content[:line_start] + capture_header + "\n" + content[line_start:]

    # Insert KV cache capture before "return out, (k, v)"
    return_line = "        return out, (k, v)"
    if return_line not in content:
        raise RuntimeError("Cannot find 'return out, (k, v)' in gemma.py")
    idx = content.index(return_line)
    content = content[:idx] + kv_capture_call + "\n" + content[idx:]

    # Insert attn output capture after "return out, (k, v)" — actually before it
    # Better: capture right before the return statement, after the output loop
    # The out list is built in a loop. After the loop, we have out = [expert0_out, expert1_out]
    # We want to capture when kv_cache is not None (denoising) and expert1 output exists
    # Let's insert right before the return statement (which now has the kv_capture before it)
    # Find the return statement again (it moved due to previous insertion)
    return_line = "        return out, (k, v)"
    idx = content.index(return_line)
    content = content[:idx] + attn_output_capture_call + "\n" + content[idx:]

    with open(GEMMA_PATH, "w") as f:
        f.write(content)

    print("Patched gemma.py with bottleneck analysis captures.")


def restore_gemma():
    if os.path.exists(BACKUP_PATH):
        shutil.copy2(BACKUP_PATH, GEMMA_PATH)
        os.remove(BACKUP_PATH)
        print("Restored original gemma.py.")


# ── Model loading ────────────────────────────────────────────────────────

def load_model():
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    train_config_name = "pi05_aloha_full_base"
    model_name = "beat_block_hammer"
    checkpoint_id = 15000

    assets_path = f"policy/pi05/checkpoints/{train_config_name}/{model_name}/{checkpoint_id}/assets/"
    entries = os.listdir(assets_path)
    assets_id = entries[0]

    config = _config.get_config(train_config_name)
    policy = _policy_config.create_trained_policy(
        config,
        f"policy/pi05/checkpoints/{train_config_name}/{model_name}/{checkpoint_id}",
        robotwin_repo_id=assets_id,
    )
    print("Model loaded.")
    return policy


def create_observation(instruction):
    np.random.seed(42)
    return {
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
        },
        "prompt": instruction,
    }


# ── Analysis functions ───────────────────────────────────────────────────

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-10))


def l2_dist(a, b):
    """L2 distance between two vectors."""
    return float(np.linalg.norm(a.flatten().astype(np.float32) - b.flatten().astype(np.float32)))


def analyze_vlm_representations(all_kv_caches, instructions, num_images=3, img_tokens_per_image=256):
    """
    Experiment A: Compare VLM prefix KV cache representations across instructions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: VLM Prefix Representation Analysis")
    print("=" * 70)

    total_img_tokens = num_images * img_tokens_per_image
    n_layers = 18
    instruction_names = list(instructions.keys())
    results = {}

    for layer_idx in range(n_layers):
        layer_results = {}
        for i, name_i in enumerate(instruction_names):
            for j, name_j in enumerate(instruction_names):
                if j <= i:
                    continue
                # Compare language K/V tokens between instruction pairs
                ki = all_kv_caches[name_i][layer_idx]["k"]  # [B, T_prefix, kv_heads, head_dim]
                kj = all_kv_caches[name_j][layer_idx]["k"]
                vi = all_kv_caches[name_i][layer_idx]["v"]
                vj = all_kv_caches[name_j][layer_idx]["v"]

                # Extract language tokens only (skip image tokens)
                ki_lang = ki[0, total_img_tokens:, :, :]  # [T_lang, kv_heads, head_dim]
                kj_lang = kj[0, total_img_tokens:, :, :]
                vi_lang = vi[0, total_img_tokens:, :, :]
                vj_lang = vj[0, total_img_tokens:, :, :]

                # Also extract image tokens for comparison
                ki_img = ki[0, :total_img_tokens, :, :]
                kj_img = kj[0, :total_img_tokens, :, :]

                pair_name = f"{name_i}_vs_{name_j}"
                layer_results[pair_name] = {
                    "k_lang_cosine": cosine_sim(ki_lang, kj_lang),
                    "v_lang_cosine": cosine_sim(vi_lang, vj_lang),
                    "k_lang_l2": l2_dist(ki_lang, kj_lang),
                    "v_lang_l2": l2_dist(vi_lang, vj_lang),
                    "k_img_cosine": cosine_sim(ki_img, kj_img),  # Control: image tokens should be ~identical
                }

        results[layer_idx] = layer_results

    # Print summary
    for pair_name in results[0].keys():
        print(f"\n  Pair: {pair_name}")
        print(f"  {'Layer':>5s}  {'K_lang_cos':>10s}  {'V_lang_cos':>10s}  {'K_lang_L2':>10s}  {'V_lang_L2':>10s}  {'K_img_cos':>10s}")
        for layer_idx in range(n_layers):
            r = results[layer_idx][pair_name]
            print(f"  {layer_idx:5d}  {r['k_lang_cosine']:10.6f}  {r['v_lang_cosine']:10.6f}  "
                  f"{r['k_lang_l2']:10.2f}  {r['v_lang_l2']:10.2f}  {r['k_img_cosine']:10.6f}")

    return results


def analyze_action_expert_output(all_attn_outputs, instructions):
    """
    Experiment B: Compare action expert's attention-weighted output across instructions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Action Expert Attention Output Analysis")
    print("=" * 70)

    n_layers = 18
    instruction_names = list(instructions.keys())
    results = {}

    for layer_idx in range(n_layers):
        layer_results = {}
        for i, name_i in enumerate(instruction_names):
            for j, name_j in enumerate(instruction_names):
                if j <= i:
                    continue
                oi = all_attn_outputs[name_i][layer_idx]["attn_out"]  # [B, T_action, embed_dim]
                oj = all_attn_outputs[name_j][layer_idx]["attn_out"]

                pair_name = f"{name_i}_vs_{name_j}"
                layer_results[pair_name] = {
                    "attn_out_cosine": cosine_sim(oi, oj),
                    "attn_out_l2": l2_dist(oi, oj),
                }
        results[layer_idx] = layer_results

    # Print summary
    for pair_name in results[0].keys():
        print(f"\n  Pair: {pair_name}")
        print(f"  {'Layer':>5s}  {'AttnOut_cos':>12s}  {'AttnOut_L2':>12s}")
        for layer_idx in range(n_layers):
            r = results[layer_idx][pair_name]
            print(f"  {layer_idx:5d}  {r['attn_out_cosine']:12.6f}  {r['attn_out_l2']:12.4f}")

    return results


def linear_probe_vlm(all_kv_caches, instructions, num_images=3, img_tokens_per_image=256):
    """
    Train linear probes to classify instruction type from:
    1. VLM's language K representations (per layer)
    2. VLM's language V representations (per layer)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Linear Probe on VLM Representations")
    print("=" * 70)

    total_img_tokens = num_images * img_tokens_per_image
    n_layers = 18
    instruction_names = list(instructions.keys())
    n_classes = len(instruction_names)

    # We only have 1 sample per class (single inference per instruction)
    # So we'll use per-token classification instead: each language token as a sample
    # This gives us ~200 samples per class

    results = {}

    for layer_idx in range(n_layers):
        X_k, X_v, y = [], [], []

        for cls_idx, name in enumerate(instruction_names):
            k = all_kv_caches[name][layer_idx]["k"]  # [B, T_prefix, kv_heads, head_dim]
            v = all_kv_caches[name][layer_idx]["v"]

            # Language tokens
            k_lang = k[0, total_img_tokens:, :, :].reshape(k.shape[1] - total_img_tokens, -1)
            v_lang = v[0, total_img_tokens:, :, :].reshape(v.shape[1] - total_img_tokens, -1)

            X_k.append(k_lang.astype(np.float32))
            X_v.append(v_lang.astype(np.float32))
            y.extend([cls_idx] * k_lang.shape[0])

        X_k = np.concatenate(X_k, axis=0)
        X_v = np.concatenate(X_v, axis=0)
        y = np.array(y)

        # Train/test on same data (we're measuring linear separability, not generalization)
        try:
            clf_k = LogisticRegression(max_iter=1000, C=1.0).fit(X_k, y)
            acc_k = accuracy_score(y, clf_k.predict(X_k))
        except Exception:
            acc_k = 1.0 / n_classes

        try:
            clf_v = LogisticRegression(max_iter=1000, C=1.0).fit(X_v, y)
            acc_v = accuracy_score(y, clf_v.predict(X_v))
        except Exception:
            acc_v = 1.0 / n_classes

        results[layer_idx] = {"k_probe_acc": float(acc_k), "v_probe_acc": float(acc_v)}

    print(f"\n  {'Layer':>5s}  {'K_probe_acc':>12s}  {'V_probe_acc':>12s}  {'Interpretation':>30s}")
    for layer_idx in range(n_layers):
        r = results[layer_idx]
        interp = "VLM differentiates!" if r['k_probe_acc'] > 0.8 else "Low separability"
        print(f"  {layer_idx:5d}  {r['k_probe_acc']:12.4f}  {r['v_probe_acc']:12.4f}  {interp:>30s}")

    return results


def linear_probe_action_expert(all_attn_outputs, instructions):
    """
    Train a linear probe to classify instruction type from the action expert's
    attention output at each layer.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Linear Probe on Action Expert Outputs")
    print("=" * 70)

    n_layers = 18
    instruction_names = list(instructions.keys())
    n_classes = len(instruction_names)
    results = {}

    for layer_idx in range(n_layers):
        X, y = [], []

        for cls_idx, name in enumerate(instruction_names):
            attn_out = all_attn_outputs[name][layer_idx]["attn_out"]  # [B, T_action, embed_dim]
            token_features = attn_out[0].reshape(attn_out.shape[1], -1).astype(np.float32)
            X.append(token_features)
            y.extend([cls_idx] * token_features.shape[0])

        X = np.concatenate(X, axis=0)
        y = np.array(y)

        try:
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
            acc = accuracy_score(y, clf.predict(X))
        except Exception:
            acc = 1.0 / n_classes

        results[layer_idx] = {"expert_probe_acc": float(acc)}

    print(f"\n  {'Layer':>5s}  {'Expert_probe_acc':>16s}  {'Interpretation':>30s}")
    for layer_idx in range(n_layers):
        r = results[layer_idx]
        interp = "Expert preserves signal" if r["expert_probe_acc"] > 0.8 else "Weak separability"
        print(f"  {layer_idx:5d}  {r['expert_probe_acc']:16.4f}  {interp:>30s}")

    return results


def plot_results(vlm_results, expert_results, probe_results, expert_probe_results, save_dir):
    """Create comprehensive visualization of the bottleneck analysis."""
    n_layers = 18

    # Get pair names
    pair_names = list(vlm_results[0].keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Language Grounding Bottleneck Analysis: VLM vs Action Expert",
                 fontsize=14, fontweight="bold")

    # ── Plot 1: VLM K cosine similarity (language tokens) ──
    ax = axes[0][0]
    for pair in pair_names:
        vals = [vlm_results[l][pair]["k_lang_cosine"] for l in range(n_layers)]
        ax.plot(range(n_layers), vals, "o-", label=f"K lang: {pair}", markersize=4)
    # Add image control
    for pair in pair_names:
        vals = [vlm_results[l][pair]["k_img_cosine"] for l in range(n_layers)]
        ax.plot(range(n_layers), vals, "s--", alpha=0.3, label=f"K img: {pair}", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("VLM KV Cache: K Similarity Across Instructions")
    ax.legend(fontsize=7)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

    # ── Plot 2: VLM V cosine similarity (language tokens) ──
    ax = axes[0][1]
    for pair in pair_names:
        vals = [vlm_results[l][pair]["v_lang_cosine"] for l in range(n_layers)]
        ax.plot(range(n_layers), vals, "o-", label=f"V lang: {pair}", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("VLM KV Cache: V Similarity Across Instructions")
    ax.legend(fontsize=7)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

    # ── Plot 3: Action Expert attention output cosine similarity ──
    ax = axes[1][0]
    for pair in pair_names:
        vals = [expert_results[l][pair]["attn_out_cosine"] for l in range(n_layers)]
        ax.plot(range(n_layers), vals, "o-", label=pair, markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Action Expert: Attention Output Similarity")
    ax.legend(fontsize=7)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

    # ── Plot 4: Linear probe accuracy ──
    ax = axes[1][1]
    k_accs = [probe_results[l]["k_probe_acc"] for l in range(n_layers)]
    v_accs = [probe_results[l]["v_probe_acc"] for l in range(n_layers)]
    expert_accs = [expert_probe_results[l]["expert_probe_acc"] for l in range(n_layers)]
    ax.plot(range(n_layers), k_accs, "o-", label="K probe accuracy", color="blue", markersize=4)
    ax.plot(range(n_layers), v_accs, "s-", label="V probe accuracy", color="red", markersize=4)
    ax.plot(range(n_layers), expert_accs, "D-", label="Action expert probe", color="orange", markersize=4)
    ax.axhline(y=1.0/3, color="gray", linestyle="--", alpha=0.5, label="Chance (33.3%)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Linear Probe: VLM vs Action Expert")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "bottleneck_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Summary figure: VLM vs Action Expert ──
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    fig2.suptitle("Where Does Language Grounding Break?", fontsize=14, fontweight="bold")

    # Average across pairs
    vlm_k_avg = [np.mean([vlm_results[l][p]["k_lang_cosine"] for p in pair_names]) for l in range(n_layers)]
    vlm_v_avg = [np.mean([vlm_results[l][p]["v_lang_cosine"] for p in pair_names]) for l in range(n_layers)]
    expert_avg = [np.mean([expert_results[l][p]["attn_out_cosine"] for p in pair_names]) for l in range(n_layers)]

    ax2.plot(range(n_layers), vlm_k_avg, "o-", label="VLM K (language tokens)", color="blue", linewidth=2, markersize=5)
    ax2.plot(range(n_layers), vlm_v_avg, "s-", label="VLM V (language tokens)", color="green", linewidth=2, markersize=5)
    ax2.plot(range(n_layers), expert_avg, "D-", label="Action Expert output", color="red", linewidth=2, markersize=5)
    ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

    ax2.fill_between(range(n_layers), vlm_k_avg, 1.0, alpha=0.1, color="blue")
    ax2.fill_between(range(n_layers), expert_avg, 1.0, alpha=0.1, color="red")

    ax2.set_xlabel("Transformer Layer", fontsize=12)
    ax2.set_ylabel("Cosine Similarity Between Different Instructions\n(lower = more differentiation)", fontsize=11)
    ax2.set_title("Cross-Instruction Similarity: VLM Representations vs Action Expert Output")
    ax2.legend(fontsize=10, loc="lower left")
    ax2.set_ylim(0.5, 1.02)

    # Annotate
    ax2.annotate("VLM differentiates\ninstructions here\n↓",
                 xy=(5, min(vlm_k_avg[3:8])), fontsize=9, ha="center",
                 color="blue", fontweight="bold")
    ax2.annotate("Action expert\ncollapses differences\n↓",
                 xy=(13, max(expert_avg[10:16])), fontsize=9, ha="center",
                 color="red", fontweight="bold")

    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, "bottleneck_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nPlots saved to {save_dir}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    save_dir = os.path.join(ROOT, "reports/attention_analysis")
    os.makedirs(save_dir, exist_ok=True)

    patch_gemma()

    try:
        import openpi.models.gemma as gemma_module
        print(f"Capture enabled: {gemma_module._capture_enabled}")

        print("\nLoading model...")
        policy = load_model()

        instructions = {
            "seen": "Catch the plastic handle metal hammer and use it on the block.",
            "do_nothing": "Stay still. Do not move the arms.",
            "wrong_action": "Place the hammer gently next to the block.",
        }

        all_kv_caches = {}    # {instr_name: [layer0, layer1, ...]}
        all_attn_outputs = {} # {instr_name: [layer0, layer1, ...]}

        for itype, instruction in instructions.items():
            print(f"\n{'='*60}")
            print(f"Running: {itype} — \"{instruction}\"")
            print(f"{'='*60}")

            gemma_module._kv_cache_captures.clear()
            gemma_module._attn_output_captures.clear()
            gemma_module._capture_enabled = True

            obs = create_observation(instruction)
            result = policy.infer(obs)

            gemma_module._capture_enabled = False

            n_kv = len(gemma_module._kv_cache_captures)
            n_attn = len(gemma_module._attn_output_captures)
            print(f"  KV cache captures: {n_kv} (expect 18 = prefix encoding layers)")
            print(f"  Attn output captures: {n_attn} (expect 18 * num_steps = denoising)")

            # KV cache: 18 captures from prefix encoding
            all_kv_caches[itype] = list(gemma_module._kv_cache_captures)[:18]

            # Attn output: first 18 captures = first denoising step
            all_attn_outputs[itype] = list(gemma_module._attn_output_captures)[:18]

            if n_kv > 0:
                k0 = gemma_module._kv_cache_captures[0]["k"]
                print(f"  KV cache K shape: {k0.shape}")
            if n_attn > 0:
                a0 = gemma_module._attn_output_captures[0]["attn_out"]
                print(f"  Attn output shape: {a0.shape}")

        # Verify we have data
        for itype in instructions:
            assert len(all_kv_caches[itype]) == 18, f"Missing KV cache for {itype}: got {len(all_kv_caches[itype])}"
            assert len(all_attn_outputs[itype]) >= 18, f"Missing attn output for {itype}: got {len(all_attn_outputs[itype])}"

        # Run analyses
        vlm_results = analyze_vlm_representations(all_kv_caches, instructions)
        expert_results = analyze_action_expert_output(all_attn_outputs, instructions)
        probe_results = linear_probe_vlm(all_kv_caches, instructions)
        expert_probe_results = linear_probe_action_expert(all_attn_outputs, instructions)

        # Plot
        plot_results(vlm_results, expert_results, probe_results, expert_probe_results, save_dir)

        # Save JSON
        summary = {
            "experiment": "Language Grounding Bottleneck Verification",
            "instructions": instructions,
            "vlm_representation_similarity": {},
            "action_expert_output_similarity": {},
            "linear_probe_accuracy": {},
            "action_expert_probe_accuracy": {},
        }

        for layer_idx in range(18):
            summary["vlm_representation_similarity"][str(layer_idx)] = {
                pair: {
                    "k_lang_cosine": vlm_results[layer_idx][pair]["k_lang_cosine"],
                    "v_lang_cosine": vlm_results[layer_idx][pair]["v_lang_cosine"],
                } for pair in vlm_results[layer_idx]
            }
            summary["action_expert_output_similarity"][str(layer_idx)] = {
                pair: {
                    "attn_out_cosine": expert_results[layer_idx][pair]["attn_out_cosine"],
                } for pair in expert_results[layer_idx]
            }
            summary["linear_probe_accuracy"][str(layer_idx)] = probe_results[layer_idx]
            summary["action_expert_probe_accuracy"][str(layer_idx)] = expert_probe_results[layer_idx]

        # Compute final verdict
        avg_vlm_k_sim = np.mean([
            vlm_results[l][p]["k_lang_cosine"]
            for l in range(18) for p in vlm_results[l]
        ])
        avg_vlm_v_sim = np.mean([
            vlm_results[l][p]["v_lang_cosine"]
            for l in range(18) for p in vlm_results[l]
        ])
        avg_expert_sim = np.mean([
            expert_results[l][p]["attn_out_cosine"]
            for l in range(18) for p in expert_results[l]
        ])
        avg_probe_k = np.mean([probe_results[l]["k_probe_acc"] for l in range(18)])
        avg_probe_v = np.mean([probe_results[l]["v_probe_acc"] for l in range(18)])
        avg_probe_expert = np.mean([expert_probe_results[l]["expert_probe_acc"] for l in range(18)])

        summary["verdict"] = {
            "avg_vlm_k_similarity": float(avg_vlm_k_sim),
            "avg_vlm_v_similarity": float(avg_vlm_v_sim),
            "avg_expert_output_similarity": float(avg_expert_sim),
            "avg_k_probe_accuracy": float(avg_probe_k),
            "avg_v_probe_accuracy": float(avg_probe_v),
            "avg_expert_probe_accuracy": float(avg_probe_expert),
            "vlm_differentiates": bool(avg_probe_k > 0.5),
            "expert_linearly_separable": bool(avg_probe_expert > 0.5),
            "expert_collapses": bool(avg_expert_sim > avg_vlm_k_sim),
            "bottleneck": "action_expert" if (avg_probe_k > 0.5 and avg_expert_sim > avg_vlm_k_sim) else "vlm_or_both",
        }

        with open(os.path.join(save_dir, "bottleneck_analysis.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Print final verdict
        print("\n" + "=" * 70)
        print("FINAL VERDICT")
        print("=" * 70)
        print(f"  VLM K similarity (avg):          {avg_vlm_k_sim:.6f}")
        print(f"  VLM V similarity (avg):          {avg_vlm_v_sim:.6f}")
        print(f"  Action Expert output similarity:  {avg_expert_sim:.6f}")
        print(f"  K probe accuracy (avg):           {avg_probe_k:.4f}")
        print(f"  V probe accuracy (avg):           {avg_probe_v:.4f}")
        print(f"  Expert probe accuracy (avg):      {avg_probe_expert:.4f}")
        print()
        if avg_probe_k > 0.5:
            print("  ✓ VLM DOES differentiate instructions (K probe > chance)")
        else:
            print("  ✗ VLM does NOT differentiate instructions")
        if avg_probe_expert > 0.5:
            print("  ✓ Action Expert remains linearly separable")
        else:
            print("  ✗ Action Expert is near chance under linear probe")
        if avg_expert_sim > avg_vlm_k_sim:
            print("  ✓ Action Expert COLLAPSES instruction differences")
            print("    (output similarity > VLM representation similarity)")
        else:
            print("  ✗ Action Expert preserves VLM differentiation")
        print(f"\n  BOTTLENECK: {summary['verdict']['bottleneck'].upper()}")

    finally:
        restore_gemma()


if __name__ == "__main__":
    main()
