"""
Pi0.5 Attention Weight Visualization
=====================================
Extracts and visualizes how the action expert's attention is distributed
across image tokens vs language tokens in the prefix.

Strategy:
  - Temporarily patches gemma.py to add jax.debug.callback after softmax
  - jax.debug.callback works correctly inside nn.scan and lax.while_loop
  - Only captures denoising step attention (kv_cache is not None)
  - Restores gemma.py in a finally block

Usage (on GPU node):
    conda activate robotwin
    cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin
    CUDA_VISIBLE_DEVICES=0 python policy/pi05/visualize_attention.py
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup paths
ROOT = "/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "policy/pi05"))
sys.path.insert(0, os.path.join(ROOT, "policy/pi05/src"))

GEMMA_PATH = os.path.join(ROOT, "policy/pi05/src/openpi/models/gemma.py")
BACKUP_PATH = GEMMA_PATH + ".attn_viz_bak"

# ── Step 1: Patch gemma.py BEFORE importing openpi ──────────────────────

def patch_gemma():
    """Add jax.debug.callback to gemma.py for attention capture."""
    shutil.copy2(GEMMA_PATH, BACKUP_PATH)

    with open(GEMMA_PATH, "r") as f:
        content = f.read()

    # Module-level capture code (add after imports)
    capture_header = '''
# ═══ ATTENTION CAPTURE (injected by visualize_attention.py) ═══
import numpy as _capture_np
_attn_captures = []
_attn_capture_enabled = True

def _attn_capture_fn(probs, kv_cache_len):
    _attn_captures.append({
        "probs": _capture_np.array(probs),
        "kv_cache_len": int(kv_cache_len),
    })
# ═══ END CAPTURE HEADER ═══
'''

    # Capture call (add right after softmax)
    capture_call = '''
        # ═══ ATTENTION CAPTURE CALL ═══
        if _attn_capture_enabled and kv_cache is not None:
            jax.debug.callback(_attn_capture_fn, probs, jnp.array(cache_k.shape[1]))
        # ═══ END CAPTURE CALL ═══'''

    # Find insertion point for header: after the last top-level import
    # We'll insert before the first class/function definition
    marker = "PALIGEMMA_VOCAB_SIZE"
    if marker not in content:
        raise RuntimeError(f"Cannot find '{marker}' in gemma.py")
    idx = content.index(marker)
    # Go back to start of line
    line_start = content.rfind("\n", 0, idx) + 1
    content = content[:line_start] + capture_header + "\n" + content[line_start:]

    # Find softmax line and add capture call after it
    softmax_line = "probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)"
    if softmax_line not in content:
        raise RuntimeError("Cannot find softmax line in gemma.py")
    idx = content.index(softmax_line)
    end_of_line = content.index("\n", idx)
    content = content[:end_of_line + 1] + capture_call + "\n" + content[end_of_line + 1:]

    with open(GEMMA_PATH, "w") as f:
        f.write(content)

    print("Patched gemma.py with attention capture code.")


def restore_gemma():
    """Restore original gemma.py from backup."""
    if os.path.exists(BACKUP_PATH):
        shutil.copy2(BACKUP_PATH, GEMMA_PATH)
        os.remove(BACKUP_PATH)
        print("Restored original gemma.py.")


# ── Step 2: Model loading and inference ──────────────────────────────────

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
    print("Model loaded successfully!")
    return policy


def create_observation(instruction="Grab the hammer and beat the block."):
    """Create observation with random images (for attention pattern analysis)."""
    np.random.seed(42)  # Reproducible
    return {
        "state": np.zeros(14, dtype=np.float32),  # 7 joints x 2 arms (ALOHA)
        "images": {
            "cam_high": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
        },
        "prompt": instruction,
    }


def run_inference_and_capture(policy, gemma_module, instruction):
    """Run inference and capture attention weights via the patched gemma module."""
    gemma_module._attn_captures.clear()
    gemma_module._attn_capture_enabled = True

    obs = create_observation(instruction)
    result = policy.infer(obs)

    captures = list(gemma_module._attn_captures)
    gemma_module._attn_capture_enabled = False
    print(f"  Captured {len(captures)} attention layer calls")
    return captures, result


# ── Step 3: Analysis ─────────────────────────────────────────────────────

def analyze_attention(captures, num_images=3, image_tokens_per_image=256):
    """Analyze attention distribution from first denoising step."""
    if not captures:
        print("ERROR: No captures!")
        return None

    # All captures are from denoising steps (kv_cache is not None)
    # Group into steps of 18 layers each
    n_layers = 18
    n_steps = len(captures) // n_layers
    print(f"  {len(captures)} total captures = {n_steps} denoising steps x {n_layers} layers")

    # Take first denoising step only
    first_step = captures[:n_layers]

    # Get dimensions from first capture
    probs0 = first_step[0]["probs"]  # [B, K, G, T_q, S_k]
    kv_cache_len = first_step[0]["kv_cache_len"]
    T_q = probs0.shape[-2]  # query = action tokens
    S_k = probs0.shape[-1]  # key = prefix + suffix

    print(f"  Probs shape: {probs0.shape}")
    print(f"  KV cache (prefix) length: {kv_cache_len}")
    print(f"  Action tokens (query): {T_q}")
    print(f"  Total key length: {S_k}")

    # Token boundaries
    total_image_tokens = num_images * image_tokens_per_image
    language_tokens = kv_cache_len - total_image_tokens
    suffix_start = kv_cache_len

    if language_tokens < 0:
        print(f"  WARNING: Computed negative language tokens ({language_tokens}).")
        print(f"  Adjusting image_tokens_per_image assumption...")
        # Try to infer: kv_cache_len = n_images * img_tokens + lang_tokens
        # If lang_tokens is ~200 (max for pi05), img_tokens = (kv_cache_len - 200) / 3
        estimated_img_tokens = (kv_cache_len - 200) // num_images
        total_image_tokens = num_images * estimated_img_tokens
        language_tokens = kv_cache_len - total_image_tokens
        image_tokens_per_image = estimated_img_tokens
        print(f"  Estimated: {image_tokens_per_image} image tokens/image, {language_tokens} language tokens")

    print(f"  Image tokens: {total_image_tokens} ({num_images} x {image_tokens_per_image})")
    print(f"  Language tokens: {language_tokens}")
    print(f"  Self-attention tokens: {S_k - kv_cache_len}")

    results = {
        "per_layer": {},
        "kv_cache_len": kv_cache_len,
        "total_image_tokens": total_image_tokens,
        "language_tokens": language_tokens,
        "action_len": T_q,
        "image_tokens_per_image": image_tokens_per_image,
    }

    for layer_idx in range(min(n_layers, len(first_step))):
        probs = first_step[layer_idx]["probs"]  # [B, K, G, T_q, S_k]
        # Average over batch and heads: [B=1, K, G, T_q, S_k] → [T_q, S_k]
        avg_probs = probs[0].mean(axis=(0, 1))  # average over K and G dims

        # Split attention mass by token type
        img_attn = avg_probs[:, :total_image_tokens].sum(axis=-1).mean()
        lang_attn = avg_probs[:, total_image_tokens:suffix_start].sum(axis=-1).mean()
        self_attn = avg_probs[:, suffix_start:].sum(axis=-1).mean()

        results["per_layer"][layer_idx] = {
            "image_attn": float(img_attn),
            "language_attn": float(lang_attn),
            "self_attn": float(self_attn),
            "probs_head_avg": avg_probs,  # [T_q, S_k]
            "probs_per_head": probs[0].reshape(-1, T_q, S_k),  # [K*G, T_q, S_k]
        }

    return results


# ── Step 4: Visualization ───────────────────────────────────────────────

def plot_attention(results, instruction, save_dir, tag):
    """Create attention visualizations."""
    n_layers = len(results["per_layer"])
    total_img = results["total_image_tokens"]
    total_lang = results["language_tokens"]
    prefix_len = results["kv_cache_len"]

    layers = sorted(results["per_layer"].keys())
    img_vals = [results["per_layer"][l]["image_attn"] for l in layers]
    lang_vals = [results["per_layer"][l]["language_attn"] for l in layers]
    self_vals = [results["per_layer"][l]["self_attn"] for l in layers]

    # ── Figure 1: Stacked bar chart + language zoom ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Pi0.5 Action Expert Attention Distribution\nInstruction: "{instruction[:60]}..."'
        if len(instruction) > 60 else
        f'Pi0.5 Action Expert Attention Distribution\nInstruction: "{instruction}"',
        fontsize=13, fontweight="bold"
    )

    x = np.arange(n_layers)
    ax = axes[0]
    ax.bar(x, img_vals, label=f"Image ({total_img} tok)", color="#4a9eff", alpha=0.85)
    ax.bar(x, lang_vals, bottom=img_vals, label=f"Language ({total_lang} tok)", color="#ff6b6b", alpha=0.85)
    bottom2 = [i + l for i, l in zip(img_vals, lang_vals)]
    ax.bar(x, self_vals, bottom=bottom2, label="Self-attn", color="#50c878", alpha=0.85)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Mass")
    ax.set_title("Distribution by Token Type")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    ax2.bar(x, lang_vals, color="#ff6b6b", alpha=0.85)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Language Attention Mass")
    ax2.set_title("Language Attention (zoomed)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in layers], fontsize=7)
    for i, v in enumerate(lang_vals):
        ax2.text(i, v + max(lang_vals) * 0.02, f"{v*100:.1f}%", ha="center", fontsize=6, color="#cc0000")
    avg_lang = np.mean(lang_vals)
    ax2.axhline(y=avg_lang, color="red", linestyle="--", alpha=0.5, label=f"Avg: {avg_lang*100:.2f}%")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"pi05_attn_{tag}_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Attention heatmaps for selected layers ──
    selected = [0, 3, 6, 9, 12, 15] if n_layers >= 16 else list(range(min(6, n_layers)))
    selected = [l for l in selected if l in results["per_layer"]]

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle(
        f'Action→Prefix Attention Heatmaps (head-averaged)\n"{instruction[:60]}"',
        fontsize=13, fontweight="bold"
    )

    for idx, layer_idx in enumerate(selected):
        ax = axes2[idx // 3][idx % 3]
        probs = results["per_layer"][layer_idx]["probs_head_avg"]  # [T_q, S_k]
        prefix_probs = probs[:, :prefix_len]

        im = ax.imshow(prefix_probs, aspect="auto", cmap="hot", interpolation="nearest")
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_ylabel("Action token")
        ax.set_xlabel("Prefix token")
        ax.axvline(x=total_img - 0.5, color="cyan", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(total_img // 2, -2, "Image", ha="center", fontsize=8, color="cyan")
        ax.text(total_img + total_lang // 2, -2, "Lang", ha="center", fontsize=8, color="cyan")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(selected), 6):
        axes2[idx // 3][idx % 3].set_visible(False)

    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, f"pi05_attn_{tag}_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure 3: Per-head analysis for middle layer ──
    mid_layer = n_layers // 2
    if mid_layer in results["per_layer"]:
        probs_per_head = results["per_layer"][mid_layer]["probs_per_head"]  # [n_heads, T_q, S_k]
        n_heads = probs_per_head.shape[0]
        ncols = min(4, n_heads)
        nrows = (min(8, n_heads) + ncols - 1) // ncols

        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        fig3.suptitle(f'Per-Head Attention at Layer {mid_layer}\n"{instruction[:50]}"', fontsize=13, fontweight="bold")
        if nrows == 1 and ncols == 1:
            axes3 = np.array([[axes3]])
        elif nrows == 1:
            axes3 = axes3[np.newaxis, :]
        elif ncols == 1:
            axes3 = axes3[:, np.newaxis]

        for h in range(min(n_heads, nrows * ncols)):
            ax = axes3[h // ncols][h % ncols]
            hp = probs_per_head[h]  # [T_q, S_k]
            h_img = hp[:, :total_img].sum(axis=-1).mean()
            h_lang = hp[:, total_img:prefix_len].sum(axis=-1).mean()
            h_self = hp[:, prefix_len:].sum(axis=-1).mean()

            im = ax.imshow(hp[:, :prefix_len], aspect="auto", cmap="hot", interpolation="nearest")
            ax.set_title(f"Head {h}: img={float(h_img)*100:.1f}% lang={float(h_lang)*100:.1f}% self={float(h_self)*100:.1f}%", fontsize=8)
            ax.axvline(x=total_img - 0.5, color="cyan", linewidth=1, linestyle="--", alpha=0.7)

        plt.tight_layout()
        fig3.savefig(os.path.join(save_dir, f"pi05_attn_{tag}_per_head.png"), dpi=150, bbox_inches="tight")
        plt.close(fig3)

    # ── Figure 4: Summary bar chart ──
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 5))
    categories = ["Image\n(visual)", "Language\n(instruction)", "Self\n(action)"]
    values = [np.mean(img_vals), np.mean(lang_vals), np.mean(self_vals)]
    colors = ["#4a9eff", "#ff6b6b", "#50c878"]
    bars = ax4.bar(categories, values, color=colors, alpha=0.85, width=0.5)

    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                 f"{val*100:.2f}%", ha="center", fontsize=14, fontweight="bold")

    ax4.set_ylabel("Avg Attention Mass (all layers)")
    ax4.set_title(f'"{instruction[:50]}"')
    ax4.set_ylim(0, max(values) * 1.25)
    ax4.text(0.98, 0.95,
             f"Image: {total_img} tok\nLanguage: {total_lang} tok\nRatio: {total_img/max(total_lang,1):.0f}:1",
             transform=ax4.transAxes, fontsize=10, va="top", ha="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig4.savefig(os.path.join(save_dir, f"pi05_attn_{tag}_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)

    return {
        "avg_image_attn": float(np.mean(img_vals)),
        "avg_language_attn": float(np.mean(lang_vals)),
        "avg_self_attn": float(np.mean(self_vals)),
        "per_layer_language": {str(l): float(v) for l, v in zip(layers, lang_vals)},
        "per_layer_image": {str(l): float(v) for l, v in zip(layers, img_vals)},
        "per_layer_self": {str(l): float(v) for l, v in zip(layers, self_vals)},
    }


def plot_comparison(all_results, save_dir):
    """Create a comparison chart across instruction types."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Pi0.5 Attention: Instruction Sensitivity Analysis", fontsize=14, fontweight="bold")

    types = list(all_results.keys())
    x = np.arange(len(types))
    width = 0.6

    # Plot 1: Language attention comparison
    ax = axes[0]
    lang_vals = [all_results[t]["avg_language_attn"] * 100 for t in types]
    bars = ax.bar(x, lang_vals, width, color="#ff6b6b", alpha=0.85)
    for bar, val in zip(bars, lang_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Language Attention (%)")
    ax.set_title("Language (Instruction) Attention")
    ax.set_xticks(x)
    ax.set_xticklabels(types)

    # Plot 2: Image attention comparison
    ax = axes[1]
    img_vals = [all_results[t]["avg_image_attn"] * 100 for t in types]
    bars = ax.bar(x, img_vals, width, color="#4a9eff", alpha=0.85)
    for bar, val in zip(bars, img_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Image Attention (%)")
    ax.set_title("Image (Visual) Attention")
    ax.set_xticks(x)
    ax.set_xticklabels(types)

    # Plot 3: Stacked comparison
    ax = axes[2]
    img_vals_frac = [all_results[t]["avg_image_attn"] for t in types]
    lang_vals_frac = [all_results[t]["avg_language_attn"] for t in types]
    self_vals_frac = [all_results[t]["avg_self_attn"] for t in types]
    ax.bar(x, img_vals_frac, width, label="Image", color="#4a9eff", alpha=0.85)
    ax.bar(x, lang_vals_frac, width, bottom=img_vals_frac, label="Language", color="#ff6b6b", alpha=0.85)
    bottom2 = [i + l for i, l in zip(img_vals_frac, lang_vals_frac)]
    ax.bar(x, self_vals_frac, width, bottom=bottom2, label="Self", color="#50c878", alpha=0.85)
    ax.set_ylabel("Attention Mass")
    ax.set_title("All Components")
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "pi05_attn_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    save_dir = os.path.join(ROOT, "reports/attention_analysis")
    os.makedirs(save_dir, exist_ok=True)

    # Patch gemma.py BEFORE importing openpi
    patch_gemma()

    try:
        # Now import openpi (uses patched gemma.py)
        import openpi.models.gemma as gemma_module
        print(f"Capture enabled: {gemma_module._attn_capture_enabled}")

        # Load model
        print("\nLoading Pi0.5 model...")
        policy = load_model()

        # Test instructions
        instructions = {
            "seen": "Catch the plastic handle metal hammer and use it on the block.",
            "do_nothing": "Stay still. Do not move the arms.",
            "wrong_action": "Place the hammer gently next to the block.",
        }

        all_results = {}

        for itype, instruction in instructions.items():
            print(f"\n{'='*60}")
            print(f"Running: {itype}")
            print(f"  Instruction: \"{instruction}\"")
            print(f"{'='*60}")

            captures, _ = run_inference_and_capture(policy, gemma_module, instruction)
            results = analyze_attention(captures)

            if results is None:
                print(f"  Skipping visualization for {itype}")
                continue

            stats = plot_attention(results, instruction, save_dir, itype)
            all_results[itype] = {
                "instruction": instruction,
                **stats,
            }

            print(f"  Image attention:    {stats['avg_image_attn']*100:.2f}%")
            print(f"  Language attention:  {stats['avg_language_attn']*100:.2f}%")
            print(f"  Self attention:     {stats['avg_self_attn']*100:.2f}%")

        # Comparison plot
        if len(all_results) > 1:
            plot_comparison(all_results, save_dir)

        # Save JSON
        json_path = os.path.join(save_dir, "attention_summary.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved summary: {json_path}")

        # Final comparison
        print("\n" + "=" * 70)
        print("FINAL COMPARISON: Attention Across Instruction Types")
        print("=" * 70)
        for itype, res in all_results.items():
            print(f"  {itype:15s}: lang={res['avg_language_attn']*100:.2f}%  "
                  f"img={res['avg_image_attn']*100:.2f}%  "
                  f"self={res['avg_self_attn']*100:.2f}%")

        # Conclusion
        if len(all_results) > 1:
            lang_values = [r["avg_language_attn"] for r in all_results.values()]
            lang_range = max(lang_values) - min(lang_values)
            print(f"\n  Language attention range across instructions: {lang_range*100:.2f}%")
            if lang_range < 0.01:
                print("  CONCLUSION: Language attention is virtually IDENTICAL across instructions.")
                print("  -> The action expert barely differentiates between instruction types.")
            elif lang_range < 0.05:
                print("  CONCLUSION: Language attention shows MINOR variation across instructions.")
            else:
                print("  CONCLUSION: Language attention shows SIGNIFICANT variation across instructions.")
                print("  -> The action expert IS sensitive to instruction content.")

    finally:
        # Always restore gemma.py
        restore_gemma()


if __name__ == "__main__":
    main()
