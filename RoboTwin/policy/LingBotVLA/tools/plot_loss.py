import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

# Load loss data
steps, losses, grad_norms, lrs, epochs = [], [], [], [], []
with open('checkpoints/beat_block_hammer_randomized500/checkpoints/loss.jsonl') as f:
    for line in f:
        d = json.loads(line)
        steps.append(d['step'])
        losses.append(d['loss'])
        grad_norms.append(d['grad_norm'])
        epochs.append(d['epoch'])

steps = np.array(steps)
losses = np.array(losses)
grad_norms = np.array(grad_norms)

# Smoothed loss (EMA)
def ema(data, alpha=0.99):
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * out[i-1] + (1 - alpha) * data[i]
    return out

loss_smooth = ema(losses, alpha=0.995)
grad_smooth = ema(grad_norms, alpha=0.995)

BG_GRAY = '#F7F8FA'
C_RAW = '#B0C4DE'
C_SMOOTH = '#2563EB'
C_GRAD_RAW = '#D4C5A9'
C_GRAD_SMOOTH = '#D97706'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.08})
fig.patch.set_facecolor('white')

# --- Loss plot ---
ax1.set_facecolor(BG_GRAY)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color='white', linewidth=1.0)
ax1.xaxis.grid(False)

ax1.plot(steps, losses, color=C_RAW, alpha=0.25, linewidth=0.4, rasterized=True)
ax1.plot(steps, loss_smooth, color=C_SMOOTH, linewidth=2.0, label='Training Loss (EMA)')

# Checkpoint markers at every 5K steps
ckpt_steps = np.arange(5000, 56000, 5000)
for cs in ckpt_steps:
    idx = np.searchsorted(steps, cs)
    if idx < len(steps):
        ax1.axvline(x=cs, color='#ccc', linestyle=':', linewidth=0.8, alpha=0.6)

# Annotate key milestones
for cs, label in [(1, 'Start'), (5000, '5K'), (20000, '20K'), (56561, 'End')]:
    idx = np.searchsorted(steps, cs)
    if idx >= len(steps):
        idx = len(steps) - 1
    ax1.annotate(f'{losses[idx]:.3f}', (steps[idx], loss_smooth[idx]),
                 textcoords='offset points', xytext=(8, 10), fontsize=9,
                 color=C_SMOOTH, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C_SMOOTH, lw=1.0))

ax1.set_ylabel('Loss (L1 Flow Matching)', fontsize=12, fontweight='medium')
ax1.set_yscale('log')
ax1.set_ylim(0.005, 1.0)
ax1.legend(loc='upper right', fontsize=11, frameon=True, facecolor='white',
           edgecolor='#ddd', framealpha=0.95)
ax1.set_title('LingBot-VLA Fine-tuning Loss  (beat_block_hammer, 500 demos, 8x H200)',
              fontsize=13.5, fontweight='bold', pad=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#ccc')
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='both', labelsize=10)

# --- Gradient norm plot ---
ax2.set_facecolor(BG_GRAY)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color='white', linewidth=1.0)
ax2.xaxis.grid(False)

ax2.plot(steps, grad_norms, color=C_GRAD_RAW, alpha=0.2, linewidth=0.4, rasterized=True)
ax2.plot(steps, grad_smooth, color=C_GRAD_SMOOTH, linewidth=1.8, label='Gradient Norm (EMA)')

ax2.set_ylabel('Gradient Norm', fontsize=12, fontweight='medium')
ax2.set_xlabel('Training Step', fontsize=12, fontweight='medium')
ax2.set_yscale('log')
ax2.legend(loc='upper right', fontsize=11, frameon=True, facecolor='white',
           edgecolor='#ddd', framealpha=0.95)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#ccc')
ax2.spines['bottom'].set_color('#ccc')
ax2.tick_params(axis='both', labelsize=10)

# X-axis formatting
ax2.set_xlim(0, max(steps) + 500)
xticks = [0, 10000, 20000, 30000, 40000, 50000]
ax2.set_xticks(xticks)
ax2.set_xticklabels(['0', '10K', '20K', '30K', '40K', '50K'], fontsize=10)

# Add epoch info
epoch_final = max(epochs)
fig.text(0.99, 0.01, f'Total: {len(steps)} steps, {epoch_final} epochs  |  LR=1e-4 (constant)  |  Batch=32  |  Chunk=50',
         ha='right', va='bottom', fontsize=9, color='#888')

plt.tight_layout()
out = '/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVLA/figures/lingbotvla_loss.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved to {out}')
plt.close()
