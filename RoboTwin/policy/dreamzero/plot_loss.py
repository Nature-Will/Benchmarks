import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

# ---- Parse & deduplicate ----
raw = defaultdict(lambda: defaultdict(list))

with open('checkpoints/dreamzero_robotwin_lora/loss_log.jsonl') as f:
    for line in f:
        d = json.loads(line)
        step = d['step']
        for k, v in d.items():
            if k != 'step':
                raw[k][step].append(v)

# Average duplicates per step, sort by step
def build_series(key):
    steps = sorted(raw[key].keys())
    vals = [np.mean(raw[key][s]) for s in steps]
    return np.array(steps), np.array(vals)

steps_total, loss_total = build_series('loss')
steps_action, loss_action = build_series('action_loss_avg')
steps_dynamics, loss_dynamics = build_series('dynamics_loss_avg')
steps_lr, lr_vals = build_series('learning_rate')

# ---- EMA smoothing (match LingBot-VLA alpha=0.995) ----
def ema(data, alpha=0.995):
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * out[i-1] + (1 - alpha) * data[i]
    return out

action_smooth = ema(loss_action, 0.9)
dynamics_smooth = ema(loss_dynamics, 0.9)
total_smooth = ema(loss_total, 0.9)

# ---- Style constants (same as LingBot-VLA) ----
BG_GRAY = '#F7F8FA'
C_TOTAL_RAW = '#B0C4DE'
C_TOTAL = '#2563EB'
C_ACTION_RAW = '#F5B7B1'
C_ACTION = '#E74C3C'
C_DYNAMICS_RAW = '#ABEBC6'
C_DYNAMICS = '#27AE60'
C_LR = '#8E44AD'

fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
fig.patch.set_facecolor('white')

# ---- Panel 1: All losses ----
ax1.set_facecolor(BG_GRAY)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color='white', linewidth=1.0)
ax1.xaxis.grid(False)

# Raw data (visible but lighter — scatter + thin line, like LingBot-VLA style)
ax1.scatter(steps_action, loss_action, color=C_ACTION, alpha=0.12, s=6, linewidths=0, rasterized=True)
ax1.scatter(steps_dynamics, loss_dynamics, color=C_DYNAMICS, alpha=0.12, s=6, linewidths=0, rasterized=True)
ax1.scatter(steps_total, loss_total, color=C_TOTAL, alpha=0.15, s=6, linewidths=0, rasterized=True)
ax1.plot(steps_action, loss_action, color=C_ACTION, alpha=0.15, linewidth=0.6, rasterized=True)
ax1.plot(steps_dynamics, loss_dynamics, color=C_DYNAMICS, alpha=0.15, linewidth=0.6, rasterized=True)
ax1.plot(steps_total, loss_total, color=C_TOTAL, alpha=0.18, linewidth=0.6, rasterized=True)

# Smoothed curves (bold)
ax1.plot(steps_action, action_smooth, color=C_ACTION, linewidth=2.0, label='Action Loss (EMA)')
ax1.plot(steps_dynamics, dynamics_smooth, color=C_DYNAMICS, linewidth=2.0, label='Dynamics Loss (EMA)')
ax1.plot(steps_total, total_smooth, color=C_TOTAL, linewidth=2.2, label='Total Loss (EMA)')

# Checkpoint markers
for cs in [1000, 2000, 3000, 4000, 5000]:
    ax1.axvline(x=cs, color='#ddd', linestyle=':', linewidth=0.8)

# Annotate key points (spread out to avoid overlap)
annotations = [
    (steps_total, total_smooth, C_TOTAL, 'Total', 0, (12, 14), -1, (-55, 12)),
    (steps_action, action_smooth, C_ACTION, 'Action', 0, (12, -20), -1, (-55, -18)),
    (steps_dynamics, dynamics_smooth, C_DYNAMICS, 'Dynamics', 0, (12, 6), -1, (-55, -14)),
]
for steps_arr, smooth_arr, color, name, si, soff, ei, eoff in annotations:
    ax1.annotate(f'{smooth_arr[si]:.3f}', (steps_arr[si], smooth_arr[si]),
                 textcoords='offset points', xytext=soff, fontsize=9,
                 color=color, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.0))
    ax1.annotate(f'{smooth_arr[ei]:.4f}', (steps_arr[ei], smooth_arr[ei]),
                 textcoords='offset points', xytext=eoff, fontsize=9,
                 color=color, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.0))

ax1.set_ylabel('Loss (Flow Matching)', fontsize=12, fontweight='medium')
ax1.set_yscale('log')
ax1.set_ylim(0.01, 1.2)
ax1.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white',
           edgecolor='#ddd', framealpha=0.95)
ax1.set_title('DreamZero LoRA Fine-tuning Loss  (beat_block_hammer, 500 demos, 8× H200)',
              fontsize=13.5, fontweight='bold', pad=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#ccc')
ax1.spines['bottom'].set_color('#ccc')
ax1.tick_params(axis='both', labelsize=10)
ax1.set_xlabel('Training Step', fontsize=12, fontweight='medium')

# X-axis formatting
ax1.set_xlim(0, max(steps_total) + 200)
xticks = [0, 1000, 2000, 3000, 4000, 5000]
ax1.set_xticks(xticks)
ax1.set_xticklabels(['0', '1K', '2K', '3K', '4K', '5K'], fontsize=10)

# Footer
fig.text(0.99, 0.005,
         f'Total: {max(steps_total)} steps  |  LoRA rank=4, alpha=4  |  LR=1e-5 (cosine)  |  Batch=8 (1/GPU×8)  |  Action Horizon=24',
         ha='right', va='bottom', fontsize=8.5, color='#888')

plt.tight_layout()
out = '/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/dreamzero/dreamzero_loss.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved to {out}')
plt.close()
