import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor('white')

C_PI05 = '#5B8FF9'
C_VA   = '#F6634E'
C_VLA  = '#61C979'
BG_GRAY = '#F7F8FA'

steps      = [5000, 10000, 15000, 20000]
pi05_curve = [57, 69, 73, 72]
vla_curve  = [85, 78, 78, 71]

ax.set_facecolor(BG_GRAY)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='white', linewidth=1.2)
ax.xaxis.grid(False)

# LingBot-VA baseline band + line
ax.axhspan(94.5, 97.5, color=C_VA, alpha=0.08, zorder=1)
ax.axhline(y=96, color=C_VA, linestyle='--', linewidth=2.2, alpha=0.85, zorder=2,
           label='LingBot-VA  (96.0%)')

# LingBot-VLA curve
ax.plot(steps, vla_curve, 's-', color=C_VLA, linewidth=2.8, markersize=10,
        markeredgecolor='white', markeredgewidth=2.2, zorder=5,
        label='LingBot-VLA fine-tuned  (best 85.0%)')

# Pi0.5 curve
ax.plot(steps, pi05_curve, 'o-', color=C_PI05, linewidth=2.8, markersize=10,
        markeredgecolor='white', markeredgewidth=2.2, zorder=4,
        label='Pi0.5 fine-tuned  (best 73.0%)')

# Point labels - Pi0.5
for s, v in zip(steps, pi05_curve):
    ax.annotate(f'{v}%', (s, v), textcoords='offset points',
                xytext=(0, -18), ha='center', fontsize=11.5, color=C_PI05,
                fontweight='bold')

# Point labels - LingBot-VLA
for s, v in zip(steps, vla_curve):
    ax.annotate(f'{v}%', (s, v), textcoords='offset points',
                xytext=(0, 13), ha='center', fontsize=11.5, color=C_VLA,
                fontweight='bold')

# LingBot-VA label
ax.text(12500, 98.5, 'LingBot-VA  96%', fontsize=13, color=C_VA,
        fontweight='bold', ha='center', va='bottom')

# Delta arrow between Pi0.5 best and LingBot-VLA best
ax.annotate('', xy=(5000, 84), xytext=(5000, 58.5),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.6))
ax.text(5700, 71, '+28%', fontsize=12, fontweight='bold', color='#444',
        ha='left', va='center')

# Delta arrow between LingBot-VLA best and LingBot-VA
ax.annotate('', xy=(7500, 95), xytext=(7500, 86),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.4))
ax.text(8200, 90.5, '+11%', fontsize=11, fontweight='bold', color='#444',
        ha='left', va='center')

ax.set_xlim(3000, 22000)
ax.set_ylim(45, 108)
ax.set_xticks(steps)
ax.set_xticklabels(['5K', '10K', '15K', '20K'], fontsize=11.5)
ax.set_xlabel('Fine-tuning Steps', fontsize=13, fontweight='medium')
ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='medium')
ax.set_title('beat_block_hammer  (demo_randomized, 100 episodes)',
             fontsize=14.5, fontweight='bold', pad=14)
ax.legend(loc='lower right', fontsize=10.5, frameon=True, facecolor='white',
          edgecolor='#ddd', framealpha=0.95, borderpad=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#ccc')
ax.spines['bottom'].set_color('#ccc')
ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()
out = '/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVA/eval_comparison.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved to {out}')
plt.close()
