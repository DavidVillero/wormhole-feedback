"""Figures for the Medium article (light background, palette from the dataviz reference palette)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': BG, 'axes.facecolor': BG})


def style(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)

# ---- Figure 2: throat radius over time --------------------------------------
bal = json.load(open('runs/balance_log.json'))
raw = bal['log'][0]                                   # trim 0: the wormhole as it is
best = max(bal['log'], key=lambda l: l[2])            # starting state tuned to round-off, no control
fb = np.load('runs/reg_T225_gz0.05.npz')
fig, ax = plt.subplots(figsize=(10, 5.2))
style(ax)
ax.plot(raw[3], raw[4], color=ORANGE, lw=2)
ax.plot(best[3], best[4], color=AQUA, lw=2)
ax.plot(fb['t'], fb['R0'], color=BLUE, lw=2)
for tr, col in ((raw, ORANGE), (best, AQUA)):
    ax.plot(tr[3][-1], tr[4][-1], 'o', ms=8, color=col, mec=BG, mew=2)
ax.annotate('No control: collapses at t = %.1f' % raw[2], (raw[3][-1], raw[4][-1]), xytext=(14, -6), textcoords='offset points', color=INK, fontsize=11.5, va='center')
ax.annotate('Starting state tuned to 16 digits,\nno control: %s at t = %.0f' % ('blows open' if best[4][-1] > 1 else 'collapses', best[2]), (best[3][-1], best[4][-1]), xytext=(14, 0), textcoords='offset points', color=INK, fontsize=11.5, va='center')
ax.annotate('Feedback control: still open at t = 225 (run end)', (150, 1.0), xytext=(0, 12), textcoords='offset points', color=INK, fontsize=11.5, ha='center')
ax.set_xlim(0, 230); ax.set_ylim(0.4, 2.15)
ax.set_xlabel('Time (light-crossing times of the throat)'); ax.set_ylabel('Throat radius (starting radius = 1)')
ax.set_title('Feedback keeps the throat open; perfect tuning only buys time', loc='left', color=INK, fontsize=14, pad=14)
fig.tight_layout(); fig.savefig('article/fig2_throat.png', dpi=200)

# ---- Figure 3: model accuracy needed against controller distance -----------------
M = json.load(open('results_lqg_margins.json'))['rows']
d = np.array([r['d'] for r in M]); ours = np.array([100*r['tolerated_fractional_plant_error'] for r in M])
bestp = np.array([100/r['growth_over_effective_delay'] for r in M])
fig, ax = plt.subplots(figsize=(10, 5.2))
style(ax)
ax.set_yscale('log')
ax.plot(d, bestp, color=MUTED, lw=2, ls='--', marker='o', ms=8, mec=BG, mew=2)
ax.plot(d, ours, color=BLUE, lw=2, marker='o', ms=8, mec=BG, mew=2)
ax.annotate('Limit for any controller', (d[-1], bestp[-1]), xytext=(10, 0), textcoords='offset points', color=INK2, fontsize=11.5, va='center')
ax.annotate('Our controller', (d[-1], ours[-1]), xytext=(10, 0), textcoords='offset points', color=INK, fontsize=11.5, va='center')
for x, y in zip(d, ours):
    ax.annotate(('%.2g%%' % y), (x, y), xytext=(0, -18), textcoords='offset points', ha='center', color=INK2, fontsize=10.5)
ax.set_xticks(d); ax.set_xticklabels(['1', '1.5', '2', '3'])
ax.set_xlim(0.85, 3.55); ax.set_ylim(0.03, 40)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: ('%g%%' % v)))
ax.set_xlabel('Controller distance from the throat (throat radii)')
ax.set_ylabel('Largest model error that can be tolerated')
ax.set_title('Each extra throat radius of distance demands about ten times more accuracy', loc='left', color=INK, fontsize=14, pad=14)
fig.tight_layout(); fig.savefig('article/fig3_distance.png', dpi=200)
print('raw', raw[2], 'best', best[2], 'fb', float(fb['lifetime']), 'ours', ours, 'best', bestp)
