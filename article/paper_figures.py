"""Figures for the arXiv paper (PRD single-column width, vector PDF). Run from the repo root: python3 article/paper_figures.py <outdir>"""
import json, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out = sys.argv[1]
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'lines.linewidth': 1.0, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
W = 3.375
bal = json.load(open('runs/balance_log.json'))
raw = bal['log'][0]; best = max(bal['log'], key=lambda l: l[2])
fb = np.load('runs/reg_T225_gz0.05.npz'); fine = np.load('runs/reg_long_dx0.0125.npz')
fig, ax = plt.subplots(figsize=(W, 2.3))
ax.plot(fb['t'], fb['R0'], 'k-', label='feedback, $d = 1\\,b$')
ax.plot(best[3], best[4], '--', color='0.35', label='tuned initial data, no control')
ax.plot(raw[3], raw[4], ':', color='0.35', label='no control')
ax.set_xlim(0, 230); ax.set_ylim(0.45, 2.1)
ax.set_xlabel('$t/b$'); ax.set_ylabel('$R_0/b$')
ax.legend(frameon=False, loc='lower right')
ins = ax.inset_axes([0.45, 0.58, 0.52, 0.38])
ins.plot(fb['t'], 100*(fb['R0'] - 1), 'k-', lw=0.8, label='$\\Delta x = 0.025\\,b$')
ins.plot(fine['t'], 100*(fine['R0'] - 1), '-', color='0.55', lw=0.8, label='$\\Delta x = 0.0125\\,b$')
ins.set_xlim(0, 225); ins.set_ylabel('$(R_0 - b)/b$ [%]', fontsize=6); ins.tick_params(labelsize=6)
ins.legend(frameon=False, fontsize=5.5, loc='upper left')
fig.savefig(out + '/fig1.pdf')
B = json.load(open('results_bound_zeros.json'))
d = np.array([r['d'] for r in B])
fig, ax = plt.subplots(figsize=(W, 2.3))
ax.semilogy(d, [r['bound_exact_delay_pct']/100 for r in B], 'v', mfc='none', mec='0.5', ms=4, label='dead time only, $e^{-\\lambda\\tau}$')
ax.semilogy(d, [r['bound_two_channel_pct']/100 for r in B], 'o', mfc='none', mec='k', ms=4, label='with common zero (two channels)')
ax.semilogy(d - 0.05, [r['bound_radius_only_pct']/100 for r in B], '^', color='0.5', ms=3.5, label='radius channel only')
ax.semilogy(d + 0.05, [r['lqg_achieved_pct']/100 for r in B], 's', color='k', ms=3, label='LQG designs, $1/\\|T\\|_\\infty$')
ax.set_xlabel('$d/b$'); ax.set_ylabel('tolerated relative error $|\\Delta|$')
ax.set_xticks([1, 1.5, 2, 2.5, 3]); ax.set_xlim(0.8, 3.2); ax.set_ylim(3e-4, 0.5); ax.legend(frameon=False, loc='lower left')
fig.savefig(out + '/fig2.pdf')
