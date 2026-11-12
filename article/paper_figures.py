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
raw = bal['log'][0]; best = max(bal['log'], key=lambda l: l[2]); fb = np.load('runs/reg_T225_gz0.05.npz')
fig, ax = plt.subplots(figsize=(W, 2.2))
ax.plot(fb['t'], fb['R0'], 'k-', label='feedback, $d = 1\\,b$')
ax.plot(best[3], best[4], '--', color='0.35', label='tuned initial data, no control')
ax.plot(raw[3], raw[4], ':', color='0.35', label='no control')
ax.set_xlim(0, 230); ax.set_ylim(0.45, 2.1)
ax.set_xlabel('$t/b$'); ax.set_ylabel('$R_0/b$')
ax.legend(frameon=False, loc='upper right')
fig.savefig(out + '/fig1.pdf')
M = json.load(open('results_lqg_margins.json'))['rows']
d = np.array([r['d'] for r in M]); ours = np.array([r['tolerated_fractional_plant_error'] for r in M])
bound = np.array([1/r['growth_over_effective_delay'] for r in M])
fig, ax = plt.subplots(figsize=(W, 2.2))
dd = np.linspace(0.9, 3.1, 50)
ax.semilogy(dd, np.exp(-1.182*(2*dd - 0.5)), '-', color='0.6', lw=0.8, label='$e^{-\\lambda(2d-0.5b)}$')
ax.semilogy(d, bound, 'o', mfc='none', mec='k', ms=4, label='bound, $1/e^{\\lambda\\tau}$ (measured $\\tau$)')
ax.semilogy(d, ours, 's', color='k', ms=3.5, label='LQG design, $1/\\|T\\|_\\infty$')
ax.set_xlabel('$d/b$'); ax.set_ylabel('tolerated relative error $|\\Delta|$')
ax.set_xticks([1, 1.5, 2, 2.5, 3]); ax.legend(frameon=False, loc='upper right')
fig.savefig(out + '/fig2.pdf')
print('bound', bound, 'ours', ours, 'raw', raw[2], 'best', best[2], 'fb', float(fb['lifetime']))
