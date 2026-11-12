"""Distance sweep summary.  Writes results_sweep.json and control_sweep.png."""
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAM = 1.182


def load(d, kappa):
    for pat in ('runs/sweep_d%g_o3_on10_gp0_gz0_k%g.npz', 'runs/sweep_d%g_o3_on10_gz0_k%g.npz'):
        try:
            return np.load(pat % (d, kappa))
        except FileNotFoundError:
            pass
    return None


def envelope(R):
    lt, la, k = R['launch_t'], R['launch_a'], float(R['k'])
    edges = np.arange(lt[0], lt[-1] - 2.5, 2.5)
    return edges + 1.25, np.array([k*np.abs(la[(lt >= e) & (lt < e + 2.5)]).max() for e in edges])


def growth_rate(R):
    """Exponential growth rate of the control effort while it climbs from 1e-4 to 1e-2 (throat-referred)."""
    tc, env = envelope(R)
    w = (env > 1e-4) & (env < 1e-2)
    if w.sum() < 3:
        return None
    return float(np.polyfit(tc[w], np.log(env[w]), 1)[0])

rows = []
for d in (1.0, 1.5, 2.0, 3.0):
    for kappa in (0.3, 0.35, 0.4, 0.5, 0.75):
        R = load(d, kappa)
        if R is None:
            continue
        t = R['t']
        rows.append({'d': d, 'loop_delay': 2*d, 'delay_growth_factor': float(np.exp(2*LAM*d)), 'kappa': kappa,
                     'outcome': str(R['outcome']), 'lifetime': float(R['lifetime']), 'run_length': float(t[-1]) if str(R['outcome']) == 'alive' else None,
                     'growing_gain_k': float(R['k']), 'effort_growth_rate_per_b': growth_rate(R)})
res = {'setup': 'estimator order 3, readings before t = 10 b ignored, no radius set-point, start from grid-balanced wormhole, dx = 0.025, pulse width 0.3 b',
       'uncontrolled_balanced_lifetime': '20-37 b depending on round-off', 'runs': rows}
json.dump(res, open('results_sweep.json', 'w'), indent=1)
for r in rows:
    print('d=%.1f delay growth %7.1f kappa=%.2f: %-8s %6.1f   effort growth rate %s' % (
        r['d'], r['delay_growth_factor'], r['kappa'], r['outcome'], r['lifetime'], '%.2f /b' % r['effort_growth_rate_per_b'] if r['effort_growth_rate_per_b'] else '-'))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
cols = {1.0: '#08306b', 1.5: '#2171b5', 2.0: '#6baed6', 3.0: '#b2182b'}
a = ax[0]
for d in cols:
    R = load(d, 0.5)
    a.plot(R['t'], R['R0'], color=cols[d], label='d = %g b' % d)
a.set_ylim(0.9, 1.3); a.set_xlabel('t / b'); a.set_ylabel('throat radius / b'); a.set_title('Throat radius, gain 0.5'); a.legend(fontsize=8)
a = ax[1]
for d in cols:
    tc, env = envelope(load(d, 0.5))
    a.semilogy(tc, env, color=cols[d], label='d = %g b' % d)
a.set_xlabel('t / b'); a.set_ylabel('largest pulse per 2.5 b, throat-referred (k·a)'); a.set_title('Control effort, gain 0.5'); a.legend(fontsize=8)
a = ax[2]
for kappa, mk in ((0.5, 'o'), (0.35, 's')):
    pts = [(r['d'], r['lifetime'], r['outcome']) for r in rows if r['kappa'] == kappa]
    a.plot([p[0] for p in pts], [p[1] for p in pts], mk + '-', color='#555555' if kappa == 0.5 else '#1b7837', label='gain %.2f' % kappa)
    for p in pts:
        if p[2] == 'alive':
            a.annotate('still open', (p[0], p[1]), textcoords='offset points', xytext=(6, -12), fontsize=7)
a.axhspan(20, 37, color='#b2182b', alpha=0.12, label='no control (balanced start)')
a.set_xlabel('controller distance d / b'); a.set_ylabel('lifetime / b'); a.set_title('Lifetime against controller distance'); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig('control_sweep.png', dpi=150)
