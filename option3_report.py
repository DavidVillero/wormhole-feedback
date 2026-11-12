"""Option 3: does the fast oscillation depend on resolution?  Writes results_option3.json, control_option3.png."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(d, kp, fine):
    if fine:
        return np.load('runs/sweep_d%g_o3_on10_gz0_k%g_dx0.0125.npz' % (d, kp))
    return np.load('runs/sweep_d%g_o3_on10_gp0_gz0_k%g_cut2.npz' % (d, kp))


def fast(R, win=5.0):
    """Amplitude of the sign-flipping part of the pulse sequence, throat-referred, in windows."""
    lt, la, k = R['launch_t'], R['launch_a'], float(R['k'])
    alt = np.zeros_like(la); alt[1:-1] = la[1:-1] - 0.5*(la[:-2] + la[2:])
    edges = np.arange(lt[0] + 2, lt[-1] - win, win)
    tc = edges + win/2
    env = np.array([k*np.sqrt(np.mean(alt[(lt >= e) & (lt < e + win)]**2)) for e in edges])
    return tc, env, np.interp(tc, R['t'], R['R0'] - 1)


def offset_at(R, level=1e-5):
    tc, env, dR = fast(R)
    i = np.nonzero(env > level)[0]
    return (float(tc[i[0]]), float(dR[i[0]])) if len(i) else (None, None)

rows = []
for d in (1.5, 2.0):
    for kp in (0.5, 0.35):
        C, F = load(d, kp, False), load(d, kp, True)
        tc_, oc = offset_at(C); tf_, of = offset_at(F)
        rows.append({'d': d, 'gain': kp,
                     'coarse': {'outcome': str(C['outcome']), 'lifetime': float(C['lifetime']), 't_fast_1e-5': tc_, 'radius_offset_then': oc},
                     'fine': {'outcome': str(F['outcome']), 'lifetime': float(F['lifetime']), 't_fast_1e-5': tf_, 'radius_offset_then': of},
                     'lifetime_ratio': float(F['lifetime'])/float(C['lifetime']) if str(C['outcome']) != 'alive' and str(F['outcome']) != 'alive' else None})
res = {'question': 'fine grid has 4x less radius drift: does the fast oscillation start 4x later (drift threshold), at the same time (drift irrelevant), or in between?',
       'predictions': {'pure_drift_threshold': 4.0, 'growth_rate_proportional_to_offset': 2.0, 'drift_irrelevant': 1.0},
       'runs': rows}
json.dump(res, open('results_option3.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for j, (d, kp) in enumerate(((2.0, 0.5), (1.5, 0.5))):
    a = ax[j]
    for fine, col, lab in ((False, '#92c5de', 'dx = 0.025'), (True, '#08306b', 'dx = 0.0125')):
        tc, env, dR = fast(load(d, kp, fine))
        a.semilogy(tc, env, color=col, label=lab)
    a.set_xlabel('t / b'); a.set_ylabel('fast (sign-flipping) pulse amplitude, k·a'); a.set_title('d = %g b, gain %g' % (d, kp)); a.legend(fontsize=8)
a = ax[2]
for d, mk in ((2.0, 'o'), (1.5, 's')):
    for fine, col in ((False, '#92c5de'), (True, '#08306b')):
        tc, env, dR = fast(load(d, 0.5, fine))
        a.semilogy(dR*1e3, env, mk + '-', ms=3, color=col, label='d = %g, dx = %s' % (d, '0.0125' if fine else '0.025'))
a.set_xlabel('throat-radius offset (×1e-3 b)'); a.set_ylabel('fast pulse amplitude, k·a'); a.set_title('Same data against radius offset'); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig('control_option3.png', dpi=150)
