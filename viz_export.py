"""Export the identified models and the tested observer-based controller for the visual demo.
Also checks, in Python, the exact loop the page runs (plant at the throat, light-travel delay on the
readings, Kalman filter + LQR) and finds how much plant-gain error each distance tolerates."""
import json
import math
import numpy as np
from control_lqg import build_model, design, DT
from control_lqg_run import TUNE

MARG = {r['d']: r for r in json.load(open('results_lqg_margins.json'))['rows']}


def sparse(Mx):
    i, j = np.nonzero(Mx)
    return [[int(a), int(b), float(Mx[a, b])] for a, b in zip(i, j)]


def loop_matrix(mc, g, mp, nd, eps):
    """Closed loop: plant (throat readings) -> delay of nd steps -> filter + LQR -> pulse, plant gain (1+eps)."""
    A, B, C, K, Mk = mc['A'], mc['B'], mc['C'], g['K'], g['Mk']
    Ap, Bp, Cp = mp['A'], mp['B']*(1 + eps), mp['C']
    n, npl = A.shape[0], Ap.shape[0]
    I = np.eye(n)
    Ac = (A - B @ K) @ (I - Mk @ C); Bc = (A - B @ K) @ Mk
    Cc = -K @ (I - Mk @ C); Dc = -K @ Mk
    nb = 2*nd                                              # buffer of delayed readings
    N = npl + nb + n
    T = np.zeros((N, N))
    # y_recv = last buffer entry (or plant output if nd == 0)
    Yr = np.zeros((2, N))
    if nd:
        Yr[:, npl + nb - 2:npl + nb] = np.eye(2)
    else:
        Yr[:, :npl] = Cp
    Urow = Dc @ Yr; Urow[:, npl + nb:] += Cc
    T[:npl, :npl] = Ap; T[:npl] += Bp @ Urow
    if nd:
        T[npl:npl + 2, :npl] = Cp
        for b in range(1, nd):
            T[npl + 2*b:npl + 2*b + 2, npl + 2*(b - 1):npl + 2*b] = np.eye(2)
    T[npl + nb:, npl + nb:] = Ac; T[npl + nb:] += Bc @ Yr
    return T


def radius(mc, g, mp, nd, eps):
    return float(np.abs(np.linalg.eigvals(loop_matrix(mc, g, mp, nd, eps))).max())


def gain_margin(mc, g, mp, nd, sign):
    lo, hi = 0.0, None
    e = 0.002
    while e < 5.0:
        if radius(mc, g, mp, nd, sign*e) > 1.0 + 1e-6:
            hi = e; break
        lo = e; e *= 1.25
    if hi is None:
        return None
    for _ in range(30):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if radius(mc, g, mp, nd, sign*mid) <= 1.0 + 1e-6 else (lo, mid)
    return sign*0.5*(lo + hi)

out = {'dt': DT, 'distances': []}
for d in (1.0, 1.5, 2.0, 3.0):
    mc = build_model(d); g = design(mc, **TUNE); mp = build_model(d, shift=d)
    nd = int(round(d/DT))
    # plant delayed by nd steps must reproduce the controller's model exactly
    def impulse(m, steps):
        x = m['B'][:, 0].copy(); ys = []
        for _ in range(steps):
            ys.append(m['C'] @ x); x = m['A'] @ x
        return np.array(ys)
    yc, yp = impulse(mc, 120), impulse(mp, 120)
    mism = float(np.abs(yc[nd:] - yp[:-nd]).max()/np.abs(yc).max())
    r0 = radius(mc, g, mp, nd, 0.0)
    gm = [gain_margin(mc, g, mp, nd, -1), gain_margin(mc, g, mp, nd, +1)]
    print('d=%g: plant %d states, model %d, delay steps %d, plant-vs-model mismatch %.1e, loop radius %.5f, gain error tolerated %s' % (
        d, mp['A'].shape[0], mc['A'].shape[0], nd, mism, r0, ['%.3f' % v if v is not None else None for v in gm]))
    ix, ixp = mc['idx'], mp['idx']
    out['distances'].append({
        'd': d, 'nd': nd, 'lam': mc['lam'], 'k': mc['k'][0],
        'ctrl': {'n': int(mc['A'].shape[0]), 'A': sparse(mc['A']), 'B': mc['B'][:, 0].tolist(), 'C': mc['C'].tolist(),
                 'K': g['K'][0].tolist(), 'Mk': g['Mk'].tolist(), 'iZ': ix['Z'], 'iV': ix['V'], 'iG': ix['G']},
        'plant': {'n': int(mp['A'].shape[0]), 'A': sparse(mp['A']), 'B': mp['B'][:, 0].tolist(), 'C': mp['C'].tolist(),
                  'iG': ixp['G'], 'iZ': ixp['Z'], 'iF': ixp['F'], 'iV': ixp['V']},
        'margin': {'tolerated': MARG[d]['tolerated_fractional_plant_error'], 'best': 1/MARG[d]['growth_over_effective_delay'],
                   'loop_delay': MARG[d]['effective_loop_delay'], 'gain_lo': gm[0], 'gain_hi': gm[1]}})
json.dump(out, open('viz_models.json', 'w'))
import os
print('viz_models.json', os.path.getsize('viz_models.json'), 'bytes')

# time-domain rehearsal of the page's loop at d = 1: seed, hold, then a kick
D = out['distances'][0]
mc = build_model(1.0); g = design(mc, **TUNE); mp = build_model(1.0, shift=1.0); nd = 4
rng = np.random.default_rng(1)
for label, on in (('no control', False), ('control on', True)):
    xp = np.zeros(mp['A'].shape[0]); xbar = np.zeros(mc['A'].shape[0]); buf = [np.zeros(2)]*nd; started = False
    xp[mp['idx']['G']] = 1e-6
    trace = []
    for n in range(800):
        y = mp['C'] @ xp; buf.append(y); yr = buf[-1 - nd]
        u = 0.0
        if on:
            if not started:
                xbar[mc['idx']['Z']], xbar[mc['idx']['V']] = yr[0], yr[1]; started = True
            xh = xbar + g['Mk'] @ (yr - mc['C'] @ xbar)
            u = float(np.clip(-(g['K'] @ xh)[0], -0.02, 0.02))
            xbar = mc['A'] @ xh + mc['B'][:, 0]*u
        xp = mp['A'] @ xp + mp['B'][:, 0]*u
        xp[mp['idx']['G']] += 1e-8*rng.standard_normal()
        if n == 400: xp[mp['idx']['G']] -= 1e-4
        trace.append(y[0])
        if abs(y[0]) > 0.3: break
    print('%s: steps %d (t = %.1f b), final radius offset %+.2e, max |offset| %.2e' % (label, len(trace), len(trace)*DT, trace[-1], np.abs(trace).max()))
