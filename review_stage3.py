"""Independent checks for the Stage 3 review.   WH_TRACK=1 WH_MCAL=0.005 python3 review_stage3.py
A. drift: is the post-crossing throat drift a boost of the conformal coordinates?  Gauge-invariant
   profiles (grad R)^2 and Misner-Sharp mass at fixed areal radius on each branch, the two trapping
   horizons (R_v = 0, R_u = 0), and the fit R_t = -beta R_x near the throat.
B. anchor: SensedLQG takes its anchor copy before the first-reading initialisation (transit_sensed.py
   l.128-147), so the filter starts from zero.  Re-run with the initialisation applied first.
C. shape: sensed controller, calibrated at width 1, against travellers of width 0.6 and 1.6.
"""
import os
os.environ.setdefault('WH_SIDES', '1'); os.environ.setdefault('WH_TRACK', '1'); os.environ.setdefault('WH_MCAL', '0.005')
import json
from multiprocessing import Pool
import numpy as np
import transit_ff as TF
import transit_sensed as TS
from control_solver import grid, evolve, throat_reading
from control_feedback import OUT_DT, DX
from control_lqg import build_model, design, DT
from control_lqg_run import TUNE


def invariants(t, x, U):
    dx = x[1] - x[0]
    R, S, PR = U[0], U[1], U[4]
    Rx = np.gradient(R, dx)
    chi = np.exp(-2*S)*(Rx**2 - PR**2)
    xt, rmin, pt = throat_reading(x, U)
    it = int(round((xt - x[0])/dx))
    row = {'t': t, 'xt': xt, 'Rmin': rmin, 'Rt_at_min': pt, 'sigma_t': float(S[it])}
    for Rs in (1.5, 2.0, 3.0):
        for side, sl in (('R', slice(it, it + int(6/dx))), ('L', slice(it, it - int(6/dx), -1))):
            Rb, cb = R[sl], chi[sl]
            row['chi_%s_%g' % (side, Rs)] = float(np.interp(Rs, Rb, cb))
            row['m_%s_%g' % (side, Rs)] = float(0.5*Rs*(1 - np.interp(Rs, Rb, cb)))
    w = slice(it - int(3/dx), it + int(3/dx))
    for nm, f in (('xv', PR + Rx), ('xu', PR - Rx)):              # trapping horizons R_v = 0, R_u = 0
        g = f[w]; k = np.nonzero(np.sign(g[:-1]) != np.sign(g[1:]))[0]
        row[nm] = [float(x[w][j] - g[j]*dx/(g[j+1] - g[j])) for j in k]
    a, b = PR[w], Rx[w]
    beta = -float((a*b).sum()/(b*b).sum())
    row['beta_fit'] = beta
    row['boost_resid'] = float(np.sqrt(np.mean((a + beta*b)**2))/max(np.sqrt(np.mean(a**2)), 1e-300))
    row['rms_Rt'] = float(np.sqrt(np.mean(a**2)))
    return row


def drift(dM_t):
    T = 90.0
    x = grid(DX, T + TF.X0 + 16.0)
    U, dM = TF.traveller(x, TF.s_amp_balanced(), dM_t)
    m = build_model(TF.D); sig = TF.load_signature()
    ctl = TF.TransitLQG(m, design(m, **TUNE), travellers=[(TF.X0, dM)], c=float(sig['c']), a_max=0.1, sig=sig)
    rows = []

    def hook(t, x, U, h):
        if abs(t - round(t)) < 1e-6:
            rows.append(invariants(t, x, U))
        ctl(t, x, U, h)
    r = evolve(x, U, T, out_dt=OUT_DT, controller=hook, ctrl_dt=DT)
    return {'job': 'drift', 'mass': dM, 'outcome': r['outcome'], 'rows': rows}


class FixedInit(TS.SensedLQG):
    def decide(self, t, hist):
        m = self.m
        if t - m['d'] >= self.tm_on and not self.launch:
            i = min(int(round((t - m['d'])/OUT_DT)), len(hist['t']) - 1)
            self.xbar[m['idx']['Z']] = hist['R0'][i] - 1.0; self.xbar[m['idx']['V']] = hist['PR0'][i]
        first = not self.launch
        u = super().decide(t, hist)
        if first and u is not None:
            self.first_anchor = self.anchor[0] if self.anchor else None
        return u


def sensed(args):
    dM_t, width, fixed = args
    TF.WIDTH = width
    T = 90.0
    x = grid(DX, T + TF.X0 + 16.0)
    U, dM = TF.traveller(x, TF.s_amp_balanced(), dM_t)
    m = build_model(TF.D)
    ctl = (FixedInit if fixed else TS.SensedLQG)(m, design(m, **TUNE), TF.load_signature(), json.load(open(TS.SCAL)))
    anchored = []
    r = evolve(x, U, T, out_dt=OUT_DT, controller=ctl, ctrl_dt=DT/TS.SUB)
    h = r['hist']; t = h['t']; dR = h['R0'] - 1.0
    pre = (t >= 11) & (t < 16)
    return {'job': 'sensed', 'mass': dM, 'width': width, 'fixed_init': fixed, 'outcome': r['outcome'], 'lifetime': float(r['lifetime']),
            'anchor_index': ctl.anchor[0] if ctl.anchor else None, 'anchor_xbar_norm': float(np.abs(ctl.anchor[1]).max()) if ctl.anchor else None,
            'est_mass': ctl.est[0], 'est_arrival': ctl.est[1],
            'max_offset_before_arrival': float(np.abs(dR[pre]).max()),
            'max_offset_after_arrival': float(np.abs(dR[t >= TF.X0 - 2]).max())}


def job(a):
    return drift(a[1]) if a[0] == 'drift' else sensed(a[1:])


if __name__ == '__main__':
    jobs = [('drift', 5e-3), ('s', 1e-2, 1.0, True), ('s', 1.25e-2, 1.0, True), ('s', 1e-2, 1.0, False),
            ('s', 5e-3, 0.6, False), ('s', 5e-3, 1.6, False)]
    with Pool(2) as pool:
        out = pool.map(job, jobs, chunksize=1)
    json.dump(out, open('results_review_stage3.json', 'w'), indent=1)
    for o in out:
        if o['job'] == 'sensed':
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()})
