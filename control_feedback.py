"""Control study, step 4: closed-loop feedback on the wormhole's unstable mode.

The controller sits at coordinate distance d on both sides of the throat.
  * It reads the throat radius and its rate of change with a light-travel delay d.
  * It launches an ingoing phantom pulse that reaches the throat a further d later.
So the loop delay is 2d, during which the unstable mode grows by exp(2*lam*d).
The control law is a model-based predictor: estimate the growing-mode amplitude,
predict it forward to the pulse's arrival (including pulses already in flight, using
the measured impulse response), and cancel a fraction kappa of it.

Pulses are inserted as constraint-satisfying data (see control_solver.py); no source terms.
"""
import json
import math
import sys
import numpy as np
from control_solver import grid, initial_state, evolve, add_pulse, d1_4, mid, FOURPI

import os
DX = float(os.environ.get('WH_DX', 0.025))     # grid spacing; set WH_DX for the resolution check
TAG = '' if DX == 0.025 else '_dx%g' % DX      # suffix on calibration files for other resolutions
OUT_DT = 0.05


def ext_solve(x, U, i_s):
    """Solve H = 0, M = 0 for (R, R_t) from index i_s to the right edge (scalar RK4, fast)."""
    R, S, F, G, PR, PS, PF, PG = U
    dx = x[1] - x[0]
    Fx, Gx = d1_4(F, dx), d1_4(G, dx)
    arrs = [np.exp(2.0*S), d1_4(S, dx), PS, -(PF**2 + Fx**2) + (PG**2 + Gx**2), -PF*Fx + PG*Gx]
    e, sx, ps, E, J = [a[i_s:].tolist() for a in arrs]
    em, sxm, psm, Em, Jm = [mid(a)[i_s:].tolist() for a in arrs]
    r, rx, p = float(R[i_s]), float(d1_4(R, dx)[i_s]), float(PR[i_s])
    m = len(e)
    Ro, Po = [r], [p]
    tp, fp, h = 2.0*math.pi, FOURPI, dx
    for i in range(m - 1):
        a1 = rx; b1 = (e[i] + p*p - rx*rx)/(2*r) + ps[i]*p + sx[i]*rx - tp*r*E[i]; c1 = ps[i]*rx + sx[i]*p - fp*r*J[i]
        r2, x2, p2 = r + 0.5*h*a1, rx + 0.5*h*b1, p + 0.5*h*c1
        a2 = x2; b2 = (em[i] + p2*p2 - x2*x2)/(2*r2) + psm[i]*p2 + sxm[i]*x2 - tp*r2*Em[i]; c2 = psm[i]*x2 + sxm[i]*p2 - fp*r2*Jm[i]
        r3, x3, p3 = r + 0.5*h*a2, rx + 0.5*h*b2, p + 0.5*h*c2
        a3 = x3; b3 = (em[i] + p3*p3 - x3*x3)/(2*r3) + psm[i]*p3 + sxm[i]*x3 - tp*r3*Em[i]; c3 = psm[i]*x3 + sxm[i]*p3 - fp*r3*Jm[i]
        r4, x4, p4 = r + h*a3, rx + h*b3, p + h*c3
        j = i + 1
        a4 = x4; b4 = (e[j] + p4*p4 - x4*x4)/(2*r4) + ps[j]*p4 + sx[j]*x4 - tp*r4*E[j]; c4 = ps[j]*x4 + sx[j]*p4 - fp*r4*J[j]
        r += h/6*(a1 + 2*a2 + 2*a3 + a4); rx += h/6*(b1 + 2*b2 + 2*b3 + b4); p += h/6*(c1 + 2*c2 + 2*c3 + c4)
        Ro.append(r); Po.append(p)
    return np.array(Ro), np.array(Po)


def inject(x, U, amp, d, w):
    """Launch an ingoing phantom pulse of field amplitude amp from |x| = d on both sides.
    Only the *change* in the exterior constraint solution is applied, so nothing but the
    pulse is altered.  Requires data symmetric under x -> -x."""
    n = len(x)
    dx = x[1] - x[0]
    i_s = int(round((d - 3.0*w - x[0])/dx))
    R0, P0 = ext_solve(x, U, i_s)
    add_pulse(x, U, 'phantom', amp, d, w)
    R1, P1 = ext_solve(x, U, i_s)
    dR, dP = R1 - R0, P1 - P0
    U[0][i_s:] += dR; U[4][i_s:] += dP
    U[0][:n - i_s] += dR[::-1]; U[4][:n - i_s] += dP[::-1]


def estimate(R0, PR0, i, lam, order):
    """Growing-mode amplitude from throat readings up to sample i (causal, backward differences).
    y = R0 - 1, subscripts are time derivatives.
      order 1: (y + y_t/lam)/2                exact for growing + decaying mode only
      order 2: (y_tt + lam y_t)/(2 lam^2)     also rejects a constant offset in the throat radius
      order 3: (y_ttt + lam y_tt)/(2 lam^3)   also rejects a steady drift"""
    if order == 1:
        return 0.5*(R0[i] - 1.0 + PR0[i]/lam)
    if i < order:
        return 0.0
    y2 = (PR0[i] - PR0[i-1])/OUT_DT
    if order == 2:
        return (y2 + lam*PR0[i])/(2*lam**2)
    y3 = (PR0[i] - 2*PR0[i-1] + PR0[i-2])/OUT_DT**2
    return (y3 + lam*y2)/(2*lam**3)


def calibrate(d, w, s_amp, a=1e-7, order=1, settle=3.0):
    """Impulse response of the throat to one pulse, from a +/- pair on the grid-balanced wormhole."""
    x = grid(DX, d + 24.0)
    runs = []
    for sgn in (1, -1):
        U = initial_state(x, s_amp=s_amp)
        inject(x, U, sgn*a, d, w)
        runs.append(evolve(x, U, d + 6.0, out_dt=OUT_DT)['hist'])
    n = min(len(runs[0]['t']), len(runs[1]['t']))
    t = runs[0]['t'][:n]
    dR = 0.5*(runs[0]['R0'][:n] - runs[1]['R0'][:n])/a
    dP = 0.5*(runs[0]['PR0'][:n] - runs[1]['PR0'][:n])/a
    win = (t >= d + 3.0) & (t <= d + 5.5)
    lam = float(np.polyfit(t[win], np.log(np.abs(dR[win])), 1)[0])
    k = float(np.mean(dR[win]*np.exp(-lam*(t[win] - d))))
    # estimator's response to a unit pulse launched at t = 0 (the estimator is linear, so apply it to the odd part)
    h = np.array([estimate(dR + 1.0, dP, i, lam, order) for i in range(n)])
    # Bookkeeping for a pulse stops at age d + settle.  k is defined from h at that age so the
    # hand-over is continuous; using the separately fitted radius gain here leaves a mismatch
    # that the predictor then amplifies exponentially.
    cut = int(round((d + settle)/OUT_DT))
    k_fit, k = k, float(h[cut]*math.exp(-lam*settle))
    return {'d': d, 'w': w, 'lam': lam, 'k': k, 'k_radius_fit': k_fit, 't': t[:cut + 1], 'h': h[:cut + 1], 'order': order}


class Feedback:
    def __init__(self, cal, kappa=0.5, a_max=0.02, t_on=0.0, kz=0.0, tau_z=0.0, ctrl_dt=0.25):
        self.d, self.w, self.lam, self.k = cal['d'], cal['w'], cal['lam'], cal['k']
        self.ht, self.h, self.order = cal['t'], cal['h'], cal['order']
        self.kappa, self.a_max, self.t_on, self.kz = kappa, a_max, t_on, kz
        self.zlog = []
        self.tau_z, self.zf, self.ctrl_dt = tau_z, 0.0, ctrl_dt
        self.launch = []                      # (time, amplitude)
        self.sat = 0

    def __call__(self, t, x, U, hist):
        a = self.decide(t, hist)
        if a is not None:
            inject(x, U, a, self.d, self.w)

    def decide(self, t, hist):
        tm = t - self.d                        # what the controller can see left the throat at tm
        if tm < 0 or t < self.t_on:
            return None
        i = min(int(round(tm/OUT_DT)), len(hist['t']) - 1)
        A = estimate(hist['R0'], hist['PR0'], i, self.lam, self.order)
        # neutral (static-family) part of the throat offset: what is left of y after removing e^{+-lam t}.
        # It does not grow, so it must not be multiplied by the delay growth factor; a slow filter tracks it.
        z = (hist['R0'][i] - 1.0) - ((hist['PR0'][i] - hist['PR0'][i-1])/OUT_DT)/self.lam**2 if i > 0 else 0.0
        if self.tau_z > 0:
            self.zf += self.ctrl_dt/self.tau_z*(z - self.zf)
        grow = A - 0.5*self.zf if self.order == 1 else A
        pred = grow*math.exp(2*self.lam*self.d) + self.kz*self.zf   # amplitude when a pulse launched now arrives
        tmax = self.ht[-1]
        for s, a in self.launch:
            age = tm - s                       # how much of this pulse the measurement has already seen
            if age < tmax:
                seen = float(np.interp(age, self.ht, self.h)) if age > 0 else 0.0
                pred += a*(self.k*math.exp(self.lam*(t - s)) - math.exp(2*self.lam*self.d)*seen)
        a = -self.kappa*pred/self.k
        self.zlog.append((t, A, self.zf))
        if abs(a) > self.a_max:
            a = math.copysign(self.a_max, a); self.sat += 1
        self.launch.append((t, a))
        return a


def closed_loop(d=3.0, w=0.5, kappa=0.5, T=100.0, ctrl_dt=0.25, s_amp_cal=None, pulses=(), trimmed=False,
                cal=None, order=1, **kw):
    if s_amp_cal is None:
        b = json.load(open('runs/balance_log%s.json' % TAG)); s_amp_cal = 0.5*(b['lo'] + b['hi'])
    cal = cal or calibrate(d, w, s_amp_cal, order=order)
    x = grid(DX, T + d + 16.0)
    U = initial_state(x, pulses, s_amp=s_amp_cal if trimmed else 0.0)
    fb = Feedback(cal, kappa=kappa, ctrl_dt=ctrl_dt, **kw)
    r = evolve(x, U, T, out_dt=OUT_DT, controller=fb, ctrl_dt=ctrl_dt)
    L = np.array(fb.launch) if fb.launch else np.zeros((0, 2))
    return {'cal': {k: cal[k] for k in ('d', 'w', 'lam', 'k')}, 'order': cal['order'], 'kappa': kappa, 'ctrl_dt': ctrl_dt, 'T': T,
            'outcome': r['outcome'], 'lifetime': r['lifetime'], 'saturated_steps': fb.sat,
            'hist': r['hist'], 'launch_t': L[:, 0], 'launch_a': L[:, 1], 'zlog': np.array(fb.zlog)}


def summary(r, t_from=20.0):
    h = r['hist']; m = h['t'] >= min(t_from, 0.5*h['t'][-1])
    la = r['launch_a'][r['launch_t'] >= min(t_from, 0.5*h['t'][-1])]
    return {'d': r['cal']['d'], 'kappa': r['kappa'], 'order': r['order'], 'outcome': r['outcome'], 'lifetime': float(r['lifetime']),
            'k': r['cal']['k'], 'lam': r['cal']['lam'],
            'max_abs_dR0_late': float(np.abs(h['R0'][m] - 1).max()), 'rms_dR0_late': float(np.sqrt(np.mean((h['R0'][m] - 1)**2))),
            'rms_pulse_amp_late': float(np.sqrt(np.mean(la**2))) if len(la) else float('nan'),
            'max_pulse_amp': float(np.abs(r['launch_a']).max()) if len(r['launch_a']) else 0.0,
            'saturated_steps': r['saturated_steps'], 'max_H': float(h['H'].max()), 'max_M': float(h['M'].max())}


if __name__ == '__main__':
    import time
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    t0 = time.time()
    r = closed_loop(**kw)
    print(json.dumps(summary(r), indent=1), 'wall %.0fs' % (time.time() - t0))
