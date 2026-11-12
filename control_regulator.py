"""Reduced plant model fitted to pulse-response data, and a regulator designed on it.

Identified structure of the throat's response to one phantom pulse (s = time since arrival):
    growing mode      k e^{lam s}                    the instability (lam = 1.1819 /b)
    neutral shift     n_inf - n_r e^{-mu s}          a permanent change of the wormhole (it ends up on a
                                                     neighbouring static solution), approached slowly
    transient         measured directly, gone by s ~ 2
The step-4 controller assumed only the first term; the reduced model with the second term
reproduces its failure (see sysid_report in control_regulator_run.py).
"""
import json
import math
import numpy as np
from control_feedback import OUT_DT, TAG, estimate

import os
# s beyond which the analytic tail replaces the measured response.  2.0 was the original choice; the
# fitted tail is wrong by up to 40% of the loop margin between s = 2 and 5 once it has been through the
# estimator's derivatives, so the measured response is now used out to s = 6.
CUT = float(os.environ.get('WH_CUT', 2.0))
S_TOP = 13.0       # backward recursion for the non-growing kernel starts here, from the fitted tail
PAIR = 1.0         # spacing of the virtual pulse pair used to cancel the growing mode in the data


def dtag(d):
    """File suffix of the pair calibration: distance, and launch time if the pulses were not launched at t = 0."""
    t0 = float(os.environ.get('WH_T0', 0.0))
    return ('' if d == 1.0 else '_d%g' % d) + ('' if t0 == 0.0 else '_t%g' % t0)


def est_series(h, lam, order):
    R0, PR0 = h[:, 0] + 1.0, h[:, 1]
    return np.array([estimate(R0, PR0, i, lam, order) for i in range(len(h))])


def fit_power_tail(d, gap=1.0, win=None):
    """Neutral radius shift of ONE pulse, n(s) = n_inf - c (s + s0)^(-alpha), from the long pair measurement
    (runs/pair_response.npz): pair(t) = n(t - d) - beta n(t - gap - d) once the growing mode is cancelled."""
    win = win or (d + 3.0, d + 16.0)
    P = np.load('runs/pair_response%s%s.npz' % (dtag(d), TAG))
    t, p, beta = P['t'], P['odd'][:, 0], float(P['beta'])
    m = (t >= win[0]) & (t <= win[1])
    sa, sb = t[m] - d, t[m] - gap - d
    best = None
    for alpha in np.arange(0.05, 2.01, 0.025):
        for s0 in np.arange(0.0, 4.01, 0.1):
            Phi = np.array([np.full(m.sum(), 1.0 - beta), -((sa + s0)**(-alpha) - beta*(sb + s0)**(-alpha))]).T
            c, *_ = np.linalg.lstsq(Phi, p[m], rcond=None)
            r = float(np.sqrt(np.mean((Phi @ c - p[m])**2)))
            if best is None or r < best[0]:
                best = (r, alpha, s0, c)
    r, alpha, s0, c = best
    return {'n_inf': float(c[0]), 'c': float(c[1]), 's0': float(s0), 'alpha': float(alpha), 'rms': r, 'beta': beta}


def tail_fns(pl, lam, order):
    """(estimator reading, radius shift) of the neutral tail at time s after arrival, per unit pulse."""
    n = lambda s: pl['n_inf'] - pl['c']*(s + pl['s0'])**(-pl['alpha'])
    n1 = lambda s: pl['c']*pl['alpha']*(s + pl['s0'])**(-pl['alpha'] - 1)
    n2 = lambda s: -pl['c']*pl['alpha']*(pl['alpha'] + 1)*(s + pl['s0'])**(-pl['alpha'] - 2)
    n3 = lambda s: pl['c']*pl['alpha']*(pl['alpha'] + 1)*(pl['alpha'] + 2)*(s + pl['s0'])**(-pl['alpha'] - 3)
    if order == 1:
        return (lambda s: 0.5*(n(s) + n1(s)/lam)), n
    if order == 2:
        return (lambda s: (n2(s) + lam*n1(s))/(2*lam**2)), n
    return (lambda s: (n3(s) + lam*n2(s))/(2*lam**3)), n       # order 3 also rejects a steady drift


def load_model(d, order=1, power_tail=False):
    D = np.load('runs/sysid_d%g%s.npz' % (d, TAG))
    h = D['h'][:, :2]
    t = OUT_DT*np.arange(len(h)); s = t - d
    n = int(round(PAIR/OUT_DT))
    win = (s >= PAIR + 1.5) & (s <= 6.3)

    def pair(x, lam):
        p = x.copy(); p[n:] -= math.exp(lam*PAIR)*x[:-n]; return p

    def gcoef(lam):                      # leftover growing-mode content of the pair response
        Phi = np.array([np.exp(lam*s[win]), np.ones(win.sum()), s[win]]).T
        return np.linalg.lstsq(Phi, pair(h[:, 0], lam)[win], rcond=None)[0][0]
    lo, hi = 1.17, 1.19
    for _ in range(60):
        mid = 0.5*(lo + hi)
        lo, hi = (lo, mid) if gcoef(lo)*gcoef(mid) <= 0 else (mid, hi)
    lam = 0.5*(lo + hi)
    lam_fit = lam
    if power_tail and os.environ.get('WH_LAM', 'beta') == 'beta':
        # The tuned cancelling ratio IS the growth factor over the pair spacing, to much better precision
        # than the fit above.  The measured kernel k e^{lam s} - E(s) needs lam to ~1e-6 once s reaches 6.
        Pp = np.load('runs/pair_response%s%s.npz' % (dtag(d), TAG))
        lam = math.log(float(Pp['beta']))/PAIR

    def tail(x):
        """x(s) = growing + n_inf - n_r e^{-mu s}: fit the non-growing part from the pair response."""
        p = pair(x, lam)[win]
        best = None
        for mu in np.arange(0.05, 1.5, 0.01):
            Phi = np.array([np.ones(win.sum()), np.exp(-mu*s[win])]).T
            c, *_ = np.linalg.lstsq(Phi, p, rcond=None)
            r = float(np.sqrt(np.mean((Phi @ c - p)**2)))
            if best is None or r < best[0]:
                best = (r, mu, c)
        r, mu, c = best
        return {'n_inf': float(c[0]/(1 - math.exp(lam*PAIR))), 'n_r': float(-c[1]/(1 - math.exp((lam + mu)*PAIR))),
                'mu': float(mu), 'rms': r}
    E = est_series(h, lam, order)
    tE, tR = tail(E), tail(h[:, 0])
    cut = int(round((d + CUT)/OUT_DT))
    ng = lambda ss, T: T['n_inf'] - T['n_r']*np.exp(-T['mu']*ss)
    pl = fit_power_tail(d) if power_tail else None
    k = float((E[cut] - (tail_fns(pl, lam, order)[0](CUT) if pl else ng(CUT, tE)))*math.exp(-lam*CUT))
    b = k*np.exp(lam*s[:cut + 1]) - E[:cut + 1]          # growing part minus what the estimator shows
    return {'d': float(d), 'w': float(D['w']), 'order': order, 'lam': float(lam), 'lam_fit': float(lam_fit), 'k': k,
            'tail_E': tE, 'tail_R': tR, 'power_tail': pl, 't': t[:cut + 1], 'b': b, 'h': h[:cut + 1],
            'kernel': ng_table(d, order, pl) if (pl and os.environ.get('WH_KERNEL', 'pair') == 'pair') else None}


def ng_table(d, order, pl):
    """Non-growing part of the estimator's response to one pulse, ng(s), s = time since arrival.

    The tuned pulse pair cancels the growing mode physically, so its estimator reading is exactly
        q(s) = ng(s) - beta ng(s - 1),      beta = growth factor over the pair spacing.
    Solving BACKWARD from large s, ng(s - 1) = (ng(s) - q(s))/beta, shrinks errors by 1/beta per step and
    needs neither the growth rate nor the growing gain to high precision.  (Subtracting k e^{lam s} from a
    single-pulse response instead needs lam to ~1e-6, which the data cannot give.)
    Before arrival ng(s) = -k beta^s, which defines the growing gain k."""
    P = np.load('runs/pair_response%s%s.npz' % (dtag(d), TAG))
    beta = float(P['beta']); lam = math.log(beta)/PAIR
    q = est_series(P['odd'][:, :2], lam, order)
    step = int(round(PAIR/OUT_DT))
    i_top = int(round((d + S_TOP)/OUT_DT)); i_min = -int(round((d + 1.0)/OUT_DT))     # tau index range
    tail_E = tail_fns(pl, lam, order)[0]
    ng = {}
    for i in range(i_top + step, i_top, -1):
        ng[i] = float(tail_E(i*OUT_DT - d))
    for i in range(i_top, i_min - 1, -1):
        j = i + step
        ng[i] = (ng[j] - (q[j] if j >= 0 else 0.0))/beta
    idx = np.arange(i_min, i_top + 1)
    s = idx*OUT_DT - d
    vals = np.array([ng[i] for i in idx])
    k = float(-vals[idx == 0][0]*beta**(d/PAIR))
    return {'s': s, 'ng': vals, 'beta': beta, 'lam': lam, 'k': k}


def radius_loop_gains(m, kz=0.06, ti=50.0):
    """Radius-loop gains for a chosen closed-loop rate kz (per b).
    Holding the growing mode at set-point G* takes pulses at rate -lam G*/k, and each unit of pulse amplitude
    shifts the radius by about n = n(5 b); so z' = -(n lam/k) G*, and G* = gp z gives z' = -kz z for
    gp = kz k/(n lam).  The integral gain is gp/ti."""
    lam = m['kernel']['lam'] if m.get('kernel') else m['lam']
    k = m['kernel']['k'] if m.get('kernel') else m['k']
    n5 = float(tail_fns(m['power_tail'], lam, m['order'])[1](5.0))
    gp = kz*k/(n5*lam)
    return gp, gp/ti


class Regulator:
    """Predictor feedback on the identified model.

    The estimator reading E is corrected for the known non-growing response of every pulse this
    controller has launched (measured transient, then the fitted neutral tail).  Only what is left,
    the genuinely growing part G, is extrapolated across the loop delay 2d and cancelled.
    gz > 0 gives G a small set-point proportional to the accumulated throat-radius shift, which
    walks the radius back and absorbs a constant disturbance (integral action).
    """
    def __init__(self, m, kappa=0.5, gz=0.0, gp=0.0, gv=0.0, a_max=0.02, tm_on=0.0, gi=0.0, tau_f=0.0, ctrl_dt=0.25, tau_b=0.0):
        self.m, self.kappa, self.gz, self.a_max = m, kappa, gz, a_max
        self.gp, self.gv = gp, gv
        self.tm_on = tm_on          # ignore readings taken before this time (start-up transient of the initial data)
        # Radius loop: the growing mode is given a small set-point gp*z + gi*integral(z), z = (filtered) measured
        # radius offset.  Holding the mode at a positive set-point takes a steady stream of negative pulses, and
        # each pulse carries a permanent neutral shift, so the radius is walked back toward its calibration value.
        self.gi, self.tau_f, self.ctrl_dt, self.zf, self.zint = gi, tau_f, ctrl_dt, None, 0.0
        # Bias rejection.  After the known pulse responses are removed, the corrected reading should obey
        # G(t + dt) = e^{lam dt} (G(t) + k a/amp).  Whatever part of it does not grow like that is a slowly
        # varying misreading (offset or drift seen by a low-order estimator); it is tracked with time constant
        # tau_b and subtracted before the reading is extrapolated across the delay.  This lets a low-order,
        # spike-free estimator be used without multiplying its misreadings by the delay growth.
        self.tau_b, self.bias, self.Gprev = tau_b, 0.0, None
        self.launch, self.sat, self.log = [], 0, []
        self.amp = math.exp(2*m['lam']*m['d'])
        self.tail = tail_fns(m['power_tail'], m['lam'], m['order']) if m.get('power_tail') else None
        self.K = m.get('kernel')
        if self.K:
            self.amp = self.K['beta']**(2*m['d']/PAIR)

    def decide(self, t, hist):
        m = self.m
        lam, d, T, TR = m['lam'], m['d'], m['tail_E'], m['tail_R']
        tm = t - d
        if tm < (max(0.0, self.tm_on - 3*self.tau_b) if self.tau_b > 0 else max(0.0, self.tm_on)):
            return None
        i = min(int(round(tm/OUT_DT)), len(hist['t']) - 1)
        K = self.K
        if K:
            lam = K['lam']
        E = estimate(hist['R0'], hist['PR0'], i, lam, m['order'])
        G, shift = E, 0.0
        for s0, a in (self.launch if K else ()):
            ss = tm - s0 - d
            if ss > K['s'][-1]:
                G -= a*self.tail[0](ss)
            elif ss < K['s'][0]:
                G -= a*K['ng'][0]*K['beta']**((ss - K['s'][0])/PAIR)
            else:
                G -= a*float(np.interp(ss, K['s'], K['ng']))
            if ss >= 2.0:
                shift += a*self.tail[1](ss)
        for s0, a in (() if K else self.launch):
            age = tm - s0
            if age < 0:
                G += a*m['k']*math.exp(lam*(age - d))
            elif age <= m['t'][-1]:
                G += a*float(np.interp(age, m['t'], m['b']))
            else:
                ss = age - d
                if self.tail:
                    G -= a*self.tail[0](ss); shift += a*self.tail[1](ss)
                else:
                    G -= a*(T['n_inf'] - T['n_r']*math.exp(-T['mu']*ss))
                    shift += a*(TR['n_inf'] - TR['n_r']*math.exp(-TR['mu']*ss))
        kk = K['k'] if K else m['k']
        if self.tau_b > 0:
            if self.Gprev is not None:
                g = math.exp(lam*self.ctrl_dt)
                b_inst = (G - g*self.Gprev)/(1.0 - g)
                self.bias += self.ctrl_dt/self.tau_b*(b_inst - self.bias)
            Graw = G
            G = G - self.bias
        z = hist['R0'][i] - 1.0
        self.zf = z if (self.zf is None or self.tau_f <= 0) else self.zf + self.ctrl_dt/self.tau_f*(z - self.zf)
        self.zint += self.ctrl_dt*self.zf
        target = self.gz*shift + self.gp*self.zf + self.gi*self.zint + self.gv*hist['PR0'][i]
        a = -self.kappa*(self.amp*G - target)/kk
        if abs(a) > self.a_max:
            a = math.copysign(self.a_max, a); self.sat += 1
        if tm < self.tm_on:
            a = 0.0                      # bias tracker is warming up; no control yet
        if self.tau_b > 0:
            self.Gprev = Graw + a*kk/self.amp
            if a == 0.0 and tm < self.tm_on:
                self.log.append((t, E, G, shift)); return None
        self.launch.append((t, a))
        self.log.append((t, E, G, shift))
        return a

    def __call__(self, t, x, U, hist):
        from control_feedback import inject
        a = self.decide(t, hist)
        if a is not None:
            inject(x, U, a, self.m['d'], self.m['w'])


def full_run(m, reg, T, trimmed=False, pulses=(), ctrl_dt=0.25):
    """Closed loop in the full Einstein-scalar simulation."""
    from control_solver import grid, initial_state, evolve
    from control_feedback import DX
    s = 0.0
    if trimmed:
        b = json.load(open('runs/balance_log%s.json' % TAG)); s = 0.5*(b['lo'] + b['hi'])
    x = grid(DX, T + m['d'] + 16.0)
    r = evolve(x, initial_state(x, pulses, s_amp=s), T, out_dt=OUT_DT, controller=reg, ctrl_dt=ctrl_dt)
    L = np.array(reg.launch) if reg.launch else np.zeros((0, 2))
    h = r['hist']
    return {'outcome': r['outcome'], 'lifetime': float(r['lifetime']), 'tau': float(r['tau']), 't': h['t'], 'R0': h['R0'],
            'PR0': h['PR0'], 'S0': h['S0'], 'H': h['H'], 'M': h['M'], 'launch_t': L[:, 0], 'launch_a': L[:, 1],
            'saturated': reg.sat, 'log': np.array(reg.log)}


# ---- two-term model used to confirm the cause of the step-4 failure ---------------------------

def simple_model(d, settle=3.0):
    """Growing mode + constant neutral shift (+ measured transient up to s = settle)."""
    D = np.load('runs/sysid_d%g%s.npz' % (d, TAG))
    h = D['h'][:, :2]
    t = OUT_DT*np.arange(len(h))
    lam = load_model(d)['lam']
    win = (t >= d + 2.0) & (t <= d + 7.0)
    Phi = np.array([np.exp(lam*(t[win] - d)), np.ones(win.sum())]).T
    k = float(np.linalg.lstsq(Phi, h[win, 0], rcond=None)[0][0])
    cut = int(round((d + settle)/OUT_DT))
    plateau = h[cut] - np.array([k, lam*k])*math.exp(lam*settle)
    return {'d': float(d), 'lam': lam, 'k': k, 'plateau': plateau, 't': t[:cut + 1], 'h': h[:cut + 1], 'base': D['base'][:, :2]}


def simple_response(m, age):
    age = np.asarray(age, dtype=float)
    out = np.zeros(age.shape + (2,))
    late = age > m['t'][-1]
    g = np.exp(m['lam']*(np.where(late, age, 0.0) - m['d']))
    for j, gain in enumerate((m['k'], m['lam']*m['k'])):
        inner = np.where((age >= 0) & ~late, np.interp(age, m['t'], m['h'][:, j]), 0.0)
        out[..., j] = np.where(late, gain*g + m['plateau'][j], inner)
    return out


def run_on_model(m, controller, T, ctrl_dt=0.25):
    """Closed loop with the two-term model as the plant: uncontrolled baseline + sum of pulse responses."""
    n = min(int(round(T/OUT_DT)) + 1, len(m['base']))
    hist = {'t': [], 'R0': [], 'PR0': []}
    ls, la = [], []
    every = int(round(ctrl_dt/OUT_DT))
    for i in range(n):
        t = i*OUT_DT
        if i % every == 0 and hist['t']:
            a = controller.decide(t, hist)
            if a is not None:
                ls.append(t); la.append(a)
        y = m['base'][i].copy()
        if ls:
            y += (np.array(la)[:, None]*simple_response(m, t - np.array(ls))).sum(axis=0)
        hist['t'].append(t); hist['R0'].append(1.0 + y[0]); hist['PR0'].append(y[1])
        if not 0.5 < 1.0 + y[0] < 2.0:
            break
    return {'lifetime': hist['t'][-1], 't': np.array(hist['t']), 'R0': np.array(hist['R0']),
            'launch_t': np.array(ls), 'launch_a': np.array(la)}
