"""Transit study, step 1: feedforward for a known traveller.

The throat's response to a symmetric normal-matter traveller is linear in the traveller's
Misner-Sharp mass dM (checked: response per unit mass equal to 5 digits for dM = 6e-8 and 2.4e-7).
Its growing-mode part is  KT * dM * exp(lam * (t - t_arr)),  KT < 0 (toward collapse), with
t_arr = launch distance (the pulse centre moves at coordinate speed 1).

Feedforward: one phantom pulse, launched from the controller at t_arr - d so that it reaches the
throat with the traveller, of amplitude  u_ff = c * dM  with  c = -KT / k  (k = actuator growing gain).
Cancelling the growing kick alone does not help (mode 'ff' in the scan: the regulator
still over-reacts to the traveller's transient, which it reads as growing mode).  So the traveller's
response together with its feedforward pulse (growing parts cancel, so it is bounded) is measured
and subtracted from the readings before the filter sees them:

    python3 transit_ff.py calib         # open loop: traveller + one actuator pulse tuned to cancel the growing
                                        # mode to round-off; gives c and the non-growing signature -> runs/traveller_response.npz
    python3 transit_ff.py scan [T] [masses]   # plain LQG / feedforward only / feedforward + signature -> results_transit_ff_scan.json

Travellers are launched from |x| = X0 = 20 (WH_X0), so they arrive at t = 20; the calibration needs the
open-loop balanced wormhole to survive the measurement window, which rules out a later arrival.
"""
import json
import math
import os
import sys
from multiprocessing import Pool
import numpy as np
from control_solver import grid, initial_state, evolve
from control_feedback import OUT_DT, TAG, DX
from control_lqg import build_model, design, LQG, DT
from control_lqg_run import TUNE

D = 1.0
X0, WIDTH = float(os.environ.get('WH_X0', 20.0)), 1.0
S_LO, S_HI = -3.0, 8.0            # range of time-since-arrival over which the traveller signature is measured
M_CAL = float(os.environ.get('WH_MCAL', 1e-4))   # traveller mass used for calibration (also run at 4x)
CAL_TAG = '' if M_CAL == 1e-4 else '_m%g' % M_CAL


def mass(x, U, i):
    """Misner-Sharp mass at grid index i.  (control_steps123.mass hard-codes dx = 0.025.)"""
    dx = x[1] - x[0]
    R, S, PR = U[0], U[1], U[4]
    Rx = (R[i+1] - R[i-1])/(2*dx)
    return 0.5*R[i]*(1.0 + np.exp(-2*S[i])*(PR[i]**2 - Rx**2))


def traveller(x, s_amp, dM_target):
    """Initial data with a symmetric traveller of (very nearly) the requested Misner-Sharp mass."""
    i_m = int(round((X0 + 8.0 - x[0])/DX))
    U0 = initial_state(x, s_amp=s_amp)
    if not dM_target:
        return U0, 0.0
    m0 = mass(x, U0, i_m)
    a = 1e-4
    m1 = mass(x, initial_state(x, [('normal', a, X0, WIDTH)], s_amp=s_amp), i_m) - m0      # mass is quadratic in amplitude
    amp = a*math.sqrt(dM_target/m1)
    U = initial_state(x, [('normal', amp, X0, WIDTH)], s_amp=s_amp)
    return U, float(mass(x, U, i_m) - m0)


def s_amp_balanced():
    b = json.load(open('runs/balance_log%s.json' % TAG))
    return 0.5*(b['lo'] + b['hi'])


def load_signature():
    """Calibration output plus kern (response to one unit actuator pulse) and comb (traveller + mean-c pulse).
    The controller rebuilds comb per traveller with its own mass-dependent c, so the subtracted response
    always matches the pulse actually launched."""
    from control_lqg import raw_kernel
    f = dict(np.load('runs/traveller_response_x%g%s%s.npz' % (X0, CAL_TAG, TAG)))
    K = raw_kernel(D)
    if 'c1' in f:
        # The tuned pulse per unit mass falls with mass (coarse grid: 0.404 at 1e-4 b, 0.380 at 5e-3, 0.329 at
        # 2e-2).  Keep both points; the controller interpolates linearly in mass (extrapolating beyond).
        f['c_points'] = np.array([[float(f['dM1']), float(f['c1'])], [float(f['dM4']), float(f['c4'])]])
    f['kern'] = np.array([np.where(f['s'] < K['s'][0], 0.0, np.interp(f['s'], K['s'], K['ng'][ch])) for ch in (0, 1)])   # one actuator pulse
    f['comb'] = f['ng'] + float(f['c'])*f['kern']
    return f


class TransitLQG(LQG):
    """LQG regulator plus a feedforward pulse for each announced traveller (t_arr, dM)."""

    def __init__(self, m, g, travellers=(), c=0.0, a_max=0.02, sig=None):
        """c: feedforward pulse amplitude per unit traveller mass (0: none).
        sig: calibration from load_signature(), or None.  With sig, the readings that the traveller and
        its feedforward pulse are known to cause together are subtracted before the filter sees them, and
        the feedforward pulse is kept out of the filter's model (it is already accounted for)."""
        super().__init__(m, g, a_max=a_max)
        self.travellers, self.c, self.sig = list(travellers), c, sig
        self.ff, self.fb = [], []

    def c_of(self, dM):
        if self.sig is not None and 'c_points' in self.sig and os.environ.get('WH_CMODE', 'interp') == 'interp':
            (m1, c1), (m4, c4) = self.sig['c_points']
            return CSCALE*(c1 + (c4 - c1)*(dM - m1)/(m4 - m1))
        return CSCALE*self.c

    def known(self, t):
        """Readings that the travellers and their feedforward pulses are known to cause at received time t."""
        y = np.zeros(2)
        if self.sig is None:
            return y
        for (ta, dM) in self.travellers:
            sr = t - self.m['d'] - ta
            if sr >= S_LO:
                comb = self.sig['ng'] + self.c_of(dM)*self.sig['kern']           # traveller + its feedforward pulse
                y[0] += dM*np.interp(sr, self.sig['s'], comb[0])                 # held at its last value afterwards
                y[1] += dM*(np.interp(sr, self.sig['s'], comb[1]) if sr <= S_HI else 0.0)
        return y

    def decide(self, t, hist):
        m, g = self.m, self.g
        d = m['d']
        tm = t - d
        if tm < self.tm_on:
            return None
        i = min(int(round(tm/OUT_DT)), len(hist['t']) - 1)
        y = np.array([hist['R0'][i] - 1.0, hist['PR0'][i]]) - self.known(t)
        if not self.launch:
            self.xbar[m['idx']['Z']] = y[0]; self.xbar[m['idx']['V']] = y[1]
        xh = self.xbar + g['Mk'] @ (y - m['C'] @ self.xbar)
        u_fb = float(-(g['K'] @ xh)[0])
        if abs(u_fb) > self.a_max:
            u_fb = math.copysign(self.a_max, u_fb); self.sat += 1
        u_ff = sum(self.c_of(dM)*dM for (ta, dM) in self.travellers if abs(t - (ta - d)) < 0.5*DT)
        u = u_fb + u_ff
        self.xbar = m['A'] @ xh + m['B'][:, 0]*(u_fb if self.sig is not None else u)
        ix = m['idx']
        self.launch.append((t, u)); self.fb.append((t, u_fb)); self.ff.append((t, u_ff))
        self.log.append((t, xh[ix['G']], xh[ix['Z']], xh[ix['F']], xh[ix['V']]))
        return u


def one(args):
    dM_target, mode, T, a_max = args                     # mode: 'plain' | 'ff' | 'ff+sig'
    x = grid(DX, T + X0 + 16.0)
    U, dM = traveller(x, s_amp_balanced(), dM_target)
    m = build_model(D)
    sig = load_signature() if mode != 'plain' else None
    ctl = TransitLQG(m, design(m, **TUNE), travellers=[(X0, dM)] if dM else [], c=float(sig['c']) if sig is not None else 0.0,
                     a_max=a_max, sig=sig if mode == 'ff+sig' else None)
    r = evolve(x, U, T, out_dt=OUT_DT, controller=ctl, ctrl_dt=DT)
    h = r['hist']; t = h['t']; dR = h['R0'] - 1.0
    fb = np.array(ctl.fb); ff = np.array(ctl.ff)
    after = fb[:, 0] >= X0 - D
    late = fb[:, 0] >= r['lifetime'] - 15.0
    return {'traveller_mass': dM, 'mode': mode, 'a_max': a_max, 'outcome': r['outcome'], 'lifetime': float(r['lifetime']),
            'max_H': float(np.nanmax(h['H'])), 'max_M': float(np.nanmax(h['M'])),
            'max_offset_after_arrival': float(np.abs(dR[t >= X0 - 2]).max()), 'final_offset': float(dR[-1]),
            'late_drift_per_b': float(np.polyfit(t[t >= T - 20], dR[t >= T - 20], 1)[0]) if r['outcome'] == 'alive' else None,
            'saturated': int(ctl.sat), 'ff_pulse': float(ff[:, 1].sum()),
            'peak_feedback_pulse_after_arrival': float(np.abs(fb[after, 1]).max()) if after.any() else None,
            'rms_feedback_pulse_last15': float(np.sqrt(np.mean(fb[late, 1]**2))),
            'sum_abs_pulse': float(np.abs(fb[:, 1]).sum() + np.abs(ff[:, 1]).sum())}


class OnePulse:
    """Open loop: a single actuator pulse of amplitude a launched at t_launch."""

    def __init__(self, a, t_launch):
        self.a, self.tl, self.done = a, t_launch, False

    def __call__(self, t, x, U, hist):
        from control_feedback import inject
        from sysid_pair import W
        if not self.done and abs(t - self.tl) < 0.5*DT:
            inject(x, U, self.a, D, W); self.done = True


T_CAL = X0 + 25.0
CSCALE = float(os.environ.get('WH_CSCALE', 1.0))   # deliberate feedforward gain error, for robustness tests
C_LO = float(os.environ.get('WH_CLO', 0.2))   # lower end of the search bracket (must collapse); the fine-grid
                                             # balanced wormhole leans toward expansion, so use 0 there


# The grid-balanced start is balanced only on the domain its trim was tuned on (initial data differ at
# 1e-11 between domain sizes, and the unstable mode amplifies that to 1e-3 by t ~ 20).  Calibrate there:
# coarse trim on L = 80 (control_steps123), fine trim on L = 62 (rescheck.py).  On the coarse grid L = 81
# and 80 give the same baseline (expands at 33.76 b both).
CAL_L = {0.025: T_CAL + X0 + 16.0, 0.0125: 62.0}.get(DX, T_CAL + X0 + 16.0)


def cal_run(c, mass_target=M_CAL):
    x = grid(DX, CAL_L)
    U, dM = traveller(x, s_amp_balanced(), mass_target)
    r = evolve(x, U, T_CAL, out_dt=OUT_DT, controller=OnePulse(c*dM, X0 - D), ctrl_dt=DT)
    return c, r['outcome'], r['lifetime'], r['hist']['t'], r['hist']['R0'], r['hist']['PR0'], dM


def cal_run4(c):
    return cal_run(c, 4.0*M_CAL)


def show(rows):
    for r in rows:
        print('dM %.3e %-7s %-8s at %5.1f  max|dR| %.2e  final dR %+.2e  drift %s  ff %.2e  peak fb %.2e  late fb rms %.1e  sat %d  maxH %.1e' % (
            r['traveller_mass'], r['mode'], r['outcome'], r['lifetime'], r['max_offset_after_arrival'], r['final_offset'], ('%+.1e' % r['late_drift_per_b']) if r['late_drift_per_b'] is not None else '-', r['ff_pulse'],
            r['peak_feedback_pulse_after_arrival'] or 0, r['rms_feedback_pulse_last15'], r['saturated'], r['max_H']))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'calib':
        from control_lqg import raw_kernel
        from control_steps123 import section_search
        # Two tuned runs, traveller masses M_CAL and 4 M_CAL.  Their difference removes the balanced
        # wormhole's own drift and round-off seed (an untuned baseline run diverges after ~25 b).
        with Pool(2) as pool:
            lo1, hi1, log1 = section_search(cal_run, C_LO, 0.7, pool, iters=14)
            lo4, hi4, log4 = section_search(cal_run4, C_LO, 0.7, pool, iters=14)
        c1, _, life1, t, R1, P1, dM1 = max(log1, key=lambda r: r[2])
        c4, _, life4, t4, R4, P4, dM4 = max(log4, key=lambda r: r[2])
        K = raw_kernel(D)
        n = min(len(t), len(t4))
        s = t[:n] - X0
        w = (s >= S_LO) & (s <= S_HI)
        ngp = [np.where(s < K['s'][0], 0.0, np.interp(s, K['s'], K['ng'][ch])) for ch in (0, 1)]
        ng = np.array([(((Y4[:n] - Y1[:n]) - (c4*dM4 - c1*dM1)*ngp[ch])/(dM4 - dM1))[w] for ch, (Y1, Y4) in enumerate(((R1, R4), (P1, P4)))])
        c = 0.5*(c1 + c4)
        np.savez('runs/traveller_response_x%g%s%s.npz' % (X0, CAL_TAG, TAG), s=s[w], ng=ng, c=c, kT=-c*K['k'][0], c1=c1, c4=c4, dM1=dM1, dM4=dM4,
                 brackets=[lo1, hi1, lo4, hi4], lifetimes=[life1, life4])
        print('c = %.7f at mass %.2e, %.7f at %.2e (linearity %.1e); kT = %.5f; tuned lifetimes %.1f, %.1f' % (
            c1, dM1, c4, dM4, c4/c1 - 1, -c*K['k'][0], life1, life4))
        for sv in (-3, -2, -1, 0, 1, 2, 3, 4, 6, 8):
            j = int(np.argmin(np.abs(s[w] - sv)))
            print('   s = %4.1f: radius signature %+.4f, rate signature %+.4f  (per unit traveller mass)' % (sv, ng[0][j], ng[1][j]))
    elif mode == 'openloop':
        # The best open-loop answer: one actuator pulse tuned to round-off for this traveller (the calibration's
        # own runs), continued to T.  How long after arrival does it hold, and how sensitive is it to the pulse?
        T = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
        sig = dict(np.load('runs/traveller_response_x%g%s%s.npz' % (X0, CAL_TAG, TAG)))
        c4, dM4 = float(sig['c4']), float(sig['dM4'])

        def ol(scale):
            x = grid(DX, CAL_L if T <= T_CAL else T + X0 + 16.0)
            U, dM = traveller(x, s_amp_balanced(), dM4)
            r = evolve(x, U, T, out_dt=OUT_DT, controller=OnePulse(scale*c4*dM, X0 - D), ctrl_dt=DT)
            return scale, r['outcome'], float(r['lifetime'])
        with Pool(2) as pool:
            rows = pool.map(ol, [1.0, 1 - 1e-6, 1 + 1e-6, 0.99, 1.01, 0.9, 1.1])
        for sc, oc, lt in rows:
            print('open loop, mass %.2e, pulse x %.7f: %s at %.1f b (%.1f b after arrival)' % (dM4, sc, oc, lt, lt - X0))
        json.dump({'mass': dM4, 'c_tuned': c4, 'rows': [{'scale': a, 'outcome': b, 'lifetime': c} for a, b, c in rows]},
                  open('results_transit_openloop%s%s.json' % (CAL_TAG, TAG), 'w'), indent=1)
    elif mode == 'scan':
        T = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
        masses = [float(v) for v in sys.argv[3:]] or [1e-3, 1e-2, 3e-2, 1e-1]
        with Pool(2) as pool:
            rows = pool.map(one, [(dm, md, T, 0.1) for dm in masses for md in os.environ.get('WH_MODES', 'plain,ff,ff+sig').split(',')], chunksize=1)
        sig = load_signature()
        json.dump({'d': D, 'x0': X0, 'dx': DX, 'T': T, 'cal_mass': M_CAL, 'c': float(sig['c']), 'c_mode': os.environ.get('WH_CMODE', 'interp'), 'kT': float(sig['kT']), 'rows': rows},
                  open('results_transit_ff_scan%s%s.json' % (CAL_TAG, TAG), 'w'), indent=1)
        show(rows)
