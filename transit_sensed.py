"""Transit study, Stage 3b: a traveller the controller is not told about.

In Stage 2 and Stage 3a the controller is handed each traveller's mass and arrival time in advance.
Here it has only a detector: the energy flux of the normal field through the sphere at |x| = XS,
read locally and relayed to the controller at d at light speed (relay delay XS - d).

Why this can work on time.  A slice of the traveller passing XS at time tau reaches the throat at
about tau + XS.  Its reading reaches the controller at tau + XS - d, and a pulse launched then from d
reaches the throat at tau + XS: the same moment.  So the feedforward is streamed, slice by slice, as
the traveller passes the detector, with no advance notice and no assumed pulse shape.  Each slice is
scaled by exp(lam * lateness) for the part of a control step by which its pulse is late.

The traveller's own non-growing signature (Stage 2 calibration) is subtracted using the mass and
arrival time measured so far; whenever those estimates change, the Kalman filter is re-run from an
anchor placed before any reading the traveller can have touched, so early readings are corrected
after the fact.  The responses of the streamed pulses are subtracted exactly (they are known).

    python3 transit_sensed.py sensorcal                 # detector calibration: mass per unit flux, timing offset
    python3 transit_sensed.py scan [T] [masses]         # results_transit_sensed_scan[...].json

WH_SIDES=1 (default here) for one-sided travellers, 2 for the Stage 2 symmetric pairs.
"""
import os
os.environ.setdefault('WH_SIDES', '1')
import json
import math
import sys
from multiprocessing import Pool
import numpy as np
import transit_ff as TF
from control_solver import grid, evolve
from control_feedback import OUT_DT, TAG, DX
from control_lqg import build_model, design, DT
from control_lqg_run import TUNE

XS = float(os.environ.get('WH_XS', 2.0))       # detector radius (coordinate x), must be > d
SUB = 5                                         # detector samples per control step
SCAL = 'runs/sensor_cal_x%g_xs%g%s%s%s.json' % (TF.X0, XS, '' if TF.SIDES == 2 else '_1side', '_track' if TF.TRACK else '', TAG)


def inflow(x, U, i_p, i_m):
    """Ingoing energy flux of the normal field through both detector spheres (flat-space weighting)."""
    dx = x[1] - x[0]
    R, G, PG = U[0], U[3], U[7]
    gp = (G[i_p + 1] - G[i_p - 1])/(2*dx)
    gm = (G[i_m + 1] - G[i_m - 1])/(2*dx)
    return 4*np.pi*(R[i_p]**2*PG[i_p]*gp - R[i_m]**2*PG[i_m]*gm)


class Detector:
    def __init__(self):
        self.t, self.f = [], []

    def read(self, t, x, U):
        i_p = int(round((XS - x[0])/DX)); i_m = len(x) - 1 - i_p
        self.t.append(t); self.f.append(float(inflow(x, U, i_p, i_m)))

    def passage(self, t_upto):
        """Energy passed inward and its flux-weighted mean time, from samples up to t_upto."""
        t = np.array(self.t); f = np.clip(np.array(self.f), 0.0, None)
        sel = t <= t_upto + 1e-9
        t, f = t[sel], f[sel]
        if len(t) < 2:
            return 0.0, None, t, f
        w = np.empty_like(t); w[1:-1] = 0.5*(t[2:] - t[:-2]); w[0] = 0.5*(t[1] - t[0]); w[-1] = 0.5*(t[-1] - t[-2])
        E = float((f*w).sum())
        return E, (float((t*f*w).sum()/E) if E > 0 else None), t, f*w


class SensedLQG(TF.TransitLQG):
    def __init__(self, m, g, sig, cal, a_max=0.1, stream=True):
        super().__init__(m, g, travellers=[], c=float(sig['c']), a_max=a_max, sig=sig)
        self.cal, self.stream = cal, stream
        self.det = Detector()
        self.k = 0
        self.used = 0.0                # detector time up to which flux has been turned into feedforward
        self.stored = []               # (t, y_raw, u_fb) for every decision after the anchor
        self.anchor = None             # (index into stored, xbar at that point)
        self.ff_launch = []            # (t, amplitude) of streamed feedforward pulses
        self.est = (0.0, None)
        self.frozen = False

    def __call__(self, t, x, U, hist):
        self.det.read(t, x, U)
        if self.k % SUB == 0:
            from sysid_pair import W
            a = self.decide(t, hist)
            if a is not None:
                TF.launch(x, U, a, self.m['d'], W, t, hist)
        self.k += 1

    def known(self, t):
        """Readings the traveller (with the current mass and arrival estimates) and the streamed pulses cause."""
        y = np.zeros(2)
        dM, ta = self.est
        s_g = self.sig['s']
        if ta is not None and dM > 0:
            sr = t - self.m['d'] - ta
            if sr >= TF.S_LO:
                y[0] += dM*np.interp(sr, s_g, self.sig['ng'][0])
                y[1] += dM*(np.interp(sr, s_g, self.sig['ng'][1]) if sr <= TF.S_HI else 0.0)
        d = self.m['d']
        for (tl, a) in self.ff_launch:
            sk = t - d - (tl + d)
            if sk >= TF.S_LO:
                y[0] += a*np.interp(sk, s_g, self.sig['kern'][0])
                y[1] += a*(np.interp(sk, s_g, self.sig['kern'][1]) if sk <= TF.S_HI else 0.0)
        return y

    def step(self, xbar, y, u_fb):
        m, g = self.m, self.g
        xh = xbar + g['Mk'] @ (y - m['C'] @ xbar)
        return xh, m['A'] @ xh + m['B'][:, 0]*u_fb

    def decide(self, t, hist):
        m, g = self.m, self.g
        d = m['d']
        if t - d < self.tm_on:
            return None
        # detector data that can have reached the controller by now
        t_av = t - (XS - d)
        E, tc, ts, fw = self.det.passage(t_av)
        lam = m['lam']
        u_ff = 0.0
        if E > 0:
            dM = self.cal['kappa']*E
            self.est = (dM, tc + XS + self.cal['delta'])
            if self.anchor is None:
                self.anchor = (len(self.stored), self.xbar.copy())
            if self.stream:
                new = ts > self.used + 1e-9
                if new.any():
                    # each slice's pulse should reach the throat when the slice does: slice at detector time tau
                    # reaches the throat at tau + XS + delta; this pulse arrives at t + d
                    late = (t + d) - (ts[new] + XS + self.cal['delta'])
                    u_ff = self.c_of(dM)*self.cal['kappa']*float((fw[new]*np.exp(lam*np.clip(late, 0.0, None))).sum())
                    self.used = float(ts[new][-1])
        i = min(int(round((t - d)/OUT_DT)), len(hist['t']) - 1)
        y_raw = np.array([hist['R0'][i] - 1.0, hist['PR0'][i]])
        if not self.launch:
            self.xbar[m['idx']['Z']] = y_raw[0]; self.xbar[m['idx']['V']] = y_raw[1]
        if self.anchor is not None and not self.frozen:
            # re-run the filter from the anchor with the current estimates
            j0, xb = self.anchor
            for (tt, yr, uf) in self.stored[j0:]:
                _, xb = self.step(xb, yr - self.known(tt), uf)
            self.xbar = xb
            dM, ta = self.est
            if ta is not None and t - d - ta > TF.S_HI + 1.0:
                self.frozen = True
        xh = self.xbar + g['Mk'] @ (y_raw - self.known(t) - m['C'] @ self.xbar)
        u_fb = float(-(g['K'] @ xh)[0])
        if abs(u_fb) > self.a_max:
            u_fb = math.copysign(self.a_max, u_fb); self.sat += 1
        self.xbar = m['A'] @ xh + m['B'][:, 0]*u_fb
        self.stored.append((t, y_raw, u_fb))
        if u_ff:
            self.ff_launch.append((t, u_ff))
        ix = m['idx']
        self.launch.append((t, u_fb + u_ff)); self.fb.append((t, u_fb)); self.ff.append((t, u_ff))
        self.log.append((t, xh[ix['G']], xh[ix['Z']], xh[ix['F']], xh[ix['V']]))
        return u_fb + u_ff


def sensorcal():
    """Open loop, traveller only: detector energy passed vs the traveller's Misner-Sharp mass, and the
    timing offset between (flux-weighted passage time + XS) and the nominal arrival X0."""
    out = []
    for mt in (1e-4, 5e-3, 2e-2):
        x = grid(DX, TF.CAL_L)
        U, dM = TF.traveller(x, TF.s_amp_balanced(), mt)
        det = Detector()
        evolve(x, U, TF.X0 + 2.0 - XS + 3.0, out_dt=OUT_DT, controller=lambda t, x, U, h: det.read(t, x, U), ctrl_dt=DT/SUB)
        E, tc, _, _ = det.passage(1e9)
        out.append({'mass': dM, 'E': E, 'kappa': dM/E, 'delta': TF.X0 - (tc + XS)})
        print('mass %.3e: detector energy %.4e, mass per unit energy %.5f, timing offset %+.4f b' % (dM, E, dM/E, TF.X0 - (tc + XS)))
    cal = {'xs': XS, 'kappa': out[1]['kappa'], 'delta': out[1]['delta'], 'points': out}
    json.dump(cal, open(SCAL, 'w'), indent=1)
    return cal


def one(args):
    dM_target, mode, T = args                 # mode: 'sensed' | 'sensed-nostream'
    x = grid(DX, T + TF.X0 + 16.0)
    U, dM = TF.traveller(x, TF.s_amp_balanced(), dM_target)
    m = build_model(TF.D)
    sig = TF.load_signature()
    cal = json.load(open(SCAL))
    ctl = SensedLQG(m, design(m, **TUNE), sig, cal, stream=(mode == 'sensed'))
    r = evolve(x, U, T, out_dt=OUT_DT, controller=ctl, ctrl_dt=DT/SUB)
    h = r['hist']; t = h['t']; dR = h['R0'] - 1.0
    fb = np.array(ctl.fb); ff = np.array(ctl.ff)
    after = fb[:, 0] >= TF.X0 - TF.D
    return {'traveller_mass': dM, 'mode': mode, 'outcome': r['outcome'], 'lifetime': float(r['lifetime']),
            'est_mass': ctl.est[0], 'est_arrival': ctl.est[1],
            'max_H': float(np.nanmax(h['H'])), 'max_offset_after_arrival': float(np.abs(dR[t >= TF.X0 - 2]).max()),
            'final_offset': float(dR[-1]), 'saturated': int(ctl.sat), 'ff_pulse': float(ff[:, 1].sum()),
            'ff_pulses_launched': int((ff[:, 1] != 0).sum()),
            'peak_feedback_pulse_after_arrival': float(np.abs(fb[after, 1]).max()) if after.any() else None}


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'sensorcal':
        sensorcal()
    elif what == 'scan':
        T = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
        masses = [float(v) for v in sys.argv[3:]] or [5e-3, 1e-2, 2e-2, 3e-2, 4e-2]
        modes = os.environ.get('WH_MODES', 'sensed').split(',')
        with Pool(2) as pool:
            rows = pool.map(one, [(dm, md, T) for dm in masses for md in modes], chunksize=1)
        json.dump({'d': TF.D, 'x0': TF.X0, 'xs': XS, 'dx': DX, 'T': T, 'sides': TF.SIDES, 'rows': rows},
                  open('results_transit_sensed_scan%s%s%s%s.json' % ('' if TF.SIDES == 2 else '_1side', '_track' if TF.TRACK else '', os.environ.get('WH_OUT', ''), TAG), 'w'), indent=1)
        for r in rows:
            print('dM %.3e %-8s %-8s at %5.1f  est mass %.3e arr %.3f  max|dR| %.2e  ff %.2e in %d pulses  peak fb %.2e  sat %d  maxH %.1e' % (
                r['traveller_mass'], r['mode'], r['outcome'], r['lifetime'], r['est_mass'], r['est_arrival'] or 0,
                r['max_offset_after_arrival'], r['ff_pulse'], r['ff_pulses_launched'], r['peak_feedback_pulse_after_arrival'] or 0, r['saturated'], r['max_H']))
