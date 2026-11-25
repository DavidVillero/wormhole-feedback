"""Transit study, Stage 3a: one traveller, launched from one side only, crossing the throat.

The Stage 2 travellers came in symmetric pairs (one from each side), so nothing actually went *through*
the wormhole: the two pulses met at the throat and each went back out its own side, which is the same
data.  Here a single normal-matter pulse comes in from +X0 and is followed out of the other mouth.

    WH_SIDES=1 WH_MCAL=0.005 WH_CLO=0 python3 transit_ff.py calib   # one-sided feedforward + signature
    python3 transit_onesided.py testfield                            # linear transmission on a fixed static throat
    python3 transit_onesided.py scan [T] [masses]                    # results_transit_1side_scan[_dx...].json

Per run it reports the outcome, the fraction of the traveller's energy found on the far side
(transmitted) and back on the near side (reflected), the change of the Misner-Sharp mass seen from far
away on each side, and how far the throat (the minimum of the areal radius) moves off x = 0.
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

assert TF.SIDES == 1


def energies(x, U, x_min=6.0):
    """Energy of the normal (traveller) field on each side, |x| > x_min, flat-space weighting 4 pi R^2.
    Used only where the pulses are far from the throat (sigma ~ 0 there)."""
    dx = x[1] - x[0]
    R, G, PG = U[0], U[3], U[7]
    Gx = np.gradient(G, dx)
    e = 2.0*np.pi*R**2*(PG**2 + Gx**2)
    return float(e[x < -x_min].sum()*dx), float(e[x > x_min].sum()*dx)


def throat(x, U, half=3.0):
    """Location and value of the minimum areal radius near x = 0 (parabolic refinement)."""
    sel = np.nonzero(np.abs(x) <= half)[0]
    R = U[0][sel]
    j = int(np.argmin(R))
    if 0 < j < len(R) - 1:
        a, b, c = R[j - 1], R[j], R[j + 1]
        den = a - 2*b + c
        off = 0.5*(a - c)/den if den != 0 else 0.0
        return float(x[sel][j] + off*(x[1] - x[0])), float(b - 0.25*(a - c)*off)
    return float(x[sel][j]), float(R[j])


class Probe:
    """Wraps a controller (or none) and records diagnostics at every control step."""

    def __init__(self, inner, t_energy, far):
        self.inner, self.t_energy, self.far = inner, t_energy, far
        self.rows, self.E0, self.E1, self.m0 = [], None, None, None

    def masses(self, x, U):
        i_r = int(round((self.far - x[0])/DX)); i_l = len(x) - 1 - i_r
        return float(TF.mass(x, U, i_l)), float(TF.mass(x, U, i_r))

    def __call__(self, t, x, U, hist):
        if self.E0 is None:
            self.E0 = energies(x, U); self.m0 = self.masses(x, U)
        if self.E1 is None and t >= self.t_energy - 1e-9:
            self.E1 = energies(x, U)
        self.rows.append((t, *throat(x, U)))
        self.m1 = self.masses(x, U)
        if self.inner is not None:
            self.inner(t, x, U, hist)


def one(args):
    dM_target, mode, T = args
    x = grid(DX, T + TF.X0 + 16.0)
    U, dM = TF.traveller(x, TF.s_amp_balanced(), dM_target)
    ctl = None
    if mode != 'none':
        m = build_model(TF.D)
        sig = TF.load_signature() if mode != 'plain' else None
        ctl = TF.TransitLQG(m, design(m, **TUNE), travellers=[(TF.X0, dM)] if dM else [],
                            c=float(sig['c']) if sig is not None else 0.0, a_max=0.1,
                            sig=sig if mode == 'ff+sig' else None)
    pr = Probe(ctl, TF.X0 + 20.0, T - TF.X0 + 8.0)
    r = evolve(x, U, T, out_dt=OUT_DT, controller=pr, ctrl_dt=DT)
    h = r['hist']; t = h['t']; dR = h['R0'] - 1.0
    th = np.array(pr.rows)
    out = {'traveller_mass': dM, 'mode': mode, 'outcome': r['outcome'], 'lifetime': float(r['lifetime']),
           'max_H': float(np.nanmax(h['H'])), 'max_M': float(np.nanmax(h['M'])),
           'max_offset_after_arrival': float(np.abs(dR[t >= TF.X0 - 2]).max()),
           'max_throat_shift': float(np.abs(th[:, 1]).max()), 'final_throat_shift': float(th[-1, 1]),
           'E_in': pr.E0[1]}
    if pr.E1 is not None and pr.E0[1] > 0:
        out.update(transmitted=pr.E1[0]/pr.E0[1], reflected=pr.E1[1]/pr.E0[1])
    if r['outcome'] == 'alive':
        out.update(dm_far_left=pr.m1[0] - pr.m0[0], dm_far_right=pr.m1[1] - pr.m0[1],
                   final_offset=float(dR[-1]))
    if ctl is not None and ctl.fb:
        fb = np.array(ctl.fb)
        out.update(saturated=int(ctl.sat), ff_pulse=float(np.array(ctl.ff)[:, 1].sum()),
                   peak_feedback_pulse_after_arrival=float(np.abs(fb[fb[:, 0] >= TF.X0 - TF.D, 1]).max()))
    return out


def testfield(width=TF.WIDTH, L=60.0, dx=0.0125, x0=20.0):
    """Linear test field on the exact static throat (no back-reaction): u = R G obeys
    u_tt = u_xx - V u with V = R''/R = 1/(1 + x^2)^2.  Returns the transmitted energy fraction."""
    xg = np.arange(-L, L + dx/2, dx)
    V = 1.0/(1.0 + xg**2)**2
    R = np.sqrt(xg**2 + 1.0)
    g = np.exp(-((xg - x0)/width)**2)
    u = R*g
    ut = R*(-2.0*(xg - x0)/width**2*g)              # ingoing, G_t = G_x; R is static
    dt = 0.5*dx

    def acc(u):
        a = np.zeros_like(u)
        a[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2])/dx**2 - V[1:-1]*u[1:-1]
        return a
    T = x0 + 25.0
    for _ in range(int(round(T/dt))):              # velocity Verlet
        ut += 0.5*dt*acc(u); u += dt*ut; ut += 0.5*dt*acc(u)
    ux = np.gradient(u, dx)
    e = 0.5*(ut**2 + ux**2 + V*u**2)
    E = e.sum()*dx
    return float(e[xg < 0].sum()*dx/E), float(e[xg > 0].sum()*dx/E)


def show(rows):
    for r in rows:
        print('dM %.3e %-7s %-8s at %5.1f  max|dR| %.2e  throat shift max %.2e final %+.2e  T %s R %s  dm_far L %s R %s  maxH %.1e' % (
            r['traveller_mass'], r['mode'], r['outcome'], r['lifetime'], r['max_offset_after_arrival'],
            r['max_throat_shift'], r['final_throat_shift'],
            '%.4f' % r['transmitted'] if 'transmitted' in r else '-', '%.4f' % r['reflected'] if 'reflected' in r else '-',
            '%+.3e' % r['dm_far_left'] if 'dm_far_left' in r else '-', '%+.3e' % r['dm_far_right'] if 'dm_far_right' in r else '-',
            r['max_H']))


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'testfield':
        res = {}
        for w in (0.5, 1.0, 2.0):
            for d in (0.025, 0.0125):
                tr, rf = testfield(width=w, dx=d)
                res['w%g_dx%g' % (w, d)] = {'transmitted': tr, 'reflected': rf}
                print('width %.1f, dx %.4f: transmitted %.5f, reflected %.5f' % (w, d, tr, rf))
        json.dump(res, open('results_transit_testfield.json', 'w'), indent=1)
    elif what == 'scan':
        T = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
        masses = [float(v) for v in sys.argv[3:]] or [2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2]
        modes = os.environ.get('WH_MODES', 'plain,ff+sig').split(',')
        with Pool(2) as pool:
            rows = pool.map(one, [(dm, md, T) for dm in masses for md in modes], chunksize=1)
        json.dump({'d': TF.D, 'x0': TF.X0, 'dx': DX, 'T': T, 'sides': 1, 'cal_mass': TF.M_CAL, 'rows': rows},
                  open('results_transit_1side_scan%s%s%s.json' % ('_track' if TF.TRACK else '', os.environ.get('WH_OUT', ''), TAG), 'w'), indent=1)
        show(rows)
