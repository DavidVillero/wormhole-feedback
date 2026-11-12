"""Reference runs from the Python solver for checking the JavaScript port (viz_live.html).
Uses the same controller matrices as the page (viz_models.json) and dx = 0.05."""
import json
import math
import numpy as np
from control_solver import grid, initial_state, evolve, add_pulse
from control_feedback import ext_solve

DX, L, W = 0.05, 60.0, 0.3
TRIM = json.load(open('runs/balance_log_dx0.05.json'))
TRIM = 0.5*(TRIM['lo'] + TRIM['hi'])
VM = {D['d']: D for D in json.load(open('viz_models.json'))['distances']}


def inject_field(x, U, field, amp, x0, w):
    n = len(x); dx = x[1] - x[0]
    i_s = int(round((x0 - 3.0*w - x[0])/dx))
    R0, P0 = ext_solve(x, U, i_s)
    add_pulse(x, U, field, amp, x0, w)
    R1, P1 = ext_solve(x, U, i_s)
    dR, dP = R1 - R0, P1 - P0
    U[0][i_s:] += dR; U[4][i_s:] += dP
    U[0][:n - i_s] += dR[::-1]; U[4][:n - i_s] += dP[::-1]


class Ctl:
    def __init__(self, d, tm_on=10.0, disturb=None):
        D = VM[d]['ctrl']; n = D['n']
        self.A = np.zeros((n, n))
        for i, j, v in D['A']:
            self.A[i, j] = v
        self.B = np.array(D['B']); self.C = np.array(D['C']); self.K = np.array(D['K']); self.Mk = np.array(D['Mk'])
        self.iZ, self.iV, self.d, self.tm_on = D['iZ'], D['iV'], d, tm_on
        self.xbar = np.zeros(n); self.launch = []; self.disturb = disturb

    def __call__(self, t, x, U, hist):
        if self.disturb and abs(t - self.disturb[0]) < 1e-9:
            inject_field(x, U, *self.disturb[1:])
        tm = t - self.d
        if tm < self.tm_on:
            return
        i = min(int(round(tm/0.05)), len(hist['t']) - 1)
        y = np.array([hist['R0'][i] - 1.0, hist['PR0'][i]])
        if not self.launch:
            self.xbar[self.iZ] = y[0]; self.xbar[self.iV] = y[1]
        xh = self.xbar + self.Mk @ (y - self.C @ self.xbar)
        u = float(np.clip(-self.K @ xh, -0.02, 0.02))
        self.xbar = self.A @ xh + self.B*u
        self.launch.append((t, u))
        inject_field(x, U, 'phantom', u, self.d, W)


def pack(r, ctl=None):
    h = r['hist']
    return {'outcome': r['outcome'], 'lifetime': r['lifetime'], 't': h['t'].tolist(), 'R0': h['R0'].tolist(),
            'PR0': h['PR0'].tolist(), 'S0': h['S0'].tolist(), 'H': h['H'].tolist(), 'M': h['M'].tolist(),
            'launch': ctl.launch if ctl else []}

out = {'dx': DX, 'L': L, 'trim': TRIM}
x = grid(DX, L)
out['free_normal_pulse'] = pack(evolve(x, initial_state(x, [('normal', 0.02, 6.0, 1.0)]), 12.0))
out['free_balanced'] = pack(evolve(x, initial_state(x, s_amp=TRIM), 45.0))
for d, T, dist, name in ((1.0, 40.0, None, 'lqg_d1'), (2.0, 40.0, None, 'lqg_d2'),
                         (1.0, 40.0, (20.0, 'normal', 0.002, 6.0, 1.0), 'lqg_d1_matter')):
    c = Ctl(d, disturb=dist)
    out[name] = pack(evolve(x, initial_state(x, s_amp=TRIM), T, controller=c, ctrl_dt=0.25), c)
for k, v in out.items():
    if isinstance(v, dict):
        print('%-18s %-8s %.2f  launches %d  final R0 %.8f' % (k, v['outcome'], v['lifetime'], len(v['launch']), v['R0'][-1]))
json.dump(out, open('runs/viz_reference.json', 'w'))
