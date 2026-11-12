"""System identification of the wormhole as a control plant.

    python3 sysid.py data d w     # impulse response (odd part of a +/- pulse pair) and uncontrolled baseline
    python3 sysid.py fit d        # ERA state-space fit, modes, validation

Outputs are throat readings y = [R0 - 1, R0_t, sigma0, sigma0_t], sampled every 0.05 b.
Input is the field amplitude of one phantom pulse pair launched from |x| = d.
"""
import json
import sys
import numpy as np
from control_solver import grid, initial_state, evolve
from control_feedback import inject, DX, OUT_DT, TAG

KEYS = ('R0', 'PR0', 'S0', 'PS0')
TAU = 12.0            # response recorded for d + TAU after launch


def s_amp():
    b = json.load(open('runs/balance_log%s.json' % TAG))
    return 0.5*(b['lo'] + b['hi'])


def outputs(h):
    y = np.array([h[k] for k in KEYS]).T.copy()
    y[:, 0] -= 1.0
    return y


def record(d, w, a=1e-9):
    T = d + TAU
    x = grid(DX, T + d + 16.0)
    ys = []
    for sgn in (1, -1):
        U = initial_state(x, s_amp=s_amp())
        inject(x, U, sgn*a, d, w)
        ys.append(outputs(evolve(x, U, T, out_dt=OUT_DT)['hist']))
    n = min(len(ys[0]), len(ys[1]))
    h = 0.5*(ys[0][:n] - ys[1][:n])/a
    even = 0.5*(ys[0][:n] + ys[1][:n])
    xb = grid(DX, 30.0 + 16.0)
    base = outputs(evolve(xb, initial_state(xb, s_amp=s_amp()), 30.0, out_dt=OUT_DT)['hist'])
    raw = outputs(evolve(xb, initial_state(xb), 30.0, out_dt=OUT_DT)['hist'])
    np.savez('runs/sysid_d%g%s.npz' % (d, TAG), h=h, even=even, base=base, raw=raw, d=d, w=w, a=a)
    return h


def era(h, n, mu, rows=None):
    """Eigensystem Realization Algorithm on Markov parameters h[1:], with exponential
    de-weighting e^{-mu t} so the growing mode does not swamp the Hankel matrix."""
    N, p = h.shape
    rho = np.exp(-mu*OUT_DT)
    Y = h*(rho**np.arange(N))[:, None]
    r = rows or (N - 2)//2
    c = N - 2 - r
    H0 = np.block([[Y[i + j + 1][:, None] for j in range(c)] for i in range(r)])
    H1 = np.block([[Y[i + j + 2][:, None] for j in range(c)] for i in range(r)])
    Uu, S, Vt = np.linalg.svd(H0, full_matrices=False)
    Sn = np.sqrt(S[:n])
    A = (Uu[:, :n].T @ H1 @ Vt[:n].T)/Sn[:, None]/Sn[None, :]
    B = (Sn[:, None]*Vt[:n])[:, :1]
    C = (Uu[:, :n]*Sn[None, :])[:p]
    return A/rho, B/rho, C, S           # undo the weighting: y_j = C A^(j-1) B


def simulate(A, B, C, u, x0=None):
    """y_j for inputs u_j applied at sample j (x_{j+1} = A x_j + B u_j, y_j = C x_j)."""
    x = np.zeros(A.shape[0]) if x0 is None else x0.copy()
    out = []
    for uj in u:
        out.append(C @ x)
        x = A @ x + B[:, 0]*uj
    return np.array(out)


if __name__ == '__main__':
    stage, d = sys.argv[1], float(sys.argv[2])
    if stage == 'data':
        h = record(d, float(sys.argv[3]))
        print('recorded', h.shape, 'final |h| =', np.abs(h[-1]))
    else:
        D = np.load('runs/sysid_d%g%s.npz' % (d, TAG))
        h = D['h']
        mu = 0.6
        A, B, C, S = era(h, 2, mu)
        print('Hankel singular values (first 16, relative):', np.round(S[:16]/S[0], 7))
        for n in (4, 6, 8, 10, 12, 16):
            A, B, C, S = era(h, n, mu)
            ev = np.linalg.eigvals(A)
            rates = np.log(ev.astype(complex))/OUT_DT
            u = np.zeros(len(h)); u[0] = 1.0
            yh = simulate(A, B, C, u)
            wgt = np.exp(-1.18*OUT_DT*np.arange(len(h)))[:, None]
            err = np.sqrt(np.mean(((yh - h)*wgt)**2, axis=0))/np.sqrt(np.mean((h*wgt)**2, axis=0))
            order = np.argsort(-rates.real)
            print('n=%2d  weighted fit error per output %s' % (n, np.round(err, 5)))
            print('      continuous-time rates:', ', '.join('%.3f%+.3fi' % (z.real, z.imag) for z in rates[order][:8]))
