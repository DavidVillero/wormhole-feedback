"""Spherically symmetric Einstein + massless scalar solver (conformal gauge).

    ds^2 = e^{2 sigma(t,x)} (-dt^2 + dx^2) + R(t,x)^2 dOmega^2,   G = c = 1
    T_ab = eps [ d_a phi d_b phi - (1/2) g_ab (d phi)^2 ],  eps=-1 phantom, +1 canonical

Evolution equations (verified symbolically in derive_check.py), Q = (e^{2 sigma} + R_t^2 - R_x^2)/R:
    R_tt     = R_xx - Q
    sigma_tt = sigma_xx + Q/R - 4 pi eps (phi_t^2 - phi_x^2)
    phi_tt   = phi_xx - (2/R)(R_t phi_t - R_x phi_x)
Constraints (monitored, not enforced):
    H = R_xx - Q/2 - (sigma_t R_t + sigma_x R_x) + 2 pi eps R (phi_t^2 + phi_x^2)
    M = R_tx - (sigma_t R_x + sigma_x R_t) + 4 pi eps R phi_t phi_x

Numerics: method of lines, 2nd-order centred differences, RK4, Kreiss-Oliger
dissipation.  All characteristic speeds are exactly 1 in these coordinates, so
the outer boundary (held at its initial values) cannot influence |x| < L - t.
"""
import json
import sys
import time
import numpy as np

FOURPI = 4.0*np.pi


def initial_data(x, b=1.0, amp=0.0, width=1.0, eps=-1.0):
    """Time-symmetric data: Ellis-Bronnikov areal radius, conformal factor
    sigma = amp*exp(-x^2/width^2), and phi_x fixed by solving the Hamiltonian
    constraint exactly.  amp = 0 is the exact static wormhole (eps = -1)."""
    def fx(xx):
        R = np.sqrt(xx**2 + b**2)
        Rx, Rxx = xx/R, b**2/R**3
        S = amp*np.exp(-(xx/width)**2)
        Sx = -2.0*xx/width**2*S
        # H = 0 with all first time derivatives zero (note R_tt != 0 is already eliminated in H)
        fx2 = -(Rxx - 0.5*(np.exp(2.0*S) - Rx**2)/R - Sx*Rx)/(2.0*np.pi*eps*R)
        if np.any(fx2 < 0):
            raise ValueError('initial data needs phi_x^2 < 0: no real scalar field supports this geometry')
        return np.sqrt(fx2)
    R = np.sqrt(x**2 + b**2)
    S = amp*np.exp(-(x/width)**2)
    # phi = integral of phi_x, 6-point Gauss-Legendre in every cell (resolution independent)
    gx, gw = np.polynomial.legendre.leggauss(6)
    a, c = x[:-1], x[1:]
    mid, half = 0.5*(a + c), 0.5*(c - a)
    cell = sum(w*fx(mid + half*g) for g, w in zip(gx, gw))*half
    F = np.concatenate([[0.0], np.cumsum(cell)])
    F -= F[len(x)//2]
    Z = np.zeros_like(x)
    return np.array([R, S, F, Z.copy(), Z.copy(), Z.copy()])


def d1(u, dx):
    out = np.zeros_like(u)
    out[1:-1] = (u[2:] - u[:-2])/(2.0*dx)
    return out


def d2(u, dx):
    out = np.zeros_like(u)
    out[1:-1] = (u[2:] - 2.0*u[1:-1] + u[:-2])/dx**2
    return out


def rhs(U, dx, eps, ko):
    R, S, F, PR, PS, PF = U
    Rx, Sx, Fx = d1(R, dx), d1(S, dx), d1(F, dx)
    Q = (np.exp(2.0*S) + PR**2 - Rx**2)/R
    dU = np.empty_like(U)
    dU[0], dU[1], dU[2] = PR, PS, PF
    dU[3] = d2(R, dx) - Q
    dU[4] = d2(S, dx) + Q/R - FOURPI*eps*(PF**2 - Fx**2)
    dU[5] = d2(F, dx) - (2.0/R)*(PR*PF - Rx*Fx)
    if ko > 0.0:
        dU[:, 2:-2] -= ko/(16.0*dx)*(U[:, 4:] - 4.0*U[:, 3:-1] + 6.0*U[:, 2:-2] - 4.0*U[:, 1:-3] + U[:, :-4])
    dU[:, :2] = 0.0   # boundary points held at initial values
    dU[:, -2:] = 0.0
    return dU


def constraints(U, dx, eps):
    R, S, F, PR, PS, PF = U
    Rx, Sx, Fx = d1(R, dx), d1(S, dx), d1(F, dx)
    Q = (np.exp(2.0*S) + PR**2 - Rx**2)/R
    H = d2(R, dx) - 0.5*Q - (PS*PR + Sx*Rx) + 2.0*np.pi*eps*R*(PF**2 + Fx**2)
    M = d1(PR, dx) - (PS*Rx + Sx*PR) + FOURPI*eps*R*PF*Fx
    return H, M


def outermost_crossing(x, theta, R, sign):
    """Outermost x>0 where sign*theta > 0 ends (a marginal surface); returns (x, R) or (nan, nan)."""
    idx = np.nonzero(sign*theta[:-1] > 0)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    i = idx[-1]
    w = theta[i]/(theta[i] - theta[i+1])
    return x[i] + w*(x[i+1] - x[i]), R[i] + w*(R[i+1] - R[i])


def run(dx=0.025, L=40.0, T=25.0, amp=0.0, width=1.0, eps=-1.0, b=1.0, cfl=0.5,
        ko=0.05, out_dt=0.05, xa=10.0, snap_dt=1.0, rmin_stop=0.02):
    n = int(round(L/dx))
    x = dx*np.arange(-n, n + 1)
    i0 = n
    U = initial_data(x, b, amp, width, eps)
    dt = cfl*dx
    every = int(round(out_dt/dt))
    snap_every = int(round(snap_dt/dt))
    ana = np.abs(x) <= xa + 1e-12
    pos = (x >= 0) & (x <= L - T - 2.0)
    xs_stride = max(1, int(round(0.05/dx)))
    hist = {k: [] for k in ('t', 'tau', 'R0', 'PR0', 'sig0', 'Rmin', 'H', 'M',
                            'x_trap', 'R_trap', 'x_anti', 'R_anti')}
    snaps = {'x': x[ana][::xs_stride].tolist(), 't': [], 'R': [], 'sigma': [], 'phi': []}
    tau, status, step = 0.0, 'reached final time', 0
    nsteps = int(round(T/dt))
    t0 = time.time()
    while True:
        t = step*dt
        if step % every == 0:
            R, S, F, PR, PS, PF = U
            H, M = constraints(U, dx, eps)
            Rx = d1(R, dx)
            xt, Rt = outermost_crossing(x[pos], (PR + Rx)[pos], R[pos], -1.0)   # outgoing expansion < 0
            xn, Rn = outermost_crossing(x[pos], (PR - Rx)[pos], R[pos], +1.0)   # ingoing expansion > 0
            for k, v in zip(hist, (t, tau, R[i0], PR[i0], S[i0], R[ana].min(),
                                   np.sqrt(np.mean(H[ana]**2)), np.sqrt(np.mean(M[ana]**2)),
                                   xt, Rt, xn, Rn)):
                hist[k].append(float(v))
        if step % snap_every == 0:
            snaps['t'].append(t)
            for k, j in (('R', 0), ('sigma', 1), ('phi', 2)):
                snaps[k].append(U[j][ana][::xs_stride].tolist())
        if step >= nsteps:
            break
        e_old = np.exp(U[1][i0])
        with np.errstate(all='ignore'):   # overflow near the singularity is caught just below
            k1 = rhs(U, dx, eps, ko)
            k2 = rhs(U + 0.5*dt*k1, dx, eps, ko)
            k3 = rhs(U + 0.5*dt*k2, dx, eps, ko)
            k4 = rhs(U + dt*k3, dx, eps, ko)
            U = U + dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
        step += 1
        if not np.all(np.isfinite(U)) or U[0].min() < rmin_stop:
            status = 'stopped: areal radius -> 0 (singularity) at t=%.4f' % (step*dt)
            break
        tau += 0.5*dt*(e_old + np.exp(U[1][i0]))   # proper time at the throat
    return {'params': dict(dx=dx, L=L, T=T, amp=amp, width=width, eps=eps, b=b, cfl=cfl, ko=ko, xa=xa),
            'status': status, 'wall_seconds': time.time() - t0, 'history': hist, 'snapshots': snaps}


if __name__ == '__main__':
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    out = sys.argv[2] if len(sys.argv) > 2 else 'run.json'
    res = run(**kw)
    with open(out, 'w') as fh:
        json.dump(res, fh)
    h = res['history']
    print(out, '|', res['status'], '| wall %.1fs' % res['wall_seconds'],
          '| R0 final %.5f' % h['R0'][-1], '| H %.2e M %.2e' % (max(h['H']), max(h['M'])))
