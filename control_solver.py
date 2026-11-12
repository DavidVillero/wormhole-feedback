"""Two-field Einstein-scalar solver for the wormhole control study.

Same gauge and equations as wormhole_solver.py (validated there), extended to
two massless scalars: F = phantom (eps = -1, supports the wormhole) and
G = normal (eps = +1).  Their stress tensors simply add.

Actuation never uses a source term (that would break stress-energy
conservation and the constraints).  Control pulses are put into the scalar
fields on a spatial slice and the Hamiltonian and momentum constraints are then
re-solved outward for (R, R_t), so every slice is genuine Einstein data.

State U = [R, S, F, G, PR, PS, PF, PG]  (S = sigma, P* = time derivatives).
"""
import numpy as np
from wormhole_solver import d1, d2

FOURPI = 4.0*np.pi


def d1_4(u, dx):
    out = d1(u, dx)
    out[2:-2] = (-u[4:] + 8.0*u[3:-1] - 8.0*u[1:-3] + u[:-4])/(12.0*dx)
    return out


def mid(f):
    m = 0.5*(f[:-1] + f[1:])
    m[1:-1] = (-f[:-3] + 9.0*f[1:-2] + 9.0*f[2:-1] - f[3:])/16.0
    return m


def solve_constraints(x, U, i_right, i_left, y0=None):
    """Integrate H = 0, M = 0 outward from grid indices i_right (to +x) and i_left (to -x),
    overwriting R and PR there.  RK4 in x with 4th-order interpolated coefficients."""
    R, S, F, G, PR, PS, PF, PG = U
    dx = x[1] - x[0]
    n = len(x)
    Fx, Gx = d1_4(F, dx), d1_4(G, dx)
    arrs = [np.exp(2.0*S), d1_4(S, dx), PS,
            -(PF**2 + Fx**2) + (PG**2 + Gx**2),      # sum eps (phi_t^2 + phi_x^2)
            -PF*Fx + PG*Gx]                          # sum eps phi_t phi_x
    mids = [mid(a) for a in arrs]
    Rx = d1_4(R, dx)

    def f(y, c):
        r, rx, p = y
        e, sx, ps, E, J = c
        return np.array([rx,
                         (e + p*p - rx*rx)/(2.0*r) + ps*p + sx*rx - 2.0*np.pi*r*E,
                         ps*rx + sx*p - FOURPI*r*J])
    for i, step in ((i_right, 1), (i_left, -1)):
        y = np.array([R[i], Rx[i], PR[i]]) if y0 is None else np.array(y0, dtype=float)
        if y0 is not None:
            R[i], PR[i] = y[0], y[2]
        h = step*dx
        while 0 <= i + step < n:
            c0 = [a[i] for a in arrs]
            cm = [m[i if step > 0 else i - 1] for m in mids]
            c1 = [a[i + step] for a in arrs]
            k1 = f(y, c0); k2 = f(y + 0.5*h*k1, cm); k3 = f(y + 0.5*h*k2, cm); k4 = f(y + h*k3, c1)
            y = y + h/6.0*(k1 + 2*k2 + 2*k3 + k4)
            i += step
            R[i], PR[i] = y[0], y[2]
    return U


def bump(x, amp, x0, w):
    ax, sg = np.abs(x), np.sign(x)
    g = amp*np.exp(-((ax - x0)/w)**2)
    return g, -2.0*(ax - x0)/w**2*g*sg       # g(|x|) and d/dx of it


def add_pulse(x, U, field, amp, x0, w):
    """Add an ingoing pulse on both sides of the throat.  Phantom pulses are odd in x
    (they then couple linearly to the even unstable mode); normal pulses are even."""
    g, gx = bump(x, amp, x0, w)
    sg = np.sign(x)
    if field == 'phantom':
        U[2] += sg*g; U[6] += gx          # dF/dx = sg*gx ; ingoing: F_t = sign(x) F_x
    else:
        U[3] += g; U[7] += sg*gx
    return U


def initial_state(x, pulses=(), s_amp=0.0):
    """Static Ellis-Bronnikov scalar plus optional ingoing pulses, sigma_t = 0,
    sigma = s_amp*exp(-x^2) (a small trim used to balance the wormhole on the grid),
    constraints solved outward from the throat (R = 1, R_x = 0, R_t = 0 there)."""
    U = np.zeros((8, len(x)))
    U[0] = np.sqrt(x**2 + 1.0)
    U[1] = s_amp*np.exp(-x**2)
    U[2] = np.arctan(x)/np.sqrt(FOURPI)
    for p in pulses:
        add_pulse(x, U, *p)
    i0 = len(x)//2
    return solve_constraints(x, U, i0, i0, y0=(1.0, 0.0, 0.0))


def rhs(U, dx, ko):
    R, S, F, G, PR, PS, PF, PG = U
    Rx, Fx, Gx = d1(R, dx), d1(F, dx), d1(G, dx)
    Q = (np.exp(2.0*S) + PR**2 - Rx**2)/R
    dU = np.empty_like(U)
    dU[0], dU[1], dU[2], dU[3] = PR, PS, PF, PG
    dU[4] = d2(R, dx) - Q
    dU[5] = d2(S, dx) + Q/R - FOURPI*(-(PF**2 - Fx**2) + (PG**2 - Gx**2))
    dU[6] = d2(F, dx) - (2.0/R)*(PR*PF - Rx*Fx)
    dU[7] = d2(G, dx) - (2.0/R)*(PR*PG - Rx*Gx)
    if ko > 0.0:
        dU[:, 2:-2] -= ko/(16.0*dx)*(U[:, 4:] - 4.0*U[:, 3:-1] + 6.0*U[:, 2:-2] - 4.0*U[:, 1:-3] + U[:, :-4])
    dU[:, :2] = 0.0
    dU[:, -2:] = 0.0
    return dU


def constraints(U, dx):
    R, S, F, G, PR, PS, PF, PG = U
    Rx, Sx, Fx, Gx = d1(R, dx), d1(S, dx), d1(F, dx), d1(G, dx)
    Q = (np.exp(2.0*S) + PR**2 - Rx**2)/R
    H = d2(R, dx) - 0.5*Q - (PS*PR + Sx*Rx) + 2.0*np.pi*R*(-(PF**2 + Fx**2) + (PG**2 + Gx**2))
    M = d1(PR, dx) - (PS*Rx + Sx*PR) + FOURPI*R*(-PF*Fx + PG*Gx)
    return H, M


def evolve(x, U, T, cfl=0.5, ko=0.05, out_dt=0.05, lo=0.5, hi=2.0, xa=10.0,
           controller=None, ctrl_dt=0.25, snap_times=(), resume=None):
    """Evolve until T or until the throat radius R(0) leaves [lo, hi].
    controller(t, x, U, hist) may modify U in place every ctrl_dt.
    resume = (t_start, tau, hist-of-lists) continues an earlier call; T is then the length of this leg."""
    dx = x[1] - x[0]
    dt = cfl*dx
    i0 = len(x)//2
    every = max(1, int(round(out_dt/dt)))
    cevery = max(1, int(round(ctrl_dt/dt)))
    ana = np.abs(x) <= xa + 1e-12
    hist = {k: [] for k in ('t', 'tau', 'R0', 'PR0', 'S0', 'H', 'M', 'PS0')}
    snaps = {}
    tau, step, outcome = 0.0, 0, 'alive'
    t_start = 0.0
    if resume is not None:
        t_start, tau, hist = resume
    nsteps = int(round(T/dt))
    while True:
        t = t_start + step*dt
        fresh = not (resume is not None and step == 0)       # a resumed leg already did this instant
        if fresh and controller is not None and step % cevery == 0:
            controller(t, x, U, hist)
        if fresh and step % every == 0:
            H, M = constraints(U, dx)
            for k, v in zip(hist, (t, tau, U[0][i0], U[4][i0], U[1][i0],
                                   np.sqrt(np.mean(H[ana]**2)), np.sqrt(np.mean(M[ana]**2)), U[5][i0])):
                hist[k].append(float(v))
        for ts in snap_times:
            if abs(t - ts) < 0.5*dt:
                snaps[ts] = U[:, ana].copy()
        if step >= nsteps:
            break
        e_old = np.exp(U[1][i0])
        with np.errstate(all='ignore'):
            k1 = rhs(U, dx, ko); k2 = rhs(U + 0.5*dt*k1, dx, ko)
            k3 = rhs(U + 0.5*dt*k2, dx, ko); k4 = rhs(U + dt*k3, dx, ko)
            U += dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
        step += 1
        r0 = U[0][i0]
        if not np.isfinite(r0) or r0 < lo:
            outcome = 'collapse'; break
        if r0 > hi:
            outcome = 'expand'; break
        tau += 0.5*dt*(e_old + np.exp(U[1][i0]))
    return {'outcome': outcome, 'lifetime': t_start + step*dt, 'tau': tau,
            'hist': {k: np.array(v) for k, v in hist.items()}, 'snaps': snaps, 'x': x[ana]}


def grid(dx, L):
    n = int(round(L/dx))
    return dx*np.arange(-n, n + 1)
