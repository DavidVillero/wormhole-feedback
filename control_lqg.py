"""Option 2: observer-based regulator (Kalman filter + LQR) on an identified state-space model,
with an explicit robustness margin.

Model, per controller update (0.25 b), from pulse amplitude u to the readings the controller
RECEIVES (throat radius offset and its rate, already delayed by the light-travel time d):
  * a delay line holding the last M pulses, with measured response taps (covers travel, the
    transient, and the first 3 b after arrival);
  * one unstable state (the growing mode), a neutral state (permanent radius shift) and a few slow
    states approximating the power-law tail, all fed from the end of the delay line;
  * two disturbance states: a constant push on the growing mode and a steady radius drift.
The taps come from the tuned pulse-pair data by backward recursion, so they need no high-precision
growth rate (see ng_table in control_regulator.py).
"""
import json
import math
import os
import numpy as np
from control_feedback import OUT_DT, TAG
from control_regulator import fit_power_tail, PAIR, S_TOP

DT = 0.25
S_FIR = 3.0                         # delay line covers up to this long after arrival
MU = (0.012, 0.05, 0.2, 0.8)        # slow rates (per b) used to approximate the power-law tail
GAMMA = 0.999                       # slight decay on the disturbance states (keeps the Riccati equations solvable)


def raw_kernel(d):
    """Non-growing response of [radius offset, rate] to one pulse, and the growing gains, from pair data."""
    P = np.load('runs/pair_response%s%s.npz' % ('' if d == 1.0 else '_d%g' % d, TAG))
    beta = float(P['beta']); lam = math.log(beta)/PAIR
    pl = fit_power_tail(d)
    top = (lambda s: pl['n_inf'] - pl['c']*(s + pl['s0'])**(-pl['alpha']),
           lambda s: pl['c']*pl['alpha']*(s + pl['s0'])**(-pl['alpha'] - 1))
    step = int(round(PAIR/OUT_DT))
    i_top = int(round((d + S_TOP)/OUT_DT)); i_min = -int(round((d + 1.0)/OUT_DT))
    idx = np.arange(i_min, i_top + 1)
    s = idx*OUT_DT - d
    ng, k = [], []
    for c in (0, 1):
        q = P['odd'][:, c]
        v = {i: float(top[c](i*OUT_DT - d)) for i in range(i_top + 1, i_top + step + 1)}
        for i in range(i_top, i_min - 1, -1):
            j = i + step
            v[i] = (v[j] - (q[j] if j >= 0 else 0.0))/beta
        arr = np.array([v[i] for i in idx])
        ng.append(arr); k.append(float(-arr[idx == 0][0]*beta**(d/PAIR)))
    return {'s': s, 'ng': np.array(ng), 'beta': beta, 'lam': lam, 'k': k, 'top': top, 'pl': pl}


def build_model(d, shift=None):
    """shift = delay between launch and the modelled reading: 2d (default) gives the readings as the
    controller receives them; d gives the readings at the throat itself (used as the plant in the visual demo)."""
    K = raw_kernel(d)
    lam, beta = K['lam'], K['beta']
    shift = 2*d if shift is None else shift
    M = int(round((shift + S_FIR)/DT))
    lag_s = np.arange(M + 1)*DT - shift                   # time since arrival for lag j
    taps = np.zeros((2, M + 1))
    for c in (0, 1):
        tot = K['k'][c]*beta**(lag_s/PAIR) + np.interp(lag_s, K['s'], K['ng'][c])
        taps[c] = np.where(lag_s < -1.2, 0.0, tot)
    # slow approximation of the neutral tail of the radius channel for s > S_FIR
    s_fit = np.concatenate([np.arange(S_FIR, S_TOP + 1e-9, DT), np.arange(S_TOP + DT, 80.0, DT)])
    n_fit = np.where(s_fit <= S_TOP, np.interp(s_fit, K['s'], K['ng'][0]), K['top'][0](np.maximum(s_fit, S_TOP)))
    Phi = np.array([np.ones_like(s_fit)] + [np.exp(-mu*(s_fit - S_FIR)) for mu in MU]).T
    wts = np.where(s_fit <= 30.0, 1.0, 0.3)
    coef = np.linalg.lstsq(Phi*wts[:, None], n_fit*wts, rcond=None)[0]
    fit_rms = float(np.sqrt(np.mean((Phi @ coef - n_fit)[s_fit <= S_TOP]**2)))
    rates = [lam, 0.0] + [-mu for mu in MU]                # growing, neutral, slow...
    amp_at_fir = [K['k'][0]*beta**(S_FIR/PAIR)] + list(coef)   # modal amplitudes at lag M (radius channel)
    nm = len(rates)
    n = M + nm + 2
    iG, iZ, iF, iV = M, M + 1, M + nm, M + nm + 1
    A = np.zeros((n, n)); B = np.zeros((n, 1)); C = np.zeros((2, n))
    B[0, 0] = 1.0
    for j in range(1, M):
        A[j, j - 1] = 1.0
    for i, (r, a) in enumerate(zip(rates, amp_at_fir)):
        rho = math.exp(r*DT)
        A[M + i, M + i] = rho
        A[M + i, M - 1] = rho*a                             # pulse leaving the delay line (lag M) feeds the mode
        C[0, M + i] = 1.0
        C[1, M + i] = r
    C[0, :M] = taps[0, 1:]; C[1, :M] = taps[1, 1:]
    A[iF, iF] = GAMMA; A[iV, iV] = GAMMA
    A[iG, iF] = DT                                         # constant push on the growing mode
    A[iZ, iV] = DT; C[1, iV] = 1.0                         # steady radius drift
    # consistency of the rate channel beyond the delay line (not fitted, implied by the radius fit)
    s_chk = np.arange(S_FIR + DT, S_TOP, DT)
    model_p = sum(a*r*np.exp(r*(s_chk - S_FIR)) for r, a in zip(rates[1:], amp_at_fir[1:]))
    rate_rms = float(np.sqrt(np.mean((model_p - np.interp(s_chk, K['s'], K['ng'][1]))**2)))
    return {'d': d, 'A': A, 'B': B, 'C': C, 'M': M, 'idx': {'G': iG, 'Z': iZ, 'F': iF, 'V': iV}, 'lam': lam, 'beta': beta,
            'k': K['k'], 'taps': taps, 'tail_fit_rms': fit_rms, 'rate_channel_rms': rate_rms, 'tail_coef': coef.tolist()}


def dare(A, B, Q, R, iters=200000, tol=1e-13):
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + BtP @ B, BtP @ A)
        Pn = 0.5*(Pn + Pn.T)
        if np.abs(Pn - P).max() <= tol*max(1.0, np.abs(Pn).max()):
            return Pn
        P = Pn
    return P


def design(m, r_u=1e-2, w_rate=1.0, q_g=1.0, q_f=1e-2, q_v=1e-2, q_z=1e-2, v_meas=1.0, w_radius=1.0, w_g=0.0):
    """LQR on the received readings, steady-state Kalman filter with the given noise intensities.
    Everything is linear, so only ratios matter; readings are scaled by the growing gain k."""
    A, B, C, ix = m['A'], m['B'], m['C'], m['idx']
    k = m['k'][0]
    Cs = C/k
    # cost: received readings (weights w_radius, w_rate) plus the growing-mode state itself
    Q = Cs.T @ np.diag([w_radius, w_rate]) @ Cs
    Q[ix['G'], ix['G']] += w_g/k**2
    P = dare(A, B, Q, np.array([[r_u]]))
    Kf = np.linalg.solve(np.array([[r_u]]) + B.T @ P @ B, B.T @ P @ A)
    W = np.zeros_like(A)
    W[ix['G'], ix['G']] = q_g; W[ix['F'], ix['F']] = q_f; W[ix['V'], ix['V']] = q_v; W[ix['Z'], ix['Z']] = q_z
    S = dare(A.T, Cs.T, W, v_meas*np.eye(2))
    Mk = S @ Cs.T @ np.linalg.inv(Cs @ S @ Cs.T + v_meas*np.eye(2))/k      # gain on the unscaled innovation
    return {'K': Kf, 'Mk': Mk}


def margins(m, g, nw=6000):
    """Loop broken at the pulse input.  Returns nominal stability, the modulus margin min|1+L|, and
    1/max|T|: the largest fractional error in the plant response (any frequency, any phase) that the
    loop is guaranteed to tolerate."""
    A, B, C, K, Mk = m['A'], m['B'], m['C'], g['K'], g['Mk']
    n = A.shape[0]
    I = np.eye(n)
    Ac = (A - B @ K) @ (I - Mk @ C); Bc = (A - B @ K) @ Mk
    Cc = -K @ (I - Mk @ C); Dc = -K @ Mk
    cl = np.block([[A, B @ Cc], [Bc @ C, Ac]]) + np.block([[B @ Dc @ C, np.zeros((n, n))], [np.zeros((n, n)), np.zeros((n, n))]])
    rad = float(np.abs(np.linalg.eigvals(cl)).max())
    w = np.concatenate([np.logspace(-4, math.log10(math.pi), nw), np.linspace(0.01, math.pi, nw)])
    Tm, Sm = 0.0, np.inf
    for wi in w:
        z = np.exp(1j*wi)
        G = C @ np.linalg.solve(z*I - A, B)
        Kc = Cc @ np.linalg.solve(z*I - Ac, Bc) + Dc
        L = -(Kc @ G)[0, 0]
        Tm = max(Tm, abs(L/(1 + L))); Sm = min(Sm, abs(1 + L))
    return {'closed_loop_spectral_radius': rad, 'stable': rad < 1.0 + 1e-9, 'modulus_margin': float(Sm),
            'T_peak': float(Tm), 'tolerated_fractional_plant_error': float(1/Tm)}


class LQG:
    def __init__(self, m, g, tm_on=10.0, a_max=0.02):
        self.m, self.g, self.tm_on, self.a_max = m, g, tm_on, a_max
        self.xbar = np.zeros(m['A'].shape[0])
        self.launch, self.sat, self.log = [], 0, []

    def decide(self, t, hist):
        m, g = self.m, self.g
        tm = t - m['d']
        if tm < self.tm_on:
            return None
        i = min(int(round(tm/OUT_DT)), len(hist['t']) - 1)
        y = np.array([hist['R0'][i] - 1.0, hist['PR0'][i]])
        if not self.launch:
            # first reading: attribute the existing offset and drift to the disturbance states, so the
            # filter does not start with a large innovation
            self.xbar[m['idx']['Z']] = y[0]; self.xbar[m['idx']['V']] = y[1]
        xh = self.xbar + g['Mk'] @ (y - m['C'] @ self.xbar)
        u = float(-(g['K'] @ xh)[0])
        if abs(u) > self.a_max:
            u = math.copysign(self.a_max, u); self.sat += 1
        self.xbar = m['A'] @ xh + m['B'][:, 0]*u
        ix = m['idx']
        self.launch.append((t, u)); self.log.append((t, xh[ix['G']], xh[ix['Z']], xh[ix['F']], xh[ix['V']]))
        return u

    def __call__(self, t, x, U, hist):
        from control_feedback import inject
        from sysid_pair import W
        a = self.decide(t, hist)
        if a is not None:
            inject(x, U, a, self.m['d'], W)


if __name__ == '__main__':
    for d in (1.0, 1.5, 2.0, 3.0):
        m = build_model(d)
        print('d=%g: %d states, delay line %d, k = (%.4f, %.4f; lam*k_R = %.4f), tail fit rms %.1e, rate-channel rms %.1e, pure-delay bound e^{2 lam d} = %.1f' % (
            d, m['A'].shape[0], m['M'], m['k'][0], m['k'][1], m['lam']*m['k'][0], m['tail_fit_rms'], m['rate_channel_rms'], math.exp(2*m['lam']*d)))
        first = np.nonzero(np.abs(m['taps'][0]) > 1e-3*m['k'][0])[0][0]
        print('      first non-zero tap at lag %.2f b (response reaches the controller %.2f b after launch)' % (first*DT, first*DT))
