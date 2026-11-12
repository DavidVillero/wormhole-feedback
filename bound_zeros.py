"""Delay bound with the plant's right-half-plane zero (paper Sec. 7).

For each controller distance, builds the LQG plant model (control_lqg.build_model), finds the exact dead time
(first non-zero Markov parameter), the transmission zeros of each output channel, and the zeros the two channels
share. Interpolation bound on the input complementary sensitivity: ||T|| >= |p|^m * prod |(1 - conj(z) p)/(p - z)|
over zeros outside the unit circle (p = e^(lam h), m = dead time in samples). Writes results_bound_zeros.json.
"""
import json, math, numpy as np, scipy.linalg as sl
from control_lqg import build_model
R = json.load(open("results_lqg_margins.json"))["rows"]
ach = {r['d']: 100*r['tolerated_fractional_plant_error'] for r in R}
rows = []
for d in (1.0, 1.5, 2.0, 3.0):
    m = build_model(d); A, B, C = m['A'], m['B'], m['C']; n = A.shape[0]
    p = math.exp(m['lam']*0.25)
    def zeros(Cc):
        M1 = np.block([[A, B], [Cc, np.zeros((1, 1))]]); M2 = np.zeros_like(M1); M2[:n, :n] = np.eye(n)
        w = sl.eigvals(M1, M2); return w[np.isfinite(w)]
    def reldeg(Cc):
        x = B.copy()
        for k in range(n + 1):
            if abs((Cc @ x)[0, 0]) > 1e-12: return k + 1
            x = A @ x
    zr, zt = zeros(C[0:1]), zeros(C[1:2])
    rd = min(reldeg(C[0:1]), reldeg(C[1:2]))
    # zeros common to both channels (within 1e-3 relative), outside the unit circle
    common = [z for z in zr if abs(z) > 1 + 1e-9 and min(abs(zt - z)) < 1e-3*abs(z)]
    fz = 1.0
    for z in common: fz *= abs((1 - np.conj(z)*p)/(p - z))
    full = lambda zs, rdc: 100/(p**rdc*np.prod([abs((1 - np.conj(z)*p)/(p - z)) for z in zs if abs(z) > 1 + 1e-9]))
    rows.append({'d': d, 'common_zeros_z': [complex(round(z.real, 4), round(z.imag, 4)) for z in common],
                 'zero_s': [round(math.log(abs(z))/0.25, 3) for z in common],
                 'zero_factor': round(fz, 3), 'dead_time_samples': rd,
                 'bound_paper_pct': round(100*math.exp(-m['lam']*(2*d - 0.5)), 3),
                 'bound_exact_delay_pct': round(100/p**rd, 3),
                 'bound_two_channel_pct': round(100/(p**rd*fz), 3),
                 'bound_radius_only_pct': round(full(zr, reldeg(C[0:1])), 3),
                 'bound_rate_only_pct': round(full(zt, reldeg(C[1:2])), 3),
                 'lqg_achieved_pct': round(ach[d], 3)})
for r in rows: print(r)
json.dump([{k: (float(v) if hasattr(v, 'dtype') else ([[z.real, z.imag] for z in v] if k == 'common_zeros_z' else v)) for k, v in r.items()} for r in rows], open('results_bound_zeros.json', 'w'), indent=1)
