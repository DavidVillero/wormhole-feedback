"""Long-window measurement of the non-growing throat response.

A pulse of amplitude a is launched at t = 0 and a second one of amplitude -beta*a at t = 1.
beta is tuned by section search until the growing mode is cancelled to round-off, which leaves
~30 b of clean data on everything else.  Done for +a and -a; the odd part removes the baseline.

    python3 sysid_pair.py search +1     (then -1)
    python3 sysid_pair.py fit
"""
import json
import sys
from multiprocessing import Pool
import numpy as np
from control_solver import grid, initial_state, evolve
from control_feedback import inject, DX, OUT_DT, TAG

import os
D = float(os.environ.get('WH_D', 1.0))          # controller distance; set WH_D for the distance sweep
T0 = float(os.environ.get('WH_T0', 0.0))        # launch time of the first pulse (0 = on the unsettled initial data)
W, A, GAP, T = 0.3, 1e-6, 1.0, 44.0 + D + T0
DTAG = ('' if D == 1.0 else '_d%g' % D) + ('' if T0 == 0.0 else '_t%g' % T0)
KEYS = ('R0', 'PR0', 'S0', 'PS0')


def s_amp():
    b = json.load(open('runs/balance_log%s.json' % TAG))
    return 0.5*(b['lo'] + b['hi'])


def pair_run(args):
    sgn, beta = args
    x = grid(DX, T + D + 16.0)
    U = initial_state(x, s_amp=s_amp())

    def ctl(t, x_, U_, hist):
        if abs(t - T0) < 1e-9:
            inject(x_, U_, sgn*A, D, W)
        elif abs(t - T0 - GAP) < 1e-9:
            inject(x_, U_, -sgn*beta*A, D, W)
    r = evolve(x, U, T, out_dt=OUT_DT, controller=ctl, ctrl_dt=0.25)
    return beta, r['outcome'], float(r['lifetime']), {k: r['hist'][k].tolist() for k in KEYS}


if __name__ == '__main__':
    if sys.argv[1] == 'search':
        sgn = int(sys.argv[2])
        pool = Pool(4)
        ends = pool.map(pair_run, [(sgn, 3.0), (sgn, 3.6)])
        oc = {e[0]: e[1] for e in ends}
        assert set(oc.values()) == {'collapse', 'expand'}, oc
        lo, hi = (3.0, 3.6) if oc[3.0] == 'collapse' else (3.6, 3.0)       # lo collapses, hi expands
        best = max(ends, key=lambda e: e[2])
        for it in range(24):
            out = pool.map(pair_run, [(sgn, lo + (hi - lo)*k/5.0) for k in (1, 2, 3, 4)])
            for o in out:
                if o[2] > best[2]:
                    best = o
                if o[1] == 'expand':
                    hi = o[0]; break
                lo = o[0]
            if abs(hi - lo) < 4e-16*abs(hi):
                break
        json.dump({'sgn': sgn, 'beta': best[0], 'bracket': [lo, hi], 'lifetime': best[2], 'outcome': best[1], 'hist': best[3]},
                  open('runs/pair_%+d.json' % sgn, 'w'))
        print('sign %+d: beta = %.12f, lifetime %.2f (%s)' % (sgn, best[0], best[2], best[1]))
    else:
        P = {s: json.load(open('runs/pair_%+d.json' % s)) for s in (1, -1)}
        print('beta +: %.9f   beta -: %.9f   e^lam = %.6f' % (P[1]['beta'], P[-1]['beta'], np.exp(1.181997*GAP)))
        n = min(len(P[1]['hist']['R0']), len(P[-1]['hist']['R0']))
        y = {s: np.array([P[s]['hist'][k][:n] for k in KEYS]).T for s in (1, -1)}
        odd = 0.5*(y[1] - y[-1])/A
        even = 0.5*(y[1] + y[-1]); even[:, 0] -= 1.0
        t = OUT_DT*np.arange(n)
        np.savez('runs/pair_response.npz', t=t, odd=odd, even=even, beta=0.5*(P[1]['beta'] + P[-1]['beta']))
        for tt in list(np.arange(2, 12, 1.0)) + list(np.arange(12, t[-1], 2.0)):
            i = int(round(tt/OUT_DT))
            print('t=%5.1f  pair response/a: dR0 %+.4f  R0_t %+.5f  sigma0 %+.4f  sigma0_t %+.5f   | baseline dR0 %+.2e' % (tt, *odd[i], even[i, 0]))
