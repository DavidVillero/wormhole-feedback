"""Numerical check of DELAY_BOUND.md: many stabilising LTI controllers, none beats the bound.

For each distance: random LQG weightings (log-uniform around the tuned values) and the tuned design with
its feedback gain or filter gain scaled.  Every design that stabilises the model is kept, and its peak
input complementary sensitivity is compared with the two-channel bound from results_bound_zeros.json.
Peaks are computed on a frequency grid, which can only underestimate the true peak, so a pass is
conservative.

    python3 delay_bound_check.py [n_per_distance]     # results_delay_bound_check.json
"""
import json
import math
import sys
import numpy as np
from control_lqg import build_model, design, margins
from control_lqg_run import TUNE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
bounds = {r['d']: r for r in json.load(open('results_bound_zeros.json'))}
rng = np.random.default_rng(20261125)
out = []
for d in (1.0, 2.0):
    m = build_model(d)
    lim = bounds[d]['bound_two_channel_pct']/100.0        # largest tolerated plant error any LTI controller can have
    rows = []
    for k in range(N):
        if k % 2 == 0:
            w = {key: val*10**rng.uniform(-1.5, 1.5) for key, val in TUNE.items()}
            g = design(m, **w); kind = 'random weights'
        else:
            g = design(m, **TUNE)
            g = {'K': g['K']*10**rng.uniform(-0.3, 0.3), 'Mk': g['Mk']*10**rng.uniform(-0.3, 0.3)}; kind = 'scaled gains'
        mg = margins(m, g, nw=1500)
        if not mg['stable']:
            rows.append({'kind': kind, 'stable': False}); continue
        tol = mg['tolerated_fractional_plant_error']
        rows.append({'kind': kind, 'stable': True, 'tolerated': tol, 'respects_bound': bool(tol <= lim*(1 + 1e-9))})
    st = [r for r in rows if r['stable']]
    best = max(r['tolerated'] for r in st) if st else None
    res = {'d': d, 'bound_pct': 100*lim, 'designs': N, 'stabilising': len(st),
           'violations': sum(not r['respects_bound'] for r in st), 'best_tolerated_pct': 100*best if best else None}
    out.append(res)
    print('d = %g: %d of %d designs stabilise; best tolerated plant error %.3f%% vs bound %.3f%%; violations %d' % (
        d, len(st), N, res['best_tolerated_pct'] or 0, res['bound_pct'], res['violations']))
json.dump(out, open('results_delay_bound_check.json', 'w'), indent=1)
