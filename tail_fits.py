"""Power-law fit of the neutral (non-growing) throat shift at each controller distance (paper Sec. 4.2).

Fits n(s) = n_inf - c (s + s0)^(-alpha) to the cancelled pulse-pair data over 3 <= s <= 16 b
(control_regulator.fit_power_tail). Writes results_tail_fits.json. Run: python3 tail_fits.py
"""
import json
import numpy as np
from control_regulator import fit_power_tail
out = []
for d in (1.0, 1.5, 2.0, 3.0):
    p = fit_power_tail(d)
    out.append({'d': d, **{k: float(p[k]) for k in ('n_inf', 'c', 's0', 'alpha', 'rms')}})
    print(out[-1])
json.dump(out, open('results_tail_fits.json', 'w'), indent=1)
