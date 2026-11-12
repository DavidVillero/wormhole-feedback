"""Linearity of the throat's pulse response (paper Sec. 4.1).

Launches +a and -a phantom pulses from d = 1 b on the balanced wormhole, for a range of amplitudes, and compares
the odd part per unit amplitude (the linear response) and the even part (second order) at fixed times after
arrival. Writes results_linearity.json. Run: python3 linearity.py
"""
import json, numpy as np
from control_solver import grid, initial_state, evolve
from control_feedback import inject, DX, OUT_DT
from sysid import s_amp, outputs
d, w, T = 1.0, 0.3, 13.0
x = grid(DX, T + d + 16.0)
S = [2.0, 3.0, 5.0, 7.0]                      # time since arrival at the throat
idx = [int(round((d + s)/OUT_DT)) for s in S]
out = {'d': d, 's': S, 'rows': []}
ref = None
for a in (1e-8, 1e-6, 1e-5, 3e-5, 1e-4):
    ys = []
    for sgn in (1, -1):
        U = initial_state(x, s_amp=s_amp())
        inject(x, U, sgn*a, d, w)
        ys.append(outputs(evolve(x, U, T, out_dt=OUT_DT)['hist'])[:, 0])
    n = min(len(ys[0]), len(ys[1]))
    odd = 0.5*(ys[0][:n] - ys[1][:n])/a
    base = None
    even = 0.5*(ys[0][:n] + ys[1][:n])
    g = lambda arr, i: float(arr[i]) if i < len(arr) else None
    row = {'a': a, 'odd_per_unit_a': [g(odd, i) for i in idx], 'throat_offset': [None if g(odd, i) is None else a*g(odd, i) for i in idx]}
    if ref is None: ref = odd
    else: row['rel_change_vs_smallest'] = [None if i >= len(odd) else float(odd[i]/ref[i] - 1) for i in idx]
    out['rows'].append(row); out.setdefault('even_raw', []).append([g(even, i) for i in idx])
    print(row)
# second-order part: even(a) - even(smallest a) ~ c2 a^2
e0 = out['even_raw'][0]
for row, e in zip(out['rows'], out['even_raw']):
    row['second_order_rel'] = [None if ei is None else float((ei - e0i)/(row['a']*oi)) for ei, e0i, oi in zip(e, e0, row['odd_per_unit_a'])]
json.dump(out, open('results_linearity.json', 'w'), indent=1)
for r in out['rows']: print(r['a'], r.get('rel_change_vs_smallest'), r['throat_offset'])
