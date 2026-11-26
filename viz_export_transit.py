"""Export the one-sided transit calibration (throat-tracked) and the d = 1 controller for the live page.
    WH_SIDES=1 WH_TRACK=1 WH_MCAL=0.005 python3 viz_export_transit.py     # writes viz_transit.json
"""
import os
os.environ.setdefault('WH_SIDES', '1'); os.environ.setdefault('WH_TRACK', '1'); os.environ.setdefault('WH_MCAL', '0.005')
import json
import numpy as np
import transit_ff as TF
import transit_sensed as TS
from control_lqg import build_model, design
from control_lqg_run import TUNE

m = build_model(TF.D); g = design(m, **TUNE)
sig = TF.load_signature(); cal = json.load(open(TS.SCAL))
s = sig['s']
assert np.allclose(np.diff(s), s[1] - s[0])
i, j = np.nonzero(m['A'])
r = lambda a: [float('%.10g' % v) for v in a]
out = {'d': TF.D, 'x0': TF.X0, 'width': TF.WIDTH, 'lam': float(m['lam']), 'trim': TF.s_amp_balanced(),
       'xs': cal['xs'], 'kappa': cal['kappa'], 'delta': cal['delta'], 'sLo': TF.S_LO, 'sHi': TF.S_HI,
       'cpts': [[float(a), float(b)] for a, b in sig['c_points']],
       'sig': {'s0': float(s[0]), 'ds': float(s[1] - s[0]), 'ng': [r(sig['ng'][0]), r(sig['ng'][1])], 'kern': [r(sig['kern'][0]), r(sig['kern'][1])]},
       'ctrl': {'n': int(m['A'].shape[0]), 'A': [[int(a), int(b), float(m['A'][a, b])] for a, b in zip(i, j)], 'B': [float(v) for v in m['B'][:, 0]],
                'C': [[float(v) for v in row] for row in m['C']], 'K': [float(v) for v in g['K'][0]], 'Mk': [[float(v) for v in row] for row in g['Mk']],
                'iZ': int(m['idx']['Z']), 'iV': int(m['idx']['V'])}}
json.dump(out, open('viz_transit.json', 'w'), separators=(',', ':'))
print('viz_transit.json', os.path.getsize('viz_transit.json'), 'bytes; n =', out['ctrl']['n'])
