"""Closed-loop references at dx = 0.025 (the resolution the controller was calibrated at)."""
import json
import numpy as np
from control_solver import grid, initial_state, evolve
from viz_reference import Ctl, pack
b = json.load(open('runs/balance_log.json')); TRIM = 0.5*(b['lo'] + b['hi'])
out = {'dx': 0.025, 'trim': TRIM}
x = grid(0.025, 60.0)
for d, T, dist, name in ((1.0, 50.0, None, 'd1'), (1.0, 50.0, (20.0, 'normal', 0.003, 6.0, 1.0), 'd1_matter'), (1.5, 50.0, None, 'd1.5')):
    c = Ctl(d, disturb=dist)
    out[name] = pack(evolve(x, initial_state(x, s_amp=TRIM), T, controller=c, ctrl_dt=0.25), c)
    print(name, out[name]['outcome'], out[name]['lifetime'], 'R0 end %.8f' % out[name]['R0'][-1])
json.dump(out, open('runs/viz_reference2.json', 'w'))
