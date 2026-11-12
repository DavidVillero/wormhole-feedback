"""Option 2 runs.   python3 control_lqg_run.py margins        (model-only: margin at each distance)
                    WH_D=2 python3 control_lqg_run.py run 150  (closed loop in the full simulation)"""
import json
import math
import os
import sys
import numpy as np
from control_lqg import build_model, design, margins, LQG, DT
from control_feedback import OUT_DT, TAG, DX

TUNE = dict(r_u=1e-3, q_g=1e2, q_f=1e-3, q_v=1e-5, q_z=1e-5, w_rate=1.0, w_radius=float(os.environ.get('WH_WR', 1.0)))

if __name__ == '__main__':
    if sys.argv[1] == 'margins':
        out = []
        for d in (1.0, 1.5, 2.0, 3.0):
            m = build_model(d)
            mg = margins(m, design(m, **TUNE))
            first = int(np.nonzero(np.abs(m['taps'][0]) > 1e-3*m['k'][0])[0][0])*DT
            row = {'d': d, 'states': int(m['A'].shape[0]), 'effective_loop_delay': first, 'growth_over_effective_delay': math.exp(m['lam']*first),
                   'growth_over_2d': math.exp(2*m['lam']*d), **mg}
            out.append(row)
            print('d=%g: loop delay %.2f b (growth x%.1f), stable %s, T peak %.1f, tolerated plant error %.3f%%, modulus margin %.4f' % (
                d, first, row['growth_over_effective_delay'], mg['stable'], mg['T_peak'], 100*mg['tolerated_fractional_plant_error'], mg['modulus_margin']))
        json.dump({'tuning': TUNE, 'rows': out}, open('results_lqg_margins.json', 'w'), indent=1)
    else:
        from control_solver import grid, initial_state, evolve
        d = float(os.environ.get('WH_D', 1.0)); T = float(sys.argv[2])
        m = build_model(d)
        ctl = LQG(m, design(m, **TUNE))
        b = json.load(open('runs/balance_log%s.json' % TAG))
        x = grid(DX, T + d + 16.0)
        r = evolve(x, initial_state(x, s_amp=0.5*(b['lo'] + b['hi'])), T, out_dt=OUT_DT, controller=ctl, ctrl_dt=DT)
        L = np.array(ctl.launch); h = r['hist']; lg = np.array(ctl.log)
        np.savez('runs/lqg_d%g_wr%g%s.npz' % (d, TUNE['w_radius'], TAG), t=h['t'], R0=h['R0'], H=h['H'], launch_t=L[:, 0], launch_a=L[:, 1], log=lg,
                 outcome=r['outcome'], lifetime=r['lifetime'], k=m['k'][0], d=d)
        t, dR = h['t'], h['R0'] - 1
        print('d=%g LQG: %s at %.1f, saturated %d, max H %.1e' % (d, r['outcome'], r['lifetime'], ctl.sat, np.nanmax(h['H'])))
        for a0 in range(0, int(t[-1]), 25):
            w = (t >= a0) & (t < a0 + 25); lw = (L[:, 0] >= a0) & (L[:, 0] < a0 + 25)
            if lw.any():
                print('   t %3d-%3d: radius offset %+.2e .. %+.2e   k*a rms %.1e' % (a0, a0 + 25, dR[w].min(), dR[w].max(), m['k'][0]*np.sqrt(np.mean(L[lw, 1]**2))))
