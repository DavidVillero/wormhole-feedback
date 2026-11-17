"""Transit study, gate test 0: the existing LQG regulator (d = 1, no model of the traveller)
against symmetric normal-matter travellers.

    python3 transit_gate.py baseline     # results_transit_baseline.json

A traveller is an ingoing normal-scalar pulse launched from |x| = X0 on both sides (Z2-symmetric),
present in the initial data, so it reaches the throat at t ~ X0, after the regulator has settled.
Start: grid-balanced wormhole.  Each amplitude is run with and without the regulator.
"""
import json
import sys
from multiprocessing import Pool
import numpy as np
from control_solver import grid, initial_state, evolve
from control_feedback import OUT_DT, TAG, DX
from control_lqg import build_model, design, LQG, DT
from control_lqg_run import TUNE
from control_steps123 import mass

D = 1.0
X0, WIDTH = 30.0, 1.0
T = 90.0
# amplitudes chosen for traveller masses of about 1e-6 ... 3e-2 b (mass ~ 1.42e4 * amp^2 at X0 = 30)
AMPS = (0.0, 8.4e-6, 2.65e-5, 8.4e-5, 2.65e-4, 8.4e-4, 1.45e-3)


def one(args):
    amp, controlled = args
    b = json.load(open('runs/balance_log%s.json' % TAG))
    s_amp = 0.5*(b['lo'] + b['hi'])
    x = grid(DX, T + X0 + 16.0)
    i_m = int(round((X0 + 8.0 - x[0])/DX))            # mass read outside the traveller, inside the boundary
    U0 = initial_state(x, s_amp=s_amp)
    pulses = [('normal', amp, X0, WIDTH)] if amp else []
    U = initial_state(x, pulses, s_amp=s_amp)
    dM = float(mass(U, i_m) - mass(U0, i_m))          # Misner-Sharp mass carried by one traveller
    ctl = None
    if controlled:
        m = build_model(D)
        ctl = LQG(m, design(m, **TUNE))
    r = evolve(x, U, T, out_dt=OUT_DT, controller=ctl, ctrl_dt=DT)
    h = r['hist']; t = h['t']; dR = h['R0'] - 1.0
    out = {'amp': amp, 'controlled': controlled, 'traveller_mass': dM, 'outcome': r['outcome'],
           'lifetime': float(r['lifetime']), 'max_H': float(np.nanmax(h['H'])), 'max_M': float(np.nanmax(h['M'])),
           'max_offset': float(np.abs(dR).max()), 'final_offset': float(dR[-1])}
    if ctl is not None and ctl.launch:
        L = np.array(ctl.launch)
        late = L[:, 0] >= r['lifetime'] - 15.0
        out.update(saturated=int(ctl.sat), peak_pulse=float(np.abs(L[:, 1]).max()),
                   rms_pulse_last15=float(np.sqrt(np.mean(L[late, 1]**2))), sum_abs_pulse=float(np.abs(L[:, 1]).sum()))
    return out


if __name__ == '__main__':
    # the uncontrolled balanced wormhole fails at ~24 b, before the traveller arrives at ~30 b, so only
    # one uncontrolled run is kept as a reference
    jobs = [(0.0, False)] + [(a, True) for a in AMPS]
    with Pool(2) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    json.dump({'d': D, 'x0': X0, 'width': WIDTH, 'T': T, 'dx': DX, 'rows': rows}, open('results_transit_baseline%s.json' % TAG, 'w'), indent=1)
    for r in rows:
        print('amp %-7g ctrl %-5s dM %.3e  %-8s at %5.1f  max|dR| %.2e  final dR %+.2e  sat %s  peak pulse %s  maxH %.1e' % (
            r['amp'], r['controlled'], r['traveller_mass'], r['outcome'], r['lifetime'], r['max_offset'], r['final_offset'],
            r.get('saturated', '-'), ('%.1e' % r['peak_pulse']) if 'peak_pulse' in r else '-', r['max_H']))
