"""Summary of the long closed-loop run.  Writes results_regulator.json and control_regulator.png."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from control_regulator import fit_power_tail, load_model

R = np.load('runs/reg_T225_gz0.05.npz'); R0 = np.load('runs/reg_T225_gz0.npz')
t, dR = R['t'], R['R0'] - 1
lt, la = R['launch_t'], R['launch_a']
win = []
for a0 in range(0, 225, 25):
    w = (t >= a0) & (t < a0 + 25); lw = (lt >= a0) & (lt < a0 + 25)
    win.append({'t': [a0, a0 + 25], 'max_abs_radius_offset': float(np.abs(dR[w]).max()), 'rms_pulse_amplitude': float(np.sqrt(np.mean(la[lw]**2)))})
late = t > 100
drift = float(np.polyfit(t[late], dR[late], 1)[0])
m = load_model(1.0, order=2, power_tail=True)
res = {'controller': {'distance': 1.0, 'pulse_width': 0.3, 'update_interval': 0.25, 'kappa': 0.5, 'gz': 0.05, 'estimator_order': 2,
                      'loop_delay': 2.0, 'growth_over_loop_delay': float(np.exp(2*m['lam']))},
       'neutral_tail_fit': fit_power_tail(1.0),
       'long_run': {'outcome': str(R['outcome']), 'coordinate_time': float(R['lifetime']), 'throat_proper_time': float(R['tau']),
                    'max_abs_radius_offset': float(np.abs(dR).max()), 'radius_drift_per_b_after_t100': drift,
                    'max_pulse_amplitude': float(np.abs(la).max()), 'sum_abs_pulse_amplitude': float(np.abs(la).sum()),
                    'max_H_rms': float(np.nanmax(R['H'])), 'max_M_rms': float(np.nanmax(R['M'])), 'windows': win},
       'without_radius_setpoint_gz0': {'outcome': str(R0['outcome']), 'lifetime': float(R0['lifetime'])},
       'reference_lifetimes': {'no_control': 7.4, 'step4_best': 23.9, 'open_loop_ceiling': 37.0, 'previous_regulator': 50.9},
       'traveller_test_normal_pulse_from_x6': {'amp 0.0005': 'held for 60 b, max radius offset 0.007',
                                               'amp 0.002': 'lost (expands at 17.5 b; kappa 0.25: 32 b; kappa 0.1: collapses at 10 b)',
                                               'amp 0.005-0.02': 'lost within 7-11 b'}}
json.dump(res, open('results_regulator.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
a = ax[0]
a.plot(t, R['R0'], color='#2166ac', label='regulator (gz = 0.05)')
a.plot(R0['t'], R0['R0'], color='#92c5de', lw=1, label='regulator without radius set-point')
for v, lab in ((7.4, 'no control'), (37.0, 'open-loop ceiling')):
    a.axvline(v, color='k', ls=':', lw=1)
a.set_ylim(0.9, 1.1); a.set_xlabel('t / b'); a.set_ylabel('throat radius / b')
a.set_title('Throat held for 225 b (dotted: 7.4 and 37 b)'); a.legend(fontsize=8)
a = ax[1]
a.semilogy(lt, np.abs(la), '.', ms=2, color='#b2182b')
a.set_xlabel('t / b'); a.set_ylabel('|pulse amplitude|'); a.set_title('Control effort decays')
a = ax[2]
a.semilogy(t, R['H'], color='#555555', label='H'); a.semilogy(t, R['M'], color='#555555', ls=':', label='M')
a.set_xlabel('t / b'); a.set_ylabel('rms constraint residual, |x| < 10 b'); a.set_title('Constraints stay at truncation level'); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig('control_regulator.png', dpi=150)
