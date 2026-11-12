"""Option 2 summary.  Writes results_lqg.json and control_lqg.png."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

M = json.load(open('results_lqg_margins.json'))
runs = {}
for d in (1.0, 1.5, 2.0):
    L = np.load('runs/lqg_d%g_wr1.npz' % d)
    P = np.load('runs/sweep_d%g_o3_on10_gp0_gz0_k0.5_cut2.npz' % d)
    t, dR = L['t'], L['R0'] - 1
    late = t > 0.6*t[-1]
    runs[str(d)] = {'lqg': {'outcome': str(L['outcome']), 'lifetime': float(L['lifetime']), 'run_length': float(t[-1]),
                            'max_abs_radius_offset': float(np.abs(dR).max()) if str(L['outcome']) == 'alive' else None,
                            'radius_offset_range_last_40pct': [float(dR[late].min()), float(dR[late].max())] if str(L['outcome']) == 'alive' else None},
                    'predictor_estimator3_gain0.5': {'outcome': str(P['outcome']), 'lifetime': float(P['lifetime'])}}
res = {'design': {'model': 'delay line with measured taps + growing, neutral and 4 slow states + 2 disturbance states (28-44 states)',
                  'controller': 'steady-state Kalman filter + LQR, update every 0.25 b, readings: throat radius offset and its rate, delayed by d',
                  'tuning': M['tuning'], 'start': 'grid-balanced wormhole, readings before t = 10 b ignored, dx = 0.025'},
       'margins_from_model': [{'d': r['d'], 'effective_loop_delay': r['effective_loop_delay'],
                               'tolerated_fractional_plant_error': r['tolerated_fractional_plant_error'],
                               'best_possible_for_this_delay': 1/r['growth_over_effective_delay'],
                               'T_peak': r['T_peak'], 'T_peak_lower_bound': r['growth_over_effective_delay']} for r in M['rows']],
       'bound': 'any stabilising controller has peak complementary sensitivity >= exp(growth rate x loop delay), so the tolerated fractional plant error is at most exp(-growth rate x loop delay)',
       'full_simulation': runs}
json.dump(res, open('results_lqg.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
a = ax[0]
ds = [r['d'] for r in M['rows']]
a.semilogy(ds, [100*r['tolerated_fractional_plant_error'] for r in M['rows']], 'o-', color='#2166ac', label='this design')
a.semilogy(ds, [100/r['growth_over_effective_delay'] for r in M['rows']], 'k--', label='best possible for the delay')
a.set_xlabel('controller distance d / b'); a.set_ylabel('tolerated error in plant response (%)'); a.set_title('Robustness margin against distance'); a.legend(fontsize=8)
for j, d in enumerate((1.5, 2.0)):
    a = ax[j + 1]
    L = np.load('runs/lqg_d%g_wr1.npz' % d); P = np.load('runs/sweep_d%g_o3_on10_gp0_gz0_k0.5_cut2.npz' % d)
    a.plot(P['t'], P['R0'] - 1, color='#b2182b', lw=1, label='predictor (earlier design)')
    a.plot(L['t'], L['R0'] - 1, color='#2166ac', label='observer-based')
    a.set_ylim(-0.005, 0.03); a.set_xlabel('t / b'); a.set_ylabel('throat radius − b'); a.set_title('d = %g b' % d); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig('control_lqg.png', dpi=150)
