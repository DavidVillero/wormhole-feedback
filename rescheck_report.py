"""Compare the long closed-loop run at dx = 0.025 and dx = 0.0125.  Writes results_rescheck.json, control_rescheck.png."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = np.load('runs/reg_T225_gz0.05.npz'); F = np.load('runs/reg_long_dx0.0125.npz')
PC = np.load('runs/pair_response.npz'); PF = np.load('runs/pair_response_dx0.0125.npz')
BC = json.load(open('runs/balance_log.json')); BF = json.load(open('runs/balance_log_dx0.0125.json'))


def stats(R):
    t, dR, lt, la = R['t'], R['R0'] - 1, R['launch_t'], R['launch_a']
    late = t > 100
    return {'outcome': str(R['outcome']), 'time_reached': float(t[-1]),
            'radius_offset_at_t': {str(tt): float(np.interp(tt, t, dR)) for tt in (25, 50, 100, 150, 200, 225)},
            'drift_per_b_after_t100': float(np.polyfit(t[late], dR[late], 1)[0]),
            'max_abs_radius_offset': float(np.abs(dR).max()),
            'rms_pulse_amplitude_first_25': float(np.sqrt(np.mean(la[lt < 25]**2))),
            'rms_pulse_amplitude_last_25': float(np.sqrt(np.mean(la[lt >= 200]**2))),
            'max_pulse_amplitude': float(np.abs(la).max()),
            'max_H_rms': float(np.nanmax(R['H'])), 'max_M_rms': float(np.nanmax(R['M']))}

sc, sf = stats(C), stats(F)
m = np.arange(len(PC['t']))[(PC['t'] >= 4) & (PC['t'] <= 17)]
res = {'coarse_dx0.025': sc, 'fine_dx0.0125': sf,
       'ratios_coarse_over_fine': {
           'drift_after_t100': sc['drift_per_b_after_t100']/sf['drift_per_b_after_t100'],
           'radius_offset_at_225': sc['radius_offset_at_t']['225']/sf['radius_offset_at_t']['225'],
           'max_H_rms': sc['max_H_rms']/sf['max_H_rms'],
           'balance_trim': (0.5*(BC['lo'] + BC['hi']))/(0.5*(BF['lo'] + BF['hi'])),
           'expected_for_second_order_error': 4.0},
       'calibration_agreement': {
           'cancelling_ratio_beta': [float(PC['beta']), float(PF['beta'])],
           'pair_response_max_abs_difference_4_to_17': float(np.abs(PC['odd'][m, 0] - PF['odd'][m, 0]).max()),
           'pair_response_typical_size': float(np.abs(PC['odd'][m, 0]).mean())}}
json.dump(res, open('results_rescheck.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
a = ax[0]
a.plot(C['t'], C['R0'] - 1, color='#92c5de', label='dx = 0.025')
a.plot(F['t'], F['R0'] - 1, color='#08306b', label='dx = 0.0125')
a.plot(F['t'], 4*(F['R0'] - 1), color='#08306b', ls=':', lw=1, label='4 × (dx = 0.0125)')
a.set_xlabel('t / b'); a.set_ylabel('throat radius − b'); a.set_title('Radius drift under control'); a.legend(fontsize=8)
a = ax[1]
a.semilogy(C['launch_t'], np.abs(C['launch_a']), '.', ms=2, color='#92c5de', label='dx = 0.025')
a.semilogy(F['launch_t'], np.abs(F['launch_a']), '.', ms=2, color='#08306b', label='dx = 0.0125')
a.set_xlabel('t / b'); a.set_ylabel('|pulse amplitude|'); a.set_title('Control effort'); a.legend(fontsize=8)
a = ax[2]
w = np.arange(len(PC['t']))[(PC['t'] >= 2.5) & (PC['t'] <= 18)]
a.plot(PC['t'][w], PC['odd'][w, 0], color='#92c5de', lw=3, label='dx = 0.025')
a.plot(PF['t'][w], PF['odd'][w, 0], color='#08306b', lw=1, label='dx = 0.0125')
a.set_xlabel('t / b'); a.set_ylabel('pair response of throat radius / a'); a.set_title('Neutral response (growing mode cancelled)'); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig('control_rescheck.png', dpi=150)
