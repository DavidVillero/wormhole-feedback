"""System identification report and regulator runs.  Writes results_sysid.json and control_sysid.png."""
import json
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from control_regulator import load_model, simple_model, run_on_model, Regulator, full_run
from control_feedback import Feedback, calibrate, closed_loop

D = 1.0
RUNS = {'identified regulator (estimator 2)': dict(order=2),
        'identified regulator (estimator 1)': dict(order=1),
        'estimator 2 + radius feedback (gp=-0.1, gv=0.3)': dict(order=2, gp=-0.1, gv=0.3)}


def reg_run(item):
    name, kw = item
    kw = dict(kw); order = kw.pop('order')
    m = load_model(D, order=order)
    r = full_run(m, Regulator(m, kappa=0.5, **kw), 80.0)
    return name, {k: r[k] for k in ('outcome', 'lifetime', 't', 'R0', 'launch_t', 'launch_a', 'H', 'M', 'saturated')}


def reproduce(_):
    b = json.load(open('runs/balance_log.json')); s = 0.5*(b['lo'] + b['hi'])
    cal = calibrate(D, 0.3, s, order=1)
    sm = simple_model(D)
    rm = run_on_model(sm, Feedback(dict(cal), kappa=0.5), 30.0)
    rf = closed_loop(d=D, w=0.3, T=30.0, kappa=0.5, trimmed=True, cal=cal)
    return sm, rm, rf


if __name__ == '__main__':
    pool = Pool(4)
    rep = pool.apply_async(reproduce, (0,))
    runs = dict(pool.map(reg_run, list(RUNS.items())))
    sm, rm, rf = rep.get()
    m = load_model(D, order=2)
    n = min(len(rm['launch_a']), len(rf['launch_a']))
    tt = rf['launch_t'][:n]
    am, af = rm['launch_a'][:n], rf['launch_a'][:n]
    rel = lambda w: float(np.sqrt(np.mean((am[w] - af[w])**2)/np.mean(af[w]**2)))
    res = {'distance': D,
           'identified_model': {'growth_rate': m['lam'], 'growing_gain_k': sm['k'],
                                'neutral_radius_shift_per_unit_pulse_at_s3': float(sm['plateau'][0]),
                                'slow_tail_fit': m['tail_R'],
                                'note': 'slow tail hit the lower bound of the fitted rate; it is not resolved by 6 b of data'},
           'failure_reproduction': {'step4_controller_lifetime_on_model': float(rm['lifetime']),
                                    'step4_controller_lifetime_full_simulation': float(rf['lifetime']),
                                    'pulse_amplitude_rms_relative_difference': {
                                        't<8': rel(tt < 8), 't<14': rel(tt < 14)}},
           'regulator_runs_raw_wormhole': {k: {'outcome': v['outcome'], 'lifetime': v['lifetime'],
                                               'max_H_rms': float(np.nanmax(v['H'])), 'saturated_steps': v['saturated']}
                                           for k, v in runs.items()},
           'reference': {'no_control': 7.4, 'best_step4': 23.9, 'open_loop_ceiling': 37.0}}
    json.dump(res, open('results_sysid.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    a = ax[0]
    Dz = np.load('runs/sysid_d%g.npz' % D); h = Dz['h']; t = 0.05*np.arange(len(h)); s = t - D
    ng = h[:, 0] - sm['k']*np.exp(m['lam']*s)
    w = (s > -0.5) & (s < 6.0)
    a.plot(s[w], ng[w], color='#2166ac'); a.axhline(0, color='k', lw=0.5)
    a.set_ylim(-1, 4); a.set_xlabel('time since pulse arrival / b'); a.set_ylabel('throat radius response minus growing mode')
    a.set_title('A pulse also shifts the throat permanently')
    a = ax[1]
    a.semilogy(rf['launch_t'], np.abs(rf['launch_a']), '.', ms=4, color='#b2182b', label='full simulation')
    a.semilogy(rm['launch_t'], np.abs(rm['launch_a']), '-', lw=1, color='k', label='two-term model')
    a.set_xlim(0, 17); a.set_xlabel('t / b'); a.set_ylabel('|pulse amplitude| (step-4 controller)')
    a.set_title('The model reproduces the step-4 failure'); a.legend(fontsize=8)
    a = ax[2]
    for (name, v), col in zip(runs.items(), ('#2166ac', '#92c5de', '#1b7837')):
        a.plot(v['t'], v['R0'], color=col, label='%s: %.0f b' % (name, v['lifetime']))
    a.axvline(7.4, color='#b2182b', ls=':', lw=1); a.axvline(37.0, color='k', ls=':', lw=1)
    a.set_ylim(0.5, 2.0); a.set_xlabel('t / b'); a.set_ylabel('throat radius / b')
    a.set_title('Redesigned regulator (dotted: 7.4 and 37 b)'); a.legend(fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig('control_sysid.png', dpi=150)
