"""Step 4 status: representative closed-loop runs, summary JSON and a diagnostic figure."""
import json
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from control_feedback import closed_loop, summary

CASES = {
    'no control, raw wormhole':            None,
    'd=1, estimator 1, raw':               dict(d=1.0, w=0.3, order=1, trimmed=False),
    'd=1, estimator 2, raw':               dict(d=1.0, w=0.3, order=2, trimmed=False),
    'd=1, estimator 1, grid-balanced':     dict(d=1.0, w=0.3, order=1, trimmed=True),
    'd=1, estimator 2, grid-balanced':     dict(d=1.0, w=0.3, order=2, trimmed=True),
    'd=2, estimator 2, raw':               dict(d=2.0, w=0.5, order=2, trimmed=False),
    'd=3, estimator 2, raw':               dict(d=3.0, w=0.5, order=2, trimmed=False),
}


def go(item):
    name, kw = item
    if kw is None:
        from control_solver import grid, initial_state, evolve
        x = grid(0.025, 40.0)
        r = evolve(x, initial_state(x), 25.0)
        return name, {'outcome': r['outcome'], 'lifetime': float(r['lifetime'])}, None
    r = closed_loop(T=45.0, kappa=0.5, **kw)
    keep = {'t': r['hist']['t'], 'R0': r['hist']['R0'], 'S0': r['hist']['S0'], 'H': r['hist']['H'],
            'lt': r['launch_t'], 'la': r['launch_a']}
    return name, summary(r), keep


if __name__ == '__main__':
    out, series = {}, {}
    for name, s, keep in Pool(4).map(go, list(CASES.items())):
        out[name] = s; series[name] = keep
    ol = json.load(open('results_control.json'))
    res = {'closed_loop_runs': out,
           'reference_open_loop': {'raw_wormhole_lifetime': ol['balance']['lifetime_untuned'],
                                   'grid_balanced_lifetime_no_control': ol['balance']['longest_lifetime']},
           'status': 'not stabilised: every closed-loop run eventually collapses or expands'}
    json.dump(res, open('results_step4.json', 'w'), indent=1)
    for k, v in out.items():
        print('%-34s %-8s lifetime %.1f' % (k, v['outcome'], v['lifetime']))
    k = series['d=1, estimator 1, grid-balanced']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(k['t'], k['R0'] - 1, color='#2166ac'); ax[0].set_ylim(-0.02, 0.02); ax[0].set_xlim(0, 14)
    ax[0].set_xlabel('t / b'); ax[0].set_ylabel('throat radius − b'); ax[0].set_title('Throat radius is held within 1% for a while...')
    ax[1].semilogy(k['lt'], np.abs(k['la']), '.', ms=3, color='#b2182b'); ax[1].set_xlim(0, 16)
    ax[1].set_xlabel('t / b'); ax[1].set_ylabel('|pulse amplitude|'); ax[1].set_title('...but the pulses needed keep growing')
    ax[2].semilogy(k['t'], np.abs(k['S0']), color='#1b7837'); ax[2].set_xlim(0, 16)
    ax[2].set_xlabel('t / b'); ax[2].set_ylabel('|conformal factor σ at throat|'); ax[2].set_title('and the conformal factor runs away with them')
    fig.tight_layout(); fig.savefig('control_step4_diagnostic.png', dpi=150)
