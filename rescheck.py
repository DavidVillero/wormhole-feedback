"""Resolution check: redo calibration and the long closed-loop run at another grid spacing.

    WH_DX=0.0125 python3 rescheck.py balance      # repeat each stage until it prints DONE
    WH_DX=0.0125 python3 rescheck.py sysid
    WH_DX=0.0125 python3 rescheck.py pair +1
    WH_DX=0.0125 python3 rescheck.py pair -1
    WH_DX=0.0125 python3 rescheck.py fit
    WH_DX=0.0125 python3 rescheck.py long 225

Every stage checkpoints and resumes, because one call is limited to a few minutes of wall time.
"""
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
import numpy as np
from control_solver import grid, initial_state, evolve
from control_feedback import DX, OUT_DT, TAG

BUDGET = 140.0
T0 = time.time()


def trimmed(s):
    x = grid(DX, 62.0)
    r = evolve(x, initial_state(x, s_amp=s), 45.0)
    return s, r['outcome'], float(r['lifetime']), None


def search(name, fn, lo, hi, pool, wrap=lambda v: v):
    """Resumable 5-way section search; lo must collapse and hi expand."""
    f = 'runs/search_%s%s.pkl' % (name, TAG)
    if os.path.exists(f):
        st = pickle.load(open(f, 'rb'))
    else:
        ends = pool.map(fn, [wrap(lo), wrap(hi)])
        oc = [e[1] for e in ends]
        assert set(oc) == {'collapse', 'expand'}, oc
        if oc[0] == 'expand':
            lo, hi, ends = hi, lo, ends[::-1]
        st = {'lo': lo, 'hi': hi, 'best': max(ends, key=lambda e: e[2]), 'it': 0, 'done': False}
    while not st['done'] and time.time() - T0 < BUDGET:
        out = pool.map(fn, [wrap(st['lo'] + (st['hi'] - st['lo'])*k/5.0) for k in (1, 2, 3, 4)])
        for o in out:
            if o[2] > st['best'][2]:
                st['best'] = o
            if o[1] == 'expand':
                st['hi'] = o[0]; break
            st['lo'] = o[0]
        st['it'] += 1
        st['done'] = st['it'] >= 26 or abs(st['hi'] - st['lo']) < 4e-16*max(abs(st['hi']), abs(st['lo']))
        pickle.dump(st, open(f, 'wb'))
    print('%s: iteration %d, bracket width %.1e, best lifetime %.1f  %s' % (
        name, st['it'], abs(st['hi'] - st['lo']), st['best'][2], 'DONE' if st['done'] else '(run again to continue)'))
    return st


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'balance':
        st = search('balance', trimmed, 0.0, -1e-3*(DX/0.025)**2*2, Pool(4))
        if st['done']:
            json.dump({'lo': st['lo'], 'hi': st['hi']}, open('runs/balance_log%s.json' % TAG, 'w'))
            print('trim', 0.5*(st['lo'] + st['hi']))
    elif stage == 'sysid':
        import sysid
        h = sysid.record(1.0, 0.3)
        print('recorded', h.shape, 'DONE')
    elif stage == 'pair':
        import sysid_pair
        sgn = int(sys.argv[2])

        def fn(args):
            beta, oc, life, hist = sysid_pair.pair_run(args)
            return beta, oc, life, hist
        st = search('pair%+d' % sgn, sysid_pair.pair_run, 3.0, 3.6, Pool(4), wrap=lambda b: (sgn, b))
        if st['done']:
            b = st['best']
            json.dump({'sgn': sgn, 'beta': b[0], 'lifetime': b[2], 'outcome': b[1], 'hist': b[3]}, open('runs/pair_%+d%s.json' % (sgn, TAG), 'w'))
            print('beta', b[0])
    elif stage == 'fit':
        import sysid_pair
        P = {s: json.load(open('runs/pair_%+d%s.json' % (s, TAG))) for s in (1, -1)}
        n = min(len(P[1]['hist']['R0']), len(P[-1]['hist']['R0']))
        y = {s: np.array([P[s]['hist'][k][:n] for k in sysid_pair.KEYS]).T for s in (1, -1)}
        odd = 0.5*(y[1] - y[-1])/sysid_pair.A
        even = 0.5*(y[1] + y[-1]); even[:, 0] -= 1.0
        np.savez('runs/pair_response%s.npz' % TAG, t=OUT_DT*np.arange(n), odd=odd, even=even, beta=0.5*(P[1]['beta'] + P[-1]['beta']))
        from control_regulator import fit_power_tail, load_model
        m = load_model(1.0, order=2, power_tail=True)
        print(json.dumps({'dx': DX, 'beta_plus': P[1]['beta'], 'beta_minus': P[-1]['beta'], 'lam': m['lam'], 'k': m['k'],
                          'power_tail': fit_power_tail(1.0)}, indent=1))
    elif stage == 'long':
        from control_regulator import load_model, Regulator
        T, gz, leg = float(sys.argv[2]), 0.05, 12.5
        f = 'runs/long%s_ckpt.pkl' % TAG
        m = load_model(1.0, order=2, power_tail=True)
        if os.path.exists(f):
            c = pickle.load(open(f, 'rb'))
        else:
            x = grid(DX, T + 1.0 + 16.0)
            c = {'x': x, 'U': initial_state(x), 't': 0.0, 'tau': 0.0, 'hist': None, 'launch': [], 'sat': 0, 'outcome': 'alive'}
        reg = Regulator(m, kappa=0.5, gz=gz); reg.launch, reg.sat = c['launch'], c['sat']
        while c['outcome'] == 'alive' and c['t'] < T - 1e-9 and time.time() - T0 < BUDGET:
            res = None if c['hist'] is None else (c['t'], c['tau'], c['hist'])
            r = evolve(c['x'], c['U'], leg, out_dt=OUT_DT, controller=reg, ctrl_dt=0.25, resume=res)
            c.update(t=float(r['lifetime']), tau=float(r['tau']), hist={k: list(v) for k, v in r['hist'].items()},
                     launch=reg.launch, sat=reg.sat, outcome=r['outcome'])
            pickle.dump(c, open(f, 'wb'))
        done = c['outcome'] != 'alive' or c['t'] >= T - 1e-9
        print('long run: t = %.1f, %s  %s' % (c['t'], c['outcome'], 'DONE' if done else '(run again to continue)'))
        if done:
            L = np.array(c['launch'])
            np.savez('runs/reg_long%s.npz' % TAG, **{k: np.array(v) for k, v in c['hist'].items()}, launch_t=L[:, 0], launch_a=L[:, 1],
                     outcome=c['outcome'], lifetime=c['t'], tau_final=c['tau'], dx=DX)
