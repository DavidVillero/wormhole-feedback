"""Distance sweep: calibrate and run the regulator with the controller at |x| = d.

    WH_D=2 python3 sweep.py sysid
    WH_D=2 python3 sweep.py pair +1        (repeat until DONE; then -1)
    WH_D=2 python3 sweep.py fit
    WH_D=2 python3 sweep.py run 100 0.05 [more gz values]   # closed loop from the grid-balanced wormhole

Calibration files carry the distance in their names.  The pulse width is 0.3 b at every distance.
"""
import json
import sys
from multiprocessing import Pool
import numpy as np
import sysid
import sysid_pair
from sysid_pair import D, DTAG
from control_feedback import OUT_DT, TAG
from rescheck import search
import os
ORDER = int(os.environ.get('WH_ORDER', 2))     # estimator order (3 also rejects a steady radius drift)
TM_ON = float(os.environ.get('WH_TM_ON', 0.0))  # readings before this time are ignored
GP = float(os.environ.get('WH_GP', 0.0))        # set-point gain on the measured radius offset
GI = float(os.environ.get('WH_GI', 0.0))        # integral gain of the radius loop
TAU_F = float(os.environ.get('WH_TAUF', 0.0))   # low-pass time constant on the radius reading
TAU_B = float(os.environ.get('WH_TAUB', 0.0))   # bias-rejection time constant (0 = off)
KZ = float(os.environ.get('WH_KZ', 0.0))        # radius-loop rate; if set, gp and gi are derived from the model


def run(args):
    T, gz, kappa = args
    from control_regulator import load_model, Regulator, full_run
    m = load_model(D, order=ORDER, power_tail=True)
    gp, gi = (GP, GI)
    if KZ:
        from control_regulator import radius_loop_gains
        gp, gi = radius_loop_gains(m, KZ)
    r = full_run(m, Regulator(m, kappa=kappa, gz=gz, gp=gp, gi=gi, tau_f=TAU_F, tau_b=TAU_B, tm_on=TM_ON), T, trimmed=True)
    np.savez('runs/sweep_d%g_o%d_on%g_gp%.3g_gi%.3g_tf%g_tb%g_gz%g_k%g%s.npz' % (D, ORDER, TM_ON, gp, gi, TAU_F, TAU_B, gz, kappa, '' if sysid_pair.T0 == 0 else '_calt%g' % sysid_pair.T0), **{k: r[k] for k in ('t', 'R0', 'S0', 'H', 'M', 'launch_t', 'launch_a')},
             outcome=r['outcome'], lifetime=r['lifetime'], k=m['k'], lam=m['lam'], d=D, gz=gz, kappa=kappa)
    t, la, lt = r['t'], r['launch_a'], r['launch_t']
    late = t > 0.5*t[-1]
    return ('d=%g order=%d gp=%.3g gi=%.3g gz=%g kappa=%g: %s at %.1f | late max|dR| %.2e | k*a rms: first 25 b %.2e, last 25 b %.2e | saturated %d' % (
        D, ORDER, gp, gi, gz, kappa, r['outcome'], r['lifetime'], np.abs(r['R0'][late] - 1).max(),
        m['k']*np.sqrt(np.mean(la[lt < lt[0] + 25]**2)), m['k']*np.sqrt(np.mean(la[lt > lt[-1] - 25]**2)), r['saturated']))


def runlong(args):
    """Resumable closed-loop run (checkpoints every leg; call again until DONE)."""
    import pickle, time
    import rescheck
    from control_regulator import load_model, Regulator
    from control_solver import grid, initial_state, evolve
    from control_feedback import DX
    T, gz, kappa = args
    tag = 'd%g_o%d_on%g_gp%g_gi%g_gz%g_k%g%s' % (D, ORDER, TM_ON, GP, GI, gz, kappa, TAG) if (GP or GI) else 'd%g_o%d_on%g_gz%g_k%g%s' % (D, ORDER, TM_ON, gz, kappa, TAG)
    tag += '' if sysid_pair.T0 == 0 else '_calt%g' % sysid_pair.T0
    gp, gi = GP, GI
    if KZ:
        tag = 'v2_d%g_o%d_kz%g_tb%g_k%g%s%s' % (D, ORDER, KZ, TAU_B, kappa, TAG, '' if sysid_pair.T0 == 0 else '_calt%g' % sysid_pair.T0)
    f = 'runs/runlong_%s.pkl' % tag
    m = load_model(D, order=ORDER, power_tail=True)
    if os.path.exists(f):
        c = pickle.load(open(f, 'rb'))
    else:
        b = json.load(open('runs/balance_log%s.json' % TAG))
        x = grid(DX, T + D + 16.0)
        c = {'x': x, 'U': initial_state(x, s_amp=0.5*(b['lo'] + b['hi'])), 't': 0.0, 'tau': 0.0, 'hist': None, 'launch': [], 'sat': 0, 'outcome': 'alive'}
    if KZ:
        from control_regulator import radius_loop_gains
        gp, gi = radius_loop_gains(m, KZ)
    reg = Regulator(m, kappa=kappa, gz=gz, gp=gp, gi=gi, tau_f=TAU_F, tau_b=TAU_B, tm_on=TM_ON); reg.launch, reg.sat = c['launch'], c['sat']
    reg.zf, reg.zint, reg.bias, reg.Gprev = c.get('zf'), c.get('zint', 0.0), c.get('bias', 0.0), c.get('Gprev')
    while c['outcome'] == 'alive' and c['t'] < T - 1e-9 and time.time() - rescheck.T0 < rescheck.BUDGET:
        res = None if c['hist'] is None else (c['t'], c['tau'], c['hist'])
        r = evolve(c['x'], c['U'], 12.5, out_dt=OUT_DT, controller=reg, ctrl_dt=0.25, resume=res)
        c.update(t=float(r['lifetime']), tau=float(r['tau']), hist={k: list(v) for k, v in r['hist'].items()},
                 launch=reg.launch, sat=reg.sat, outcome=r['outcome'], zf=reg.zf, zint=reg.zint, bias=reg.bias, Gprev=reg.Gprev)
        pickle.dump(c, open(f, 'wb'))
    done = c['outcome'] != 'alive' or c['t'] >= T - 1e-9
    if done:
        L = np.array(c['launch'])
        np.savez('runs/sweep_%s.npz' % tag, t=np.array(c['hist']['t']), R0=np.array(c['hist']['R0']), H=np.array(c['hist']['H']),
                 launch_t=L[:, 0], launch_a=L[:, 1], outcome=c['outcome'], lifetime=c['t'],
                 k=(m['kernel']['k'] if m.get('kernel') else m['k']), d=D, kappa=kappa)
    return 'd=%g kappa=%g dx=%g: t = %.1f, %s  %s' % (D, kappa, DX, c['t'], c['outcome'], 'DONE' if done else '(run again to continue)')


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'runlong':
        jobs = []
        for tok in sys.argv[3:]:
            gz, _, kp = tok.partition(':')
            jobs.append((float(sys.argv[2]), float(gz), float(kp) if kp else 0.5))
        for line in Pool(len(jobs)).imap(runlong, jobs):
            print(line, flush=True)
        sys.exit(0)
    if stage == 'sysid':
        h = sysid.record(D, sysid_pair.W)
        print('d=%g single-pulse response recorded DONE' % D)
    elif stage == 'pair':
        sgn = int(sys.argv[2])
        st = search('pair%+d%s' % (sgn, DTAG), sysid_pair.pair_run, 3.0, 3.6, Pool(4), wrap=lambda b: (sgn, b))
        if st['done']:
            b = st['best']
            json.dump({'sgn': sgn, 'beta': b[0], 'lifetime': b[2], 'outcome': b[1], 'hist': b[3]}, open('runs/pair_%+d%s%s.json' % (sgn, DTAG, TAG), 'w'))
    elif stage == 'fit':
        P = {s: json.load(open('runs/pair_%+d%s%s.json' % (s, DTAG, TAG))) for s in (1, -1)}
        n = min(len(P[1]['hist']['R0']), len(P[-1]['hist']['R0']))
        y = {s: np.array([P[s]['hist'][k][:n] for k in sysid_pair.KEYS]).T for s in (1, -1)}
        i0 = int(round(sysid_pair.T0/OUT_DT))                 # time is measured from the first launch
        odd = (0.5*(y[1] - y[-1])/sysid_pair.A)[i0:]; n = len(odd)
        np.savez('runs/pair_response%s%s.npz' % (DTAG, TAG), t=OUT_DT*np.arange(n), odd=odd, beta=0.5*(P[1]['beta'] + P[-1]['beta']))
        from control_regulator import fit_power_tail, load_model
        m = load_model(D, order=2, power_tail=True)
        print(json.dumps({'d': D, 'beta': [P[1]['beta'], P[-1]['beta']], 'lam': m['lam'], 'k': m['k'],
                          'delay_growth': float(np.exp(2*m['lam']*D)), 'power_tail': fit_power_tail(D)}, indent=1))
    elif stage == 'run':
        T = float(sys.argv[2])
        jobs = []
        for tok in sys.argv[3:]:
            gz, _, kp = tok.partition(':')
            jobs.append((T, float(gz), float(kp) if kp else 0.5))
        for line in Pool(min(4, len(jobs))).imap(run, jobs):
            print(line, flush=True)
