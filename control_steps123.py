"""Control study, steps 1-3: open-loop baseline, grid-balanced wormhole, unstable mode, actuator gain.

    python3 control_steps123.py open      # step 2: tune one phantom pulse against a normal pulse
    python3 control_steps123.py balance   # trim the static wormhole so truncation error does not tip it
    python3 control_steps123.py gain      # step 3: mode shape, estimator, actuator gain vs distance
    python3 control_steps123.py report    # results_control.json + control_baseline.png
"""
import json
import sys
from multiprocessing import Pool
import numpy as np
from control_solver import grid, initial_state, evolve

LAM = 1.185          # growth rate measured in the validation suite
DX = 0.025
NORMAL = ('normal', 0.02, 6.0, 1.0)      # the disturbance: a normal-matter pulse ("traveller")


def mass(U, i):
    R, S, PR = U[0], U[1], U[4]
    Rx = (R[i+1] - R[i-1])/(2*DX)
    return 0.5*R[i]*(1.0 + np.exp(-2*S[i])*(PR[i]**2 - Rx**2))


def open_loop(a):
    x = grid(DX, 80.0)
    r = evolve(x, initial_state(x, [NORMAL, ('phantom', a, 7.0, 1.0)]), 65.0)
    return a, r['outcome'], r['lifetime'], r['hist']['t'].tolist(), r['hist']['R0'].tolist(), float(r['hist']['H'].max())


def trimmed(s):
    x = grid(DX, 80.0)
    r = evolve(x, initial_state(x, s_amp=s), 65.0)
    return s, r['outcome'], r['lifetime'], r['hist']['t'].tolist(), r['hist']['R0'].tolist(), float(r['hist']['H'].max())


def section_search(fn, lo, hi, pool, iters=24):
    """lo collapses, hi expands; 5-way section until the bracket is at round-off."""
    log = [fn(lo), fn(hi)]
    assert log[0][1] == 'collapse' and log[1][1] == 'expand', (log[0][1], log[1][1])
    for _ in range(iters):
        out = pool.map(fn, [lo + (hi - lo)*k/5.0 for k in (1, 2, 3, 4)])
        log += out
        for a, oc, *_ in out:              # points are ordered from the collapse end to the expand end
            if oc == 'expand':
                hi = a
                break
            lo = a
        if any(o[1] == 'alive' for o in out) or abs(hi - lo) < 4e-16*max(abs(hi), abs(lo)):
            break
    return lo, hi, log


def pulse_pair(args):
    d, a, s_amp = args
    x = grid(DX, d + 22.0)
    out = []
    for sgn in (+1, -1):
        U = initial_state(x, [('phantom', sgn*a, d, 1.0)], s_amp=s_amp)
        m = mass(U, len(x) - 6)
        r = evolve(x, U, d + 9.0, snap_times=(d + 3.0, d + 5.0))
        out.append(({'t': r['hist']['t'], 'R0': r['hist']['R0'], 'PR0': r['hist']['PR0'],
                     'snaps': r['snaps'], 'x': r['x'], 'outcome': r['outcome']}, m))
    return d, a, out


if __name__ == '__main__':
    stage = sys.argv[1]
    pool = Pool(4)
    if stage == 'open':
        lo, hi, log = section_search(open_loop, 0.0, 0.03, pool)
        json.dump({'lo': lo, 'hi': hi, 'log': log}, open('runs/open_loop_log.json', 'w'))
        print('critical amplitude', 0.5*(lo + hi), 'runs', len(log), 'longest', max(l[2] for l in log))
    elif stage == 'balance':
        lo, hi, log = section_search(trimmed, 0.0, -1e-3, pool)
        json.dump({'lo': lo, 'hi': hi, 'log': log}, open('runs/balance_log.json', 'w'))
        print('trim', 0.5*(lo + hi), 'runs', len(log), 'longest', max(l[2] for l in log))
    elif stage == 'gain':
        b = json.load(open('runs/balance_log.json'))
        s_amp = 0.5*(b['lo'] + b['hi'])
        gains, modeinfo = {}, {}
        for d, a, out in pool.map(pulse_pair, [(3.0, 1e-7, s_amp), (5.0, 1e-7, s_amp), (8.0, 1e-7, s_amp), (12.0, 1e-7, s_amp)]):
            (rp, mp), (rm, mm) = out
            n = min(len(rp['t']), len(rm['t']))
            t = rp['t'][:n]
            odd = 0.5*(rp['R0'][:n] - rm['R0'][:n])
            dodd = 0.5*(rp['PR0'][:n] - rm['PR0'][:n])
            est = 0.5*(odd + dodd/LAM)                   # growing-mode estimator from two throat readings
            w = (t >= d + 3.0) & (t <= d + 6.0)
            rate, c = np.polyfit(t[w], np.log(np.abs(odd[w])), 1)

            def settle(sig):
                loc = np.gradient(np.log(np.abs(sig) + 1e-300), t)
                bad = np.nonzero((np.abs(loc/LAM - 1) > 0.02) & (t < d + 6.0))[0]
                return float(t[min(bad[-1] + 1, n - 1)] - d) if len(bad) else float('nan')
            gains[str(d)] = {'pulse_amplitude': a, 'outcomes': [rp['outcome'], rm['outcome']], 'fitted_rate': float(rate),
                             'gain_at_arrival_dR0_per_unit_amp': float(np.sign(odd[w][-1])*np.exp(c + rate*d)/a),
                             'pulse_mass_per_unit_amp': float((mp - mm)/(2*a)),
                             'estimator_settles_after_arrival': settle(est),
                             'raw_radius_settles_after_arrival': settle(odd)}
            if d == 5.0:
                xs = rp['x']; i0 = len(xs)//2
                norm = {ts: 0.5*(rp['snaps'][ts] - rm['snaps'][ts]) for ts in rp['snaps']}
                norm = {ts: v/v[0][i0] for ts, v in norm.items()}
                ts = sorted(norm); near = np.abs(xs) < 3
                modeinfo = {'compared_times': ts, **{'max_diff_' + nm: float(np.abs(norm[ts[0]][j] - norm[ts[1]][j])[near].max())
                                                     for j, nm in ((0, 'R'), (1, 'sigma'), (2, 'phi'))}}
                np.savez('unstable_mode.npz', x=xs, early=norm[ts[0]], late=norm[ts[1]],
                         fields=np.array(['R', 'sigma', 'phi', 'psi', 'R_t', 'sigma_t', 'phi_t', 'psi_t']))
        json.dump({'s_amp': s_amp, 'gains': gains, 'mode': modeinfo}, open('runs/gain_log.json', 'w'), indent=1)
        print(json.dumps({'gains': gains, 'mode': modeinfo}, indent=1))
    elif stage == 'report':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        res = {'growth_rate_used': LAM, 'dx': DX, 'disturbance_pulse': NORMAL}
        fits = {}
        for key, fn in (('open_loop', 'open_loop_log'), ('balance', 'balance_log')):
            L = json.load(open('runs/%s.json' % fn))
            log = L['log']; a_star = 0.5*(L['lo'] + L['hi'])
            da = np.array([abs(l[0] - a_star) for l in log]); life = np.array([l[2] for l in log])
            scale = abs(a_star)
            ok = (da > 1e-11*scale) & (da < 1e-3*scale)
            slope, icpt = np.polyfit(np.log(da[ok]), life[ok], 1)
            res[key] = {'critical_value': a_star, 'bracket_width': abs(L['hi'] - L['lo']), 'runs': len(log),
                        'lifetime_untuned': log[0][2], 'longest_lifetime': float(life.max()),
                        'lifetime_per_efold_of_tuning': float(-slope), 'expected_1_over_growth_rate': 1/LAM,
                        'lifetime_per_decade_of_tuning': float(-slope*np.log(10)),
                        'max_H_rms_longest_run': max(log, key=lambda l: l[2])[5]}
            fits[key] = (log, da, life, slope, icpt)
        g = json.load(open('runs/gain_log.json'))
        res['mode_shape_consistency'] = g['mode']; res['actuator_gain_vs_distance'] = g['gains']
        json.dump(res, open('results_control.json', 'w'), indent=1)
        print(json.dumps(res, indent=1))

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        log, da, life, slope, icpt = fits['open_loop']
        a0 = ax[0]
        a0.plot(log[0][3], log[0][4], color='k', lw=1.8, label='normal pulse only')
        picks = sorted(log[1:], key=lambda l: l[2])
        for l in [picks[int(i)] for i in np.linspace(0, len(picks) - 1, 9)]:
            a0.plot(l[3], l[4], color='#b2182b' if l[1] == 'collapse' else '#2166ac', lw=1, alpha=0.85)
        a0.plot([], [], color='#b2182b', label='+ phantom pulse, slightly too weak')
        a0.plot([], [], color='#2166ac', label='+ phantom pulse, slightly too strong')
        a0.set_ylim(0.45, 2.05); a0.set_xlabel('t / b'); a0.set_ylabel('throat areal radius / b')
        a0.set_title('Open loop: a tuned pulse only postpones the end'); a0.legend(fontsize=8, loc='upper left')
        a1 = ax[1]
        for key, col, lab in (('open_loop', '#2166ac', 'pulse amplitude (traveller case)'), ('balance', '#b2182b', 'trim of the static wormhole')):
            log, da, life, slope, icpt = fits[key]
            rel = da/abs(0.5*(json.load(open('runs/%s_log.json' % key))['lo'] + json.load(open('runs/%s_log.json' % key))['hi']))
            a1.semilogx(rel[rel > 0], life[rel > 0], 'o', ms=3, color=col, label='%s: %.2f b per decade' % (lab, -slope*np.log(10)))
        a1.set_xlabel('relative tuning error'); a1.set_ylabel('lifetime / b')
        a1.set_title('Lifetime grows only as log(1/error)'); a1.legend(fontsize=8)
        a2 = ax[2]
        m = np.load('unstable_mode.npz')
        for j, (lab, col) in enumerate((('δR  (areal radius)', '#2166ac'), ('δσ  (conformal factor)', '#b2182b'), ('δφ  (phantom field)', '#1b7837'))):
            a2.plot(m['x'], m['late'][j], color=col, label=lab)
            a2.plot(m['x'], m['early'][j], color=col, ls=':', lw=1.2)
        a2.set_xlim(-8, 8); a2.set_xlabel('x / b'); a2.set_ylabel('perturbation / δR at throat')
        a2.set_title('Unstable mode (dotted: same mode 2 b earlier)'); a2.legend(fontsize=8)
        fig.tight_layout(); fig.savefig('control_baseline.png', dpi=150)
