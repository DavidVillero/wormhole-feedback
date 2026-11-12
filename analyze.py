"""Analyse the validation suite: convergence, growth rate, horizons. Writes results.json + validation.png."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LIT_TIMESCALE = 0.846   # Gonzalez, Guzman & Sarbach, arXiv:0806.0608, Table I, zero-mass wormhole
DX = {'c': 0.05, 'm': 0.025, 'f': 0.0125, 'x': 0.00625}


def load(name):
    d = json.load(open('runs/%s.json' % name))
    return d, {k: np.array(v) for k, v in d['history'].items()}


def at(h, key, t):
    return float(np.interp(t, h['t'], h[key]))

out = {'literature_timescale_over_throat_radius': LIT_TIMESCALE, 'literature_growth_rate': 1/LIT_TIMESCALE}

# 1. convergence -------------------------------------------------------------
conv = {}
for case, tt in (('static', 4.0), ('plus_small', 4.0), ('minus_small', 4.0), ('minus', 10.0)):
    R = {l: at(load('%s_%s' % (case, l))[1], 'R0', tt) for l in 'cmfx'}
    conv[case] = {'t': tt, 'R0': R,
                  'ratio_c-m_over_m-f': (R['c'] - R['m'])/(R['m'] - R['f']),
                  'ratio_m-f_over_f-x': (R['m'] - R['f'])/(R['f'] - R['x'])}
out['self_convergence_throat_radius'] = conv
out['static_drift_R0_minus_1_at_t4'] = {l: at(load('static_%s' % l)[1], 'R0', 4.0) - 1 for l in 'cmfx'}
cons = {}
for case in ('plus_small', 'minus_small'):
    H = {l: at(load('%s_%s' % (case, l))[1], 'H', 3.0) for l in 'cmfx'}
    M = {l: at(load('%s_%s' % (case, l))[1], 'M', 3.0) for l in 'cmfx'}
    cons[case] = {'H_rms_t3': H, 'M_rms_t3': M,
                  'H_ratios': [H['c']/H['m'], H['m']/H['f'], H['f']/H['x']],
                  'M_ratios': [M['c']/M['m'], M['m']/M['f'], M['f']/M['x']]}
out['constraint_convergence'] = cons

# 2. growth rate: odd part of (+amp, -amp) pair cancels the truncation-seeded mode
growth = {}
for pair, win in (('tiny', (3.0, 6.0)), ('small', (2.5, 4.5))):
    row = {}
    for l in 'mfx' if pair == 'small' else 'fx':
        _, hp = load('plus_%s_%s' % (pair, l))
        _, hm = load('minus_%s_%s' % (pair, l))
        n = min(len(hp['t']), len(hm['t']))
        t, odd = hp['t'][:n], 0.5*(hm['R0'][:n] - hp['R0'][:n])
        tau = 0.5*(hp['tau'][:n] + hm['tau'][:n])
        m = (t >= win[0]) & (t <= win[1])
        row[l] = float(np.polyfit(tau[m], np.log(odd[m]), 1)[0])
    growth[pair] = {'fit_window_t': win, 'rate_by_resolution': row}
best = growth['tiny']['rate_by_resolution']['x']
growth['best_rate'] = best
growth['best_timescale'] = 1/best
growth['relative_difference_from_literature'] = (1/best - LIT_TIMESCALE)/LIT_TIMESCALE
out['unstable_mode'] = growth

# 3. outcomes ------------------------------------------------------------------
outc = {}
for case in ('plus_tiny', 'plus_small', 'plus', 'static', 'minus_tiny', 'minus_small', 'minus'):
    d, h = load(case + '_x')
    row = {'amp': d['params']['amp'], 'status': d['status'], 'final_throat_radius': float(h['R0'][-1])}
    ok = np.isfinite(h['R_trap'])
    if ok.any() and h['R0'][-1] < 1:
        row['outcome'] = 'collapse: trapped surfaces form, areal radius -> 0'
        row['apparent_horizon_radius_last'] = float(h['R_trap'][ok][-1])
        row['apparent_horizon_radius_by_resolution'] = {
            l: float(load('%s_%s' % (case, l))[1]['R_trap'][np.isfinite(load('%s_%s' % (case, l))[1]['R_trap'])][-1]) for l in 'fx'}
    else:
        row['outcome'] = 'expansion: throat radius grows without bound in the run'
        row['dR0_dt_final'] = float(h['PR0'][-1])
    outc[case] = row
out['outcomes'] = outc
_, hl = load('minus_long')
out['long_expansion_run'] = {'t_final': float(hl['t'][-1]), 'throat_radius_final': float(hl['R0'][-1]),
                             'throat_proper_time_final': float(hl['tau'][-1]),
                             'max_H_rms': float(hl['H'].max()), 'max_M_rms': float(hl['M'].max())}
json.dump(out, open('results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

# 4. figure ------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
a = ax[0, 0]
for case, lab, col in (('plus', 'amp = +0.01', '#b2182b'), ('plus_small', '+0.001', '#ef8a62'), ('plus_tiny', '+0.0001', '#fdbb84'),
                       ('static', '0 (truncation error only)', '#555555'),
                       ('minus_tiny', '-0.0001', '#92c5de'), ('minus_small', '-0.001', '#4393c3'), ('minus', '-0.01', '#2166ac')):
    _, h = load(case + '_x')
    a.plot(h['tau'], h['R0'], color=col, label=lab)
a.set_ylim(0, 3); a.set_xlim(0, 10)
a.set_xlabel('proper time at throat / b'); a.set_ylabel('throat areal radius / b')
a.set_title('Every run leaves the static wormhole'); a.legend(fontsize=8, title='perturbation')

a = ax[0, 1]
_, hp = load('plus_tiny_x'); _, hm = load('minus_tiny_x')
n = min(len(hp['t']), len(hm['t']))
tau = 0.5*(hp['tau'][:n] + hm['tau'][:n]); odd = 0.5*(hm['R0'][:n] - hp['R0'][:n])
a.semilogy(tau[1:], odd[1:], color='#2166ac', label='simulation: half-difference of the ±0.0001 runs')
i = np.argmin(abs(tau - 4.5))
a.semilogy(tau, odd[i]*np.exp((tau - tau[i])/LIT_TIMESCALE), 'k--', lw=1, label='published rate 1/0.846')
a.set_xlabel('proper time at throat / b'); a.set_ylabel('|throat radius − b| / b')
a.set_title('Growth rate: measured %.3f vs published %.3f' % (best, 1/LIT_TIMESCALE)); a.legend(fontsize=8)

a = ax[1, 0]
for l, col in zip('cmfx', ('#c6dbef', '#6baed6', '#2171b5', '#08306b')):
    _, h = load('minus_small_' + l)
    a.semilogy(h['t'], h['H'], color=col, label='dx = %g' % DX[l])
    a.semilogy(h['t'], h['M'], color=col, ls=':')
a.set_xlabel('t / b'); a.set_ylabel('rms constraint residual, |x| < 10b')
a.set_title('Constraints (solid H, dotted M) fall 4x per halving'); a.legend(fontsize=8)

a = ax[1, 1]
for case, col in (('plus_small_x', '#b2182b'), ('minus_small_x', '#2166ac')):
    d, _ = load(case)
    s = d['snapshots']; xs = np.array(s['x'])
    for j, tt in enumerate(s['t']):
        if tt in (0.0, 3.0, 5.0, 7.0, 9.0):
            a.plot(xs, s['R'][j], color=col, alpha=0.25 + 0.75*tt/9.0, lw=1.2)
            k = np.argmin(abs(xs - 4.0))
a.plot([], [], color='#b2182b', label='amp = +0.001: collapse (t = 0, 3, 5)')
a.plot([], [], color='#2166ac', label='amp = -0.001: expansion (t = 0 ... 9)')
a.set_xlim(-6, 6); a.set_ylim(0, 8); a.set_xlabel('x / b'); a.set_ylabel('areal radius R(t, x) / b')
a.set_title('Areal radius profiles'); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig('validation.png', dpi=150)
