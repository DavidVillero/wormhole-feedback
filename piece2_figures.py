"""Piece 2 headline figure: the heaviest traveller each controller lets through.

Reads the transit result files, derives for each controller the heaviest mass held (alive at t = 90 b,
with every lighter mass also held) and the lightest mass lost, writes results_piece2_limits.json and
article/fig_piece2_limits.png.  Coarse grid (dx = 0.025); the fine-grid check (dx = 0.0125) gives the same brackets for rows 1, 2, 5 and 6.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE = '#257de0', '#f47230'          # categorical slots 1-2 (dataviz reference palette)
INK, INK2, GRID = '#191919', '#666666', '#e6e6e6'


def rows(*files, mode=None):
    out = []
    for f in files:
        if os.path.exists(f):
            out += [r for r in json.load(open(f))['rows'] if mode is None or r['mode'] == mode]
    return out


def bracket(rs):
    rs = sorted(rs, key=lambda r: r['traveller_mass'])
    held, lost = None, None
    for r in rs:
        if r['traveller_mass'] <= 0:
            continue
        if r['outcome'] == 'alive' and lost is None:
            held = r['traveller_mass']
        elif r['outcome'] != 'alive' and lost is None:
            lost = r['traveller_mass']
    return held, lost


g = json.load(open('results_transit_gate.json'))['limit_held_lost_b']
items = [
    ('Two travellers, one from each side (Stage 2)', BLUE, [
        ('Plain regulator', tuple(g['plain_lqg']['dx0.025'])),
        ('Told in advance: feedforward + signature', tuple(g['feedforward_plus_signature']['dx0.025']))]),
    ('One traveller, crossing to the other side', ORANGE, [
        ('Plain regulator, reading at x = 0', bracket(rows('results_transit_1side_scan.json', mode='plain'))),
        ('Plain regulator, tracking the throat', bracket(rows('results_transit_1side_scan_track.json', mode='plain'))),
        ('Told in advance, tracking the throat', bracket(rows('results_transit_1side_scan_track.json', 'results_transit_1side_scan_track_refine.json', mode='ff+sig'))),
        ('Detector at the mouth, no advance notice', bracket(rows('results_transit_sensed_scan_1side_track.json', 'results_transit_sensed_scan_1side_track_refine.json', mode='sensed')))]),
]
json.dump({grp: {lab: {'held': h, 'lost': l} for lab, (h, l) in ent} for grp, _, ent in items},
          open('results_piece2_limits.json', 'w'), indent=1)

fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)
y, ticks, labels = 0, [], []
for grp, col, ent in items:
    for lab, (h, l) in ent:
        ax.plot([h, l], [y, y], color=col, lw=2, solid_capstyle='round', zorder=2)
        ax.scatter([h], [y], s=70, color=col, edgecolor='white', linewidth=2, zorder=3)
        ax.scatter([l], [y], s=70, facecolor='white', edgecolor=col, linewidth=2, zorder=3)
        ax.text(l*1.12, y, 'held %.3g, lost %.3g' % (h, l), va='center', fontsize=8, color=INK2)
        ticks.append(y); labels.append(lab); y -= 1
    y -= 0.6
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9, color=INK)
ax.set_xscale('log'); ax.set_xlim(1.2e-3, 1.5e-1)
ax.set_xticks([2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]); ax.set_xticklabels(['0.002', '0.005', '0.01', '0.02', '0.05', '0.1']); ax.minorticks_off()
ax.set_xlabel('Traveller mass (Misner–Sharp, in throat radii)', fontsize=9, color=INK2)
ax.grid(axis='x', color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#bbbbbb'); ax.tick_params(axis='x', colors=INK2, labelsize=8); ax.tick_params(axis='y', length=0)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=BLUE, lw=2, marker='o', label='Two travellers (symmetric)'),
                   Line2D([], [], color=ORANGE, lw=2, marker='o', label='One traveller (crosses through)'),
                   Line2D([], [], ls='', marker='o', mfc='white', mec=INK2, label='open dot: lost')],
          loc='upper center', bbox_to_anchor=(0.35, -0.16), ncol=3, fontsize=8, frameon=False)
ax.set_title('Heaviest traveller the controlled wormhole lets through\n(filled: held to t = 90 b; open: lost; controller at d = 1 b; dx = 0.025,\nrows 1, 2, 5 and 6 also confirmed at dx = 0.0125)',
             fontsize=10, color=INK, loc='left')
fig.tight_layout()
fig.savefig('article/fig_piece2_limits.png')
for grp, _, ent in items:
    for lab, (h, l) in ent:
        print('%-48s held %-7.3g lost %.3g' % (lab, h, l))
