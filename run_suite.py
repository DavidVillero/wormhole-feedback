"""Run the validation suite (static wormhole + perturbed wormholes at several resolutions)."""
import json
import sys
from multiprocessing import Pool
from wormhole_solver import run

DX = {'c': 0.05, 'm': 0.025, 'f': 0.0125, 'x': 0.00625}
CASES = {'static': 0.0, 'plus': +0.01, 'minus': -0.01, 'plus_small': +0.001, 'minus_small': -0.001,
         'plus_tiny': +0.0001, 'minus_tiny': -0.0001}


def job(args):
    name, amp, lvl = args
    res = run(dx=DX[lvl], amp=amp, T=25.0, L=40.0)
    fn = 'runs/%s_%s.json' % (name, lvl)
    with open(fn, 'w') as fh:
        json.dump(res, fh)
    h = res['history']
    return '%-16s %s  %-55s R0_end=%.4g  maxH=%.2e maxM=%.2e  wall=%.0fs' % (
        name, lvl, res['status'], h['R0'][-1], max(h['H']), max(h['M']), res['wall_seconds'])


if __name__ == '__main__':
    levels = sys.argv[1] if len(sys.argv) > 1 else 'cmf'
    jobs = [(n, a, l) for l in levels for n, a in CASES.items()]
    with Pool(4) as p:
        for line in p.imap(job, jobs):
            print(line, flush=True)
