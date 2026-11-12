"""Long closed-loop runs with the identified regulator.  python3 control_long_run.py T gz [gz ...]"""
import sys
from multiprocessing import Pool
import numpy as np
from control_regulator import load_model, Regulator, full_run


def go(args):
    T, gz, kw = args
    m = load_model(1.0, order=2, power_tail=True)
    r = full_run(m, Regulator(m, kappa=0.5, gz=gz), T, **kw)
    tag = 'T%g_gz%g%s' % (T, gz, '_traveller' if kw else '')
    np.savez('runs/reg_%s.npz' % tag, **{k: r[k] for k in ('t', 'R0', 'PR0', 'S0', 'H', 'M', 'launch_t', 'launch_a', 'log')},
             outcome=r['outcome'], lifetime=r['lifetime'], tau=r['tau'], gz=gz)
    return '%s: %s at %.1f (proper time %.1f), max H %.1e' % (tag, r['outcome'], r['lifetime'], r['tau'], np.nanmax(r['H']))


if __name__ == '__main__':
    T = float(sys.argv[1])
    jobs = [(T, float(g), {}) for g in sys.argv[2:]]
    for line in Pool(len(jobs)).imap(go, jobs):
        print(line, flush=True)
