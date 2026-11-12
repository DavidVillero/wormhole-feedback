"""Constraint residuals immediately before and after one control-pulse insertion (paper, Section 2.3).

Inserts the largest pulse used in the 225 b run (amplitude 1.3e-4, width 0.3 b) from d = 1 and 2 b
on the static wormhole, at dx = 0.025 and 0.0125 b, and prints the change in the Hamiltonian (H)
and momentum (M) residuals on |x| <= 10 b, including near the inner edge (|x| = d - 3w) of the
exterior constraint correction.  Run: python3 check_injection.py
"""
import sys, numpy as np

from control_solver import grid, initial_state, constraints
from control_feedback import inject
for dx in (0.025, 0.0125):
    for d in (1.0, 2.0):
        x = grid(dx, 40.0); U = initial_state(x)
        ana = np.abs(x) <= 10
        H0, M0 = constraints(U, dx)
        inject(x, U, 1.3e-4, d, 0.3)
        H1, M1 = constraints(U, dx)
        dH, dM = np.abs(H1 - H0), np.abs(M1 - M0)
        xi = d - 0.9; near = ana & (np.abs(np.abs(x) - xi) < 0.2)
        print(f"dx={dx} d={d}: rms H before {np.sqrt(np.mean(H0[ana]**2)):.2e} after {np.sqrt(np.mean(H1[ana]**2)):.2e}; "
              f"max|dH| {dH[ana].max():.2e} at |x|={abs(x[ana][dH[ana].argmax()]):.2f}; near interface max|dH| {dH[near].max():.2e}; max|dM| {dM[ana].max():.2e}")
