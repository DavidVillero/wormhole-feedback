"""Symbolic check of the field equations used by wormhole_solver.py.

Metric (conformal gauge, spherical symmetry):
    ds^2 = e^{2 sigma(t,x)} (-dt^2 + dx^2) + R(t,x)^2 dOmega^2
Matter: massless scalar, T_ab = eps [ d_a phi d_b phi - (1/2) g_ab (d phi)^2 ],
        eps = +1 canonical, eps = -1 phantom (ghost).  Units G = c = 1.

We compute E_ab = G_ab - 8 pi T_ab directly from the metric and show that
every component is a combination of the residuals the solver uses:
    eR : R evolution equation        eS : sigma evolution equation
    H  : Hamiltonian-type constraint  M  : momentum-type constraint
and that the scalar wave equation matches Box phi = 0.
"""
import sympy as sp

t, x, th, ph = sp.symbols('t x theta varphi')
eps = sp.Symbol('epsilon')
R = sp.Function('R')(t, x)
s = sp.Function('sigma')(t, x)
f = sp.Function('phi')(t, x)
X = [t, x, th, ph]
g = sp.diag(-sp.exp(2*s), sp.exp(2*s), R**2, R**2*sp.sin(th)**2)
gi = g.inv()

def christoffel(a, b, c):
    return sum(gi[a, d]*(sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d]))
               for d in range(4))/2
G3 = [[[sp.simplify(christoffel(a, b, c)) for c in range(4)] for b in range(4)] for a in range(4)]

def ricci(b, c):
    r = 0
    for a in range(4):
        r += sp.diff(G3[a][b][c], X[a]) - sp.diff(G3[a][b][a], X[c])
        for d in range(4):
            r += G3[a][a][d]*G3[d][b][c] - G3[a][c][d]*G3[d][b][a]
    return sp.simplify(r)
Ric = sp.Matrix(4, 4, lambda b, c: ricci(b, c))
Rs = sp.simplify(sum(gi[a, a]*Ric[a, a] for a in range(4)))
Gt = (Ric - g*Rs/2).applyfunc(sp.simplify)

df = [sp.diff(f, X[a]) for a in range(4)]
dphi2 = sum(gi[a, a]*df[a]**2 for a in range(4))
T = sp.Matrix(4, 4, lambda a, b: eps*(df[a]*df[b] - g[a, b]*dphi2/2))
E = (Gt - 8*sp.pi*T).applyfunc(sp.simplify)

Rt, Rx, st, sx, ft, fx = [sp.diff(u, v) for u in (R, s, f) for v in (t, x)]
Q = (sp.exp(2*s) + Rt**2 - Rx**2)/R
eR = sp.diff(R, t, 2) - sp.diff(R, x, 2) + Q
eS = sp.diff(s, t, 2) - sp.diff(s, x, 2) - Q/R + 4*sp.pi*eps*(ft**2 - fx**2)
H = (sp.diff(R, t, 2) + sp.diff(R, x, 2))/2 - (st*Rt + sx*Rx) + 2*sp.pi*eps*R*(ft**2 + fx**2)
M = sp.diff(R, t, x) - (st*Rx + sx*Rt) + 4*sp.pi*eps*R*ft*fx
eF = sp.diff(f, t, 2) - sp.diff(f, x, 2) + (2/R)*(Rt*ft - Rx*fx)

checks = {
    'E_tt + E_xx + (4/R) H        ': E[0, 0] + E[1, 1] + 4*H/R,
    'E_tx + (2/R) M               ': E[0, 1] + 2*M/R,
    'E_tt - E_xx - (2/R) eR       ': E[0, 0] - E[1, 1] - 2*eR/R,
    'E_thth + R^2 e^{-2s}(eS + eR/R)': E[2, 2] + R**2*sp.exp(-2*s)*(eS + eR/R),
    'E_phph - sin^2 E_thth        ': E[3, 3] - sp.sin(th)**2*E[2, 2],
}
box = sum(sp.diff(sp.sqrt(-g.det().subs(sp.sin(th)**2, 1))*gi[a, a]*df[a], X[a]) for a in range(2)) \
      / sp.sqrt(-g.det().subs(sp.sin(th)**2, 1))
checks['Box(phi) + e^{-2s} eF        '] = box + sp.exp(-2*s)*eF
ok = True
for k, v in checks.items():
    r = sp.simplify(v)
    print(k, '=', r)
    ok &= (r == 0)

# Ellis-Bronnikov wormhole is an exact static solution for eps = -1
b = sp.Symbol('b', positive=True)
ell = {R: sp.sqrt(x**2 + b**2), s: 0, f: sp.atan(x/b)/sp.sqrt(4*sp.pi)}
for name, expr in (('eR', eR), ('eS', eS), ('H', H), ('M', M), ('eF', eF)):
    r = sp.simplify(expr.subs(eps, -1).subs(ell).doit())
    print('Ellis-Bronnikov residual', name, '=', r)
    ok &= (r == 0)
print('ALL CHECKS PASS' if ok else 'CHECK FAILED')
