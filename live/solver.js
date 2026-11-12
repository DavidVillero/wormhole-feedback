/* Einstein + two massless scalar fields, spherical symmetry, conformal gauge (G = c = 1).
 *   ds^2 = e^{2S}(-dt^2 + dx^2) + R^2 dOmega^2
 *   F: phantom field (negative kinetic energy, holds the wormhole open), odd in x
 *   G: normal field, even in x
 * Port of control_solver.py / control_feedback.py (gravity-step).  Same equations, same
 * finite differences (2nd order), RK4, Kreiss-Oliger dissipation, same constraint-solving
 * pulse injection.  Only x >= 0 is evolved; x < 0 follows by reflection (exact for these
 * symmetric scenarios), using two ghost points.  Optional sponge near the outer edge lets
 * runs continue past the light-crossing time of the grid.
 */
function createSim(opt) {
  'use strict';
  const FOURPI = 4 * Math.PI, TWOPI = 2 * Math.PI;
  const h0 = opt.dx, L = opt.L, ko = opt.ko === undefined ? 0.05 : opt.ko, cfl = opt.cfl || 0.5;
  const nL = Math.round(L / h0);
  // spacing used in differences: numpy's grid gives x[1] - x[0], which differs from h0 in the last
  // digits; using the same value makes the port reproduce the Python runs to round-off
  const dx = opt.pyDx ? h0 * (1 - nL) - h0 * (-nL) : h0;
  const dt = cfl * dx;
  const nh = nL + 1;                          // points with x >= 0
  const N = nh + 2, J0 = 2;                   // two ghost points on the x < 0 side
  const x = new Float64Array(N);
  for (let j = 0; j < N; j++) x[j] = (j - J0) * h0;
  const PAR = [1, 1, -1, 1, 1, 1, -1, 1];     // R S F G PR PS PF PG
  const mk = () => { const a = []; for (let f = 0; f < 8; f++) a.push(new Float64Array(N)); return a; };
  let U = mk();
  const K1 = mk(), K2 = mk(), K3 = mk(), K4 = mk(), T = mk(), BG = mk();
  const sponge = new Float64Array(N);
  const xs = opt.sponge ? opt.sponge.start : Infinity;
  if (opt.sponge) for (let j = 0; j < N; j++) if (x[j] > xs) { const q = (x[j] - xs) / (L - xs); sponge[j] = opt.sponge.strength * q * q; }

  function ghosts(V) { for (let f = 0; f < 8; f++) { const a = V[f], p = PAR[f]; a[1] = p * a[3]; a[0] = p * a[4]; } }

  // ---------- right-hand side ----------
  function rhs(V, D) {
    ghosts(V);
    const R = V[0], S = V[1], F = V[2], G = V[3], PR = V[4], PS = V[5], PF = V[6], PG = V[7];
    const i2 = 1 / (2 * dx), ix2 = 1 / (dx * dx), kof = ko / (16 * dx);
    for (let j = J0; j < N - 2; j++) {
      const Rx = (R[j + 1] - R[j - 1]) * i2, Fx = (F[j + 1] - F[j - 1]) * i2, Gx = (G[j + 1] - G[j - 1]) * i2;
      const r = R[j], pr = PR[j], pf = PF[j], pg = PG[j];
      const Q = (Math.exp(2 * S[j]) + pr * pr - Rx * Rx) / r;
      D[0][j] = pr; D[1][j] = PS[j]; D[2][j] = pf; D[3][j] = pg;
      D[4][j] = (R[j + 1] - 2 * r + R[j - 1]) * ix2 - Q;
      D[5][j] = (S[j + 1] - 2 * S[j] + S[j - 1]) * ix2 + Q / r - FOURPI * (-(pf * pf - Fx * Fx) + (pg * pg - Gx * Gx));
      D[6][j] = (F[j + 1] - 2 * F[j] + F[j - 1]) * ix2 - (2 / r) * (pr * pf - Rx * Fx);
      D[7][j] = (G[j + 1] - 2 * G[j] + G[j - 1]) * ix2 - (2 / r) * (pr * pg - Rx * Gx);
    }
    if (ko > 0) for (let f = 0; f < 8; f++) {
      const u = V[f], d = D[f];
      for (let j = J0; j < N - 2; j++) d[j] -= kof * (u[j + 2] - 4 * u[j + 1] + 6 * u[j] - 4 * u[j - 1] + u[j - 2]);
    }
    if (opt.sponge) for (let f = 0; f < 8; f++) {
      const u = V[f], d = D[f], b = BG[f];
      for (let j = J0; j < N - 2; j++) if (sponge[j] > 0) d[j] -= sponge[j] * (u[j] - b[j]);
    }
    if (opt.outflow) {
      // outgoing waves leave the grid: the background is static, so each P obeys P_t + P_x = 0 at the edge
      for (const j of [N - 2, N - 1]) for (let f = 0; f < 4; f++) {
        const P = V[f + 4];
        D[f][j] = P[j];
        D[f + 4][j] = -(3 * P[j] - 4 * P[j - 1] + P[j - 2]) / (2 * dx);
      }
    } else {
      for (let f = 0; f < 8; f++) { D[f][N - 2] = 0; D[f][N - 1] = 0; }
    }
  }
  function axpy(out, a, h, b) { for (let f = 0; f < 8; f++) { const o = out[f], p = a[f], q = b[f]; for (let j = J0; j < N; j++) o[j] = p[j] + h * q[j]; } }
  function rk4() {
    rhs(U, K1); axpy(T, U, 0.5 * dt, K1);
    rhs(T, K2); axpy(T, U, 0.5 * dt, K2);
    rhs(T, K3); axpy(T, U, dt, K3);
    rhs(T, K4);
    const h6 = dt / 6;
    for (let f = 0; f < 8; f++) { const u = U[f], a = K1[f], b = K2[f], c = K3[f], d = K4[f]; for (let j = J0; j < N; j++) u[j] += h6 * (a[j] + 2 * b[j] + 2 * c[j] + d[j]); }
  }

  // ---------- constraint solving (pulse injection and initial data) ----------
  function d1_4(u) {
    const out = new Float64Array(N), i2 = 1 / (2 * dx), i12 = 1 / (12 * dx);
    for (let j = 1; j < N - 1; j++) out[j] = (u[j + 1] - u[j - 1]) * i2;
    for (let j = 2; j < N - 2; j++) out[j] = (-u[j + 2] + 8 * u[j + 1] - 8 * u[j - 1] + u[j - 2]) * i12;
    return out;
  }
  function mid(f) {            // value halfway between j and j+1, 4th order where possible
    const m = new Float64Array(N - 1);
    for (let k = 0; k < N - 1; k++) m[k] = 0.5 * (f[k] + f[k + 1]);
    for (let k = 1; k < N - 2; k++) m[k] = (-f[k - 1] + 9 * f[k] + 9 * f[k + 1] - f[k + 2]) / 16;
    return m;
  }
  function coeffs() {
    ghosts(U);
    const S = U[1], F = U[2], G = U[3], PS = U[5], PF = U[6], PG = U[7];
    const Fx = d1_4(F), Gx = d1_4(G), Sx = d1_4(S);
    const e = new Float64Array(N), E = new Float64Array(N), Jc = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      e[j] = Math.exp(2 * S[j]);
      E[j] = -(PF[j] * PF[j] + Fx[j] * Fx[j]) + (PG[j] * PG[j] + Gx[j] * Gx[j]);
      Jc[j] = -PF[j] * Fx[j] + PG[j] * Gx[j];
    }
    const a = [e, Sx, PS, E, Jc];
    // fill the mirrored side by parity so the midpoint interpolation at x = 0 matches the full-domain one
    [1, -1, 1, 1, -1].forEach((p, k) => { a[k][1] = p * a[k][3]; a[k][0] = p * a[k][4]; });
    return { a, m: a.map((f) => mid(f)) };
  }
  function integrate(js, y0) {   // H = 0, M = 0 outward from index js; returns R and R_t from js to the edge
    const { a, m } = coeffs();
    const [e, sx, ps, E, Jc] = a, [em, sxm, psm, Em, Jm] = m;
    let r = y0[0], rx = y0[1], p = y0[2];
    const Ro = new Float64Array(N - js), Po = new Float64Array(N - js);
    Ro[0] = r; Po[0] = p;
    const h = dx;
    const fb = (r_, x_, p_, ee, s_, q_, EE) => (ee + p_ * p_ - x_ * x_) / (2 * r_) + q_ * p_ + s_ * x_ - TWOPI * r_ * EE;
    const fc = (r_, x_, p_, s_, q_, JJ) => q_ * x_ + s_ * p_ - FOURPI * r_ * JJ;
    for (let i = js; i < N - 1; i++) {
      const k = i, kk = i + 1;
      const a1 = rx, b1 = fb(r, rx, p, e[k], sx[k], ps[k], E[k]), c1 = fc(r, rx, p, sx[k], ps[k], Jc[k]);
      const r2 = r + 0.5 * h * a1, x2 = rx + 0.5 * h * b1, p2 = p + 0.5 * h * c1;
      const a2 = x2, b2 = fb(r2, x2, p2, em[k], sxm[k], psm[k], Em[k]), c2 = fc(r2, x2, p2, sxm[k], psm[k], Jm[k]);
      const r3 = r + 0.5 * h * a2, x3 = rx + 0.5 * h * b2, p3 = p + 0.5 * h * c2;
      const a3 = x3, b3 = fb(r3, x3, p3, em[k], sxm[k], psm[k], Em[k]), c3 = fc(r3, x3, p3, sxm[k], psm[k], Jm[k]);
      const r4 = r + h * a3, x4 = rx + h * b3, p4 = p + h * c3;
      const a4 = x4, b4 = fb(r4, x4, p4, e[kk], sx[kk], ps[kk], E[kk]), c4 = fc(r4, x4, p4, sx[kk], ps[kk], Jc[kk]);
      r += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4); rx += h / 6 * (b1 + 2 * b2 + 2 * b3 + b4); p += h / 6 * (c1 + 2 * c2 + 2 * c3 + c4);
      Ro[i + 1 - js] = r; Po[i + 1 - js] = p;
    }
    return [Ro, Po];
  }
  function addPulse(field, amp, x0, w) {
    for (let j = J0; j < N; j++) {
      const ax = x[j], sg = ax > 0 ? 1 : 0;
      const g = amp * Math.exp(-((ax - x0) / w) * ((ax - x0) / w));
      const gx = -2 * (ax - x0) / (w * w) * g * sg;
      if (field === 'phantom') { U[2][j] += sg * g; U[6][j] += gx; }
      else { U[3][j] += g; U[7][j] += sg * gx; }
    }
  }
  function inject(field, amp, x0, w) {
    const js = J0 + Math.round((x0 - 3 * w) / h0);
    ghosts(U);
    const y0 = [U[0][js], d1_4(U[0])[js], U[4][js]];
    const [Ra, Pa] = integrate(js, y0);
    addPulse(field, amp, x0, w);
    const [Rb, Pb] = integrate(js, y0);
    for (let i = 0; i < Ra.length; i++) { U[0][js + i] += Rb[i] - Ra[i]; U[4][js + i] += Pb[i] - Pa[i]; }
    ghosts(U);
  }
  function initial(pulses, sAmp) {
    U = mk(); for (let f = 0; f < 8; f++) { K1[f].fill(0); K2[f].fill(0); K3[f].fill(0); K4[f].fill(0); }
    for (let j = 0; j < N; j++) { const xx = x[j]; U[0][j] = Math.sqrt(xx * xx + 1); U[1][j] = sAmp * Math.exp(-xx * xx); U[2][j] = Math.atan(xx) / Math.sqrt(FOURPI); }
    (pulses || []).forEach((p) => addPulse(p[0], p[1], p[2], p[3]));
    const [Ro, Po] = integrate(J0, [1, 0, 0]);
    for (let i = 0; i < Ro.length; i++) { U[0][J0 + i] = Ro[i]; U[4][J0 + i] = Po[i]; }
    ghosts(U);
    for (let f = 0; f < 8; f++) BG[f].set(U[f]);
  }

  // ---------- diagnostics ----------
  function constraints(xa) {   // rms of H and M over |x| <= xa, counted over the full (mirrored) domain
    ghosts(U);
    const R = U[0], S = U[1], F = U[2], G = U[3], PR = U[4], PS = U[5], PF = U[6], PG = U[7];
    const i2 = 1 / (2 * dx), ix2 = 1 / (dx * dx);
    let sh = 0, sm = 0, cnt = 0;
    for (let j = J0; j < N - 1 && x[j] <= xa + 1e-12; j++) {
      const Rx = (R[j + 1] - R[j - 1]) * i2, Sx = (S[j + 1] - S[j - 1]) * i2, Fx = (F[j + 1] - F[j - 1]) * i2, Gx = (G[j + 1] - G[j - 1]) * i2;
      const Q = (Math.exp(2 * S[j]) + PR[j] * PR[j] - Rx * Rx) / R[j];
      const H = (R[j + 1] - 2 * R[j] + R[j - 1]) * ix2 - 0.5 * Q - (PS[j] * PR[j] + Sx * Rx) + TWOPI * R[j] * (-(PF[j] * PF[j] + Fx * Fx) + (PG[j] * PG[j] + Gx * Gx));
      const M = (PR[j + 1] - PR[j - 1]) * i2 - (PS[j] * Rx + Sx * PR[j]) + FOURPI * R[j] * (-PF[j] * Fx + PG[j] * Gx);
      const wgt = j === J0 ? 1 : 2;
      sh += wgt * H * H; sm += wgt * M * M; cnt += wgt;
    }
    return [Math.sqrt(sh / cnt), Math.sqrt(sm / cnt)];
  }
  function mass(j) { const R = U[0], Rx = (R[j + 1] - R[j - 1]) / (2 * dx), p = U[4][j]; return 0.5 * R[j] * (1 + Math.exp(-2 * U[1][j]) * (p * p - Rx * Rx)); }
  function horizons(xmax, tol) {
    // Expansions of outgoing / ingoing light (x > 0 side): (R_t +- R_x) e^{-S} / R.
    // Trapped (black-hole interior): outgoing < 0.  Anti-trapped (runaway expansion): ingoing > 0.
    // tol keeps round-off at the static throat, where both are ~0, from being reported.
    ghosts(U);
    const R = U[0], PR = U[4], S = U[1];
    tol = tol === undefined ? 1e-3 : tol;
    let trap = null, anti = null;
    for (let j = J0; j < N - 2 && x[j] <= xmax; j++) {
      const Rx = (R[j + 1] - R[j - 1]) / (2 * dx), k = Math.exp(-S[j]) / R[j];
      if ((PR[j] + Rx) * k < -tol) trap = { x: x[j], R: R[j] };
      if ((PR[j] - Rx) * k > tol) anti = { x: x[j], R: R[j] };
    }
    return { trap, anti };
  }
  function profile(xmax, stride) {
    ghosts(U);
    const out = { x: [], R: [], S: [], dF: [], G: [], Rt: [] };
    for (let j = J0; j < N && x[j] <= xmax + 1e-9; j += stride) {
      out.x.push(x[j]); out.R.push(U[0][j]); out.S.push(U[1][j]); out.dF.push(U[2][j] - BG[2][j]); out.G.push(U[3][j]); out.Rt.push(U[4][j]);
    }
    return out;
  }

  return {
    dx, dt, x, N, J0, get U() { return U; }, BG,
    initial, rk4, inject, constraints, mass, horizons, profile,
    throat: () => ({ R: U[0][J0], Rt: U[4][J0], S: U[1][J0], St: U[5][J0] }),
    jAt: (xx) => J0 + Math.round(xx / dx),
  };
}

/* The observer-based controller (Kalman filter + LQR) tested in control_lqg.py, from viz_models.json. */
function createController(D, opts) {
  'use strict';
  const C = D.ctrl, n = C.n;
  const A = []; for (let i = 0; i < n; i++) A.push(new Float64Array(n));
  C.A.forEach(([i, j, v]) => { A[i][j] = v; });
  const B = Float64Array.from(C.B), Cm = C.C.map((r) => Float64Array.from(r)), K = Float64Array.from(C.K);
  const Mk = C.Mk.map((r) => Float64Array.from(r));
  const tmOn = opts && opts.tmOn !== undefined ? opts.tmOn : 10, aMax = 0.02;
  let xbar = new Float64Array(n), started = false;
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  return {
    d: D.d,
    restart() { started = false; xbar = new Float64Array(n); },
    // hist: arrays R0, PR0 sampled every 0.05; returns pulse amplitude or null
    decide(t, hist) {
      const tm = t - D.d;
      if (tm < tmOn) return null;
      const i = Math.min(Math.round(tm / 0.05), hist.R0.length - 1);
      if (i < 0) return null;
      const y0 = hist.R0[i] - 1, y1 = hist.PR0[i];
      if (!started) { xbar[C.iZ] = y0; xbar[C.iV] = y1; started = true; }
      const e0 = y0 - dot(Cm[0], xbar), e1 = y1 - dot(Cm[1], xbar);
      const xh = new Float64Array(n);
      for (let k = 0; k < n; k++) xh[k] = xbar[k] + Mk[k][0] * e0 + Mk[k][1] * e1;
      let u = -dot(K, xh);
      if (u > aMax) u = aMax; else if (u < -aMax) u = -aMax;
      const xn = new Float64Array(n);
      for (let k = 0; k < n; k++) xn[k] = dot(A[k], xh) + B[k] * u;
      xbar = xn;
      return u;
    },
  };
}

if (typeof module !== 'undefined') module.exports = { createSim, createController };
