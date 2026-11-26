/* Full-domain version of the solver, for runs that are not mirror-symmetric (one traveller crossing
 * from one side).  Same equations and differences as solver.js; both sides are evolved, the constraints
 * are solved outward on each side separately, and the throat is read where the areal radius is smallest.
 * Port of the WH_SIDES=1 WH_TRACK=1 path of control_solver.py / control_feedback.py / transit_ff.py,
 * and of the detector-fed controller in transit_sensed.py.
 */
function createSimFull(opt) {
  'use strict';
  const FOURPI = 4 * Math.PI, TWOPI = 2 * Math.PI;
  const dx = opt.dx, L = opt.L, ko = opt.ko === undefined ? 0.05 : opt.ko, cfl = opt.cfl || 0.5;
  const nL = Math.round(L / dx), N = 2 * nL + 1, I0 = nL, dt = cfl * dx;
  const x = new Float64Array(N);
  for (let j = 0; j < N; j++) x[j] = (j - nL) * dx;
  const mk = () => { const a = []; for (let f = 0; f < 8; f++) a.push(new Float64Array(N)); return a; };
  let U = mk();
  const K1 = mk(), K2 = mk(), K3 = mk(), K4 = mk(), T = mk(), BG = mk();

  function rhs(V, D) {
    const R = V[0], S = V[1], F = V[2], G = V[3], PR = V[4], PS = V[5], PF = V[6], PG = V[7];
    const i2 = 1 / (2 * dx), ix2 = 1 / (dx * dx), kof = ko / (16 * dx);
    for (let j = 2; j < N - 2; j++) {
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
      for (let j = 2; j < N - 2; j++) d[j] -= kof * (u[j + 2] - 4 * u[j + 1] + 6 * u[j] - 4 * u[j - 1] + u[j - 2]);
    }
    // outgoing waves leave the grid at both ends
    for (let f = 0; f < 4; f++) {
      const P = V[f + 4];
      for (const j of [N - 2, N - 1]) { D[f][j] = P[j]; D[f + 4][j] = -(3 * P[j] - 4 * P[j - 1] + P[j - 2]) / (2 * dx); }
      for (const j of [0, 1]) { D[f][j] = P[j]; D[f + 4][j] = (-3 * P[j] + 4 * P[j + 1] - P[j + 2]) / (2 * dx); }
    }
  }
  function axpy(out, a, h, b) { for (let f = 0; f < 8; f++) { const o = out[f], p = a[f], q = b[f]; for (let j = 0; j < N; j++) o[j] = p[j] + h * q[j]; } }
  function rk4() {
    rhs(U, K1); axpy(T, U, 0.5 * dt, K1);
    rhs(T, K2); axpy(T, U, 0.5 * dt, K2);
    rhs(T, K3); axpy(T, U, dt, K3);
    rhs(T, K4);
    const h6 = dt / 6;
    for (let f = 0; f < 8; f++) { const u = U[f], a = K1[f], b = K2[f], c = K3[f], d = K4[f]; for (let j = 0; j < N; j++) u[j] += h6 * (a[j] + 2 * b[j] + 2 * c[j] + d[j]); }
  }

  // ---------- constraint solving ----------
  function d1_4(u) {
    const out = new Float64Array(N), i2 = 1 / (2 * dx), i12 = 1 / (12 * dx);
    for (let j = 1; j < N - 1; j++) out[j] = (u[j + 1] - u[j - 1]) * i2;
    for (let j = 2; j < N - 2; j++) out[j] = (-u[j + 2] + 8 * u[j + 1] - 8 * u[j - 1] + u[j - 2]) * i12;
    return out;
  }
  function mid(f) {
    const m = new Float64Array(N - 1);
    for (let k = 0; k < N - 1; k++) m[k] = 0.5 * (f[k] + f[k + 1]);
    for (let k = 1; k < N - 2; k++) m[k] = (-f[k - 1] + 9 * f[k] + 9 * f[k + 1] - f[k + 2]) / 16;
    return m;
  }
  const rev = (V) => V.map((a) => Float64Array.from(a).reverse());
  // H = 0, M = 0 from index js to the last index of V (V may be the reversed state: the equations keep their form)
  function integrateOn(V, js, y0) {
    const S = V[1], F = V[2], G = V[3], PS = V[5], PF = V[6], PG = V[7];
    const Fx = d1_4(F), Gx = d1_4(G), sx = d1_4(S);
    const e = new Float64Array(N), E = new Float64Array(N), Jc = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      e[j] = Math.exp(2 * S[j]);
      E[j] = -(PF[j] * PF[j] + Fx[j] * Fx[j]) + (PG[j] * PG[j] + Gx[j] * Gx[j]);
      Jc[j] = -PF[j] * Fx[j] + PG[j] * Gx[j];
    }
    const em = mid(e), sxm = mid(sx), psm = mid(PS), Em = mid(E), Jm = mid(Jc), ps = PS;
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
  function solveFromCentre() {       // both sides outward from x = 0, with R = 1, R_x = 0, R_t = 0 there
    const [Rr, Pr] = integrateOn(U, I0, [1, 0, 0]);
    const [Rl, Pl] = integrateOn(rev(U), I0, [1, 0, 0]);
    for (let i = 0; i < Rr.length; i++) { U[0][I0 + i] = Rr[i]; U[4][I0 + i] = Pr[i]; U[0][I0 - i] = Rl[i]; U[4][I0 - i] = Pl[i]; }
  }
  function initial(sAmp) {
    U = mk(); for (let f = 0; f < 8; f++) { K1[f].fill(0); K2[f].fill(0); K3[f].fill(0); K4[f].fill(0); }
    for (let j = 0; j < N; j++) { const xx = x[j]; U[0][j] = Math.sqrt(xx * xx + 1); U[1][j] = sAmp * Math.exp(-xx * xx); U[2][j] = Math.atan(xx) / Math.sqrt(FOURPI); }
    solveFromCentre();
    for (let f = 0; f < 8; f++) BG[f].set(U[f]);
  }
  const jAt = (xx) => I0 + Math.round(xx / dx);
  function mass(j) { const R = U[0], Rx = (R[j + 1] - R[j - 1]) / (2 * dx), p = U[4][j]; return 0.5 * R[j] * (1 + Math.exp(-2 * U[1][j]) * (p * p - Rx * Rx)); }
  function addTraveller(amp, x0, w) {     // one ingoing shell of ordinary matter, on the +x side only
    for (let j = 0; j < N; j++) { const g = amp * Math.exp(-((x[j] - x0) / w) * ((x[j] - x0) / w)); U[3][j] += g; U[7][j] += -2 * (x[j] - x0) / (w * w) * g; }
    solveFromCentre();
  }
  // balanced wormhole plus one traveller of (very nearly) the requested Misner-Sharp mass; returns the mass
  function initialWithTraveller(sAmp, dM, x0, w) {
    initial(sAmp);
    if (!(dM > 0)) return 0;
    const jm = jAt(x0 + 8), m0 = mass(jm), a = 1e-4;
    addTraveller(a, x0, w);
    const amp = a * Math.sqrt(dM / (mass(jm) - m0));
    initial(sAmp); addTraveller(amp, x0, w);
    return mass(jm) - m0;
  }
  // a pair of ingoing phantom pulses at centre +- d, with the constraints re-solved outside each
  function injectPair(amp, d, w, centre) {
    const ir = Math.round((centre + d - 3 * w - x[0]) / dx);
    const jl = N - 1 - Math.round((centre - (d - 3 * w) - x[0]) / dx);
    let Vr = rev(U);
    const y0r = [U[0][ir], d1_4(U[0])[ir], U[4][ir]], y0l = [Vr[0][jl], d1_4(Vr[0])[jl], Vr[4][jl]];
    const [Ra, Pa] = integrateOn(U, ir, y0r), [La, Qa] = integrateOn(Vr, jl, y0l);
    for (let j = 0; j < N; j++) {
      const xr = x[j] - centre, ax = Math.abs(xr), sg = xr > 0 ? 1 : xr < 0 ? -1 : 0;
      const g = amp * Math.exp(-((ax - d) / w) * ((ax - d) / w)), gx = -2 * (ax - d) / (w * w) * g * sg;
      U[2][j] += sg * g; U[6][j] += gx;
    }
    Vr = rev(U);
    const [Rb, Pb] = integrateOn(U, ir, y0r), [Lb, Qb] = integrateOn(Vr, jl, y0l);
    for (let i = 0; i < Ra.length; i++) { U[0][ir + i] += Rb[i] - Ra[i]; U[4][ir + i] += Pb[i] - Pa[i]; }
    for (let i = 0; i < La.length; i++) { U[0][N - 1 - jl - i] += Lb[i] - La[i]; U[4][N - 1 - jl - i] += Qb[i] - Qa[i]; }
  }

  // ---------- diagnostics ----------
  function throat(half) {            // minimum of the areal radius near x = 0, parabolic refinement
    const k = Math.round((half || 3) / dx), R = U[0], P = U[4];
    let j = I0 - k;
    for (let i = I0 - k; i <= I0 + k; i++) if (R[i] < R[j]) j = i;
    if (j === I0 - k || j === I0 + k) return { x: x[j], R: R[j], Rt: P[j], S: U[1][j], j };
    const a = R[j - 1], b = R[j], c = R[j + 1], den = a - 2 * b + c, off = den !== 0 ? 0.5 * (a - c) / den : 0;
    const pt = off >= 0 ? P[j] + off * (P[j + 1] - P[j]) : P[j] + off * (P[j] - P[j - 1]);
    return { x: x[j] + off * dx, R: b - 0.25 * (a - c) * off, Rt: pt, S: U[1][j], j };
  }
  function constraints(xa) {
    const R = U[0], S = U[1], F = U[2], G = U[3], PR = U[4], PS = U[5], PF = U[6], PG = U[7];
    const i2 = 1 / (2 * dx), ix2 = 1 / (dx * dx);
    let sh = 0, sm = 0, cnt = 0;
    for (let j = jAt(-xa); j <= jAt(xa); j++) {
      const Rx = (R[j + 1] - R[j - 1]) * i2, Sx = (S[j + 1] - S[j - 1]) * i2, Fx = (F[j + 1] - F[j - 1]) * i2, Gx = (G[j + 1] - G[j - 1]) * i2;
      const Q = (Math.exp(2 * S[j]) + PR[j] * PR[j] - Rx * Rx) / R[j];
      const H = (R[j + 1] - 2 * R[j] + R[j - 1]) * ix2 - 0.5 * Q - (PS[j] * PR[j] + Sx * Rx) + TWOPI * R[j] * (-(PF[j] * PF[j] + Fx * Fx) + (PG[j] * PG[j] + Gx * Gx));
      const M = (PR[j + 1] - PR[j - 1]) * i2 - (PS[j] * Rx + Sx * PR[j]) + FOURPI * R[j] * (-PF[j] * Fx + PG[j] * Gx);
      sh += H * H; sm += M * M; cnt++;
    }
    return [Math.sqrt(sh / cnt), Math.sqrt(sm / cnt)];
  }
  // ingoing energy flux of the ordinary-matter field through the spheres at |x| = xs
  function inflow(xs) {
    const R = U[0], G = U[3], PG = U[7], jp = jAt(xs), jm = jAt(-xs);
    const gp = (G[jp + 1] - G[jp - 1]) / (2 * dx), gm = (G[jm + 1] - G[jm - 1]) / (2 * dx);
    return FOURPI * (R[jp] * R[jp] * PG[jp] * gp - R[jm] * R[jm] * PG[jm] * gm);
  }
  // energy of the ordinary-matter field on each side, |x| > xmin (flat-space weighting, far from the throat)
  function matterEnergy(xmin) {
    const R = U[0], G = U[3], PG = U[7];
    let eL = 0, eR = 0;
    for (let j = 1; j < N - 1; j++) {
      if (Math.abs(x[j]) <= xmin) continue;
      const gx = (G[j + 1] - G[j - 1]) / (2 * dx), e = TWOPI * R[j] * R[j] * (PG[j] * PG[j] + gx * gx) * dx;
      if (x[j] < 0) eL += e; else eR += e;
    }
    return [eL, eR];
  }
  function trapped(jt, xmax, tol) {  // outgoing light converging on either side of the throat: black-hole interior
    const R = U[0], PR = U[4], S = U[1]; tol = tol === undefined ? 1e-3 : tol;
    let trap = null;
    for (let i = 0; i * dx <= xmax; i++) for (const s of [1, -1]) {
      const j = jt + s * i; if (j < 2 || j > N - 3) continue;
      const Rx = (R[j + 1] - R[j - 1]) / (2 * dx), k = Math.exp(-S[j]) / R[j];
      if ((PR[j] + s * Rx) * k < -tol) trap = { x: i * dx, R: R[j] };
    }
    return trap;
  }

  // ---------- test particles ----------
  // A probe of unit mass on a timelike geodesic in the equatorial plane; it does not bend the spacetime.
  // State: x, phi, px (covariant radial momentum), tau (its own clock); L = R^2 dphi/dtau is conserved.
  // With En = sqrt(px^2 + e^{2S}(1 + L^2/R^2)) (from the unit norm of the 4-velocity), in coordinate time:
  //   dx/dt = px/En,  dphi/dt = e^{2S} L/(R^2 En),  dtau/dt = e^{2S}/En,
  //   dpx/dt = e^{2S}(-S_x (1 + L^2/R^2) + L^2 R_x/R^3)/En
  function probeRate(xp, px, L) {
    const q = (xp - x[0]) / dx, j = Math.max(1, Math.min(N - 3, Math.floor(q))), w = q - j, R = U[0], S = U[1];
    const r = R[j] + w * (R[j + 1] - R[j]), s = S[j] + w * (S[j + 1] - S[j]);
    const rx = ((1 - w) * (R[j + 1] - R[j - 1]) + w * (R[j + 2] - R[j])) / (2 * dx), sx = ((1 - w) * (S[j + 1] - S[j - 1]) + w * (S[j + 2] - S[j])) / (2 * dx);
    const e2 = Math.exp(2 * s), c = 1 + L * L / (r * r), En = Math.sqrt(px * px + e2 * c);
    return [px / En, e2 * L / (r * r * En), e2 * (-sx * c + L * L * rx / (r * r * r)) / En, e2 / En, En];
  }
  function probeStep(p, h) {          // RK4 with the fields held fixed over the step
    const k1 = probeRate(p.x, p.px, p.L), k2 = probeRate(p.x + 0.5 * h * k1[0], p.px + 0.5 * h * k1[2], p.L);
    const k3 = probeRate(p.x + 0.5 * h * k2[0], p.px + 0.5 * h * k2[2], p.L), k4 = probeRate(p.x + h * k3[0], p.px + h * k3[2], p.L);
    for (const [key, i] of [['x', 0], ['phi', 1], ['px', 2], ['tau', 3]]) p[key] += h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    p.v = k4[0]; p.En = k4[4];
  }
  // launched inward from x0 at speed v (fraction of light speed, as measured there), aimed to miss the centre by b
  function probeLaunch(x0, v, b, phi0) {
    const j = jAt(x0), r = U[0][j], s = U[1][j], pm = v / Math.sqrt(1 - v * v), L = b * pm;
    const pr = -Math.sqrt(Math.max(0, pm * pm - L * L / (r * r)));
    return { x: x0, phi: phi0 || 0, px: Math.exp(s) * pr, L, tau: 0, v: 0, En: 0 };
  }

  return { dx, dt, x, N, I0, get U() { return U; }, BG, initial, initialWithTraveller, rk4, injectPair, throat, constraints,
           mass, jAt, inflow, matterEnergy, trapped, probeStep, probeLaunch };
}

/* Kalman filter + regulator with a detector at the mouth (transit_sensed.py, SensedLQG).
 * M: { ctrl, lam, sig: { s0, ds, ng: [2][], kern: [2][] }, cpts: [[m, c], [m, c]], kappa, delta, xs, sLo, sHi }
 * read(t, flux) every 0.05; decide(t, hist, useDetector) every 0.25 -> { u, fb, ff } or null.
 */
function createTransitController(M, opts) {
  'use strict';
  const C = M.ctrl, n = C.n, d = M.d;
  const A = []; for (let i = 0; i < n; i++) A.push(new Float64Array(n));
  C.A.forEach(([i, j, v]) => { A[i][j] = v; });
  const B = Float64Array.from(C.B), Cm = C.C.map((r) => Float64Array.from(r)), K = Float64Array.from(C.K);
  const Mk = C.Mk.map((r) => Float64Array.from(r));
  const tmOn = opts && opts.tmOn !== undefined ? opts.tmOn : 10, aMax = (opts && opts.aMax) || 0.1;
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const sig = M.sig, ns = sig.ng[0].length;
  const interp = (arr, s) => { const q = (s - sig.s0) / sig.ds; if (q <= 0) return arr[0]; if (q >= ns - 1) return arr[ns - 1]; const i = Math.floor(q); return arr[i] + (q - i) * (arr[i + 1] - arr[i]); };
  const cOf = (dM) => M.cpts[0][1] + (M.cpts[1][1] - M.cpts[0][1]) * (dM - M.cpts[0][0]) / (M.cpts[1][0] - M.cpts[0][0]);
  let xbar, started, stored, anchor, ffLaunch, est, frozen, used, dt_, df_, sat;
  function restart() { xbar = new Float64Array(n); started = false; stored = []; anchor = null; ffLaunch = []; est = { dM: 0, ta: null }; frozen = false; used = 0; dt_ = []; df_ = []; sat = 0; }
  restart();

  function passage(tUpto) {          // energy passed inward, its flux-weighted mean time, and the per-sample energies
    let m = 0; while (m < dt_.length && dt_[m] <= tUpto + 1e-9) m++;
    if (m < 2) return { E: 0, tc: null, m: 0, fw: [] };
    const fw = new Float64Array(m); let E = 0, tE = 0;
    for (let i = 0; i < m; i++) {
      const w = i === 0 ? 0.5 * (dt_[1] - dt_[0]) : i === m - 1 ? 0.5 * (dt_[m - 1] - dt_[m - 2]) : 0.5 * (dt_[i + 1] - dt_[i - 1]);
      fw[i] = Math.max(0, df_[i]) * w; E += fw[i]; tE += dt_[i] * fw[i];
    }
    return { E, tc: E > 0 ? tE / E : null, m, fw };
  }
  function known(t) {                // readings the traveller and the streamed pulses are known to cause
    let y0 = 0, y1 = 0;
    if (est.ta !== null && est.dM > 0) {
      const sr = t - d - est.ta;
      if (sr >= M.sLo) { y0 += est.dM * interp(sig.ng[0], sr); if (sr <= M.sHi) y1 += est.dM * interp(sig.ng[1], sr); }
    }
    for (let k = 0; k < ffLaunch.length; k++) {
      const sk = t - d - (ffLaunch[k][0] + d);
      if (sk >= M.sLo) { y0 += ffLaunch[k][1] * interp(sig.kern[0], sk); if (sk <= M.sHi) y1 += ffLaunch[k][1] * interp(sig.kern[1], sk); }
    }
    return [y0, y1];
  }
  function update(xb, y0, y1) {
    const e0 = y0 - dot(Cm[0], xb), e1 = y1 - dot(Cm[1], xb), xh = new Float64Array(n);
    for (let k = 0; k < n; k++) xh[k] = xb[k] + Mk[k][0] * e0 + Mk[k][1] * e1;
    return xh;
  }
  function predict(xh, u) { const xn = new Float64Array(n); for (let k = 0; k < n; k++) xn[k] = dot(A[k], xh) + B[k] * u; return xn; }

  return {
    d, restart,
    get estimate() { return est; },
    get saturated() { return sat; },
    read(t, flux) { dt_.push(t); df_.push(flux); },
    decide(t, hist, useDetector) {
      const tm = t - d;
      if (tm < tmOn) return null;
      const i = Math.min(Math.round(tm / 0.05), hist.R0.length - 1);
      if (i < 0) return null;
      const yr0 = hist.R0[i] - 1, yr1 = hist.PR0[i];
      if (!started) { xbar[C.iZ] = yr0; xbar[C.iV] = yr1; started = true; }
      let uff = 0;
      if (useDetector) {
        const p = passage(t - (M.xs - d));               // detector data that can have reached the controller
        if (p.E > 0) {
          const dM = M.kappa * p.E;
          est = { dM, ta: p.tc + M.xs + M.delta };
          if (anchor === null) anchor = { j: stored.length, x: Float64Array.from(xbar) };
          let s = 0, any = false;
          for (let q = 0; q < p.m; q++) if (dt_[q] > used + 1e-9) {
            const late = (t + d) - (dt_[q] + M.xs + M.delta);
            s += p.fw[q] * Math.exp(M.lam * Math.max(0, late)); any = true;
          }
          if (any) { uff = cOf(dM) * M.kappa * s; used = dt_[p.m - 1]; }
        }
      }
      if (anchor !== null && !frozen) {                  // re-run the filter from the anchor with the current estimates
        let xb = Float64Array.from(anchor.x);
        for (let q = anchor.j; q < stored.length; q++) { const r = stored[q], kn = known(r.t); xb = predict(update(xb, r.y0 - kn[0], r.y1 - kn[1]), r.u); }
        xbar = xb;
        if (est.ta !== null && t - d - est.ta > M.sHi + 1) frozen = true;
      }
      const kn = known(t), xh = update(xbar, yr0 - kn[0], yr1 - kn[1]);
      let ufb = -dot(K, xh);
      if (Math.abs(ufb) > aMax) { ufb = Math.sign(ufb) * aMax; sat++; }
      xbar = predict(xh, ufb);
      stored.push({ t, y0: yr0, y1: yr1, u: ufb });
      if (uff) ffLaunch.push([t, uff]);
      return { u: ufb + uff, fb: ufb, ff: uff };
    },
  };
}

if (typeof module !== 'undefined') module.exports = { createSimFull, createTransitController };
