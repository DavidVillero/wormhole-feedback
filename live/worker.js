/* Runs the Einstein-scalar solver and the controller off the main thread.  Prepended with solver.js and
 * solver_full.js.  Two kinds of run: the mirror-symmetric one (half domain), and 'transit', where one traveller
 * crosses from the +x side and both sides are evolved. */
(function () {
  'use strict';
  const DX = 0.025, L = 90, W_PULSE = 0.3, X_KICK = 6, W_KICK = 1, X_VIEW = 12, X_MASS = 30;
  const OUT = 0.05, CTRL = 0.25, R_GONE = 0.05, R_OPEN = 8, H_BAD = 3e-3;
  const L_T = 60, X_ENERGY = 6, X_PROBE = 6, MAX_PROBES = 24;
  // Auto pace: fast-forward while the viewer is only waiting, play at the chosen speed while something is within
  // NEAR of the throat, and move along in between.  Only the playback rate changes, never the physics.
  const NEAR = 3.5, PACE_WAIT = 5, PACE_IDLE = 2.5, SPEED_MAX = 12;
  let pace = 1;
  // a held throat breathes by a few parts in a thousand after a crossing, which makes it marginally trapped for
  // moments at a time; only report a horizon when light is converging much faster than that
  const TRAP_TOL = 2e-2;
  let LIVE = null, cfg = { d: 1, controlOn: true, eps: 0, speed: 1, paused: false, mode: 'sym', travMass: 5e-3, detector: true, auto: true };
  let sim, ctl, every, cevery, st, kicks = [], lastWall = 0, lastPost = 0;

  function resetTransit() {
    const T = LIVE.transit;
    sim = createSimFull({ dx: DX, L: L_T });
    const dM = sim.initialWithTraveller(T.trim, cfg.travMass, T.x0, T.width);
    ctl = createTransitController(T, { tmOn: 10 });
    every = Math.round(OUT / sim.dt); cevery = Math.round(CTRL / sim.dt);
    const jm = sim.jAt(X_MASS);
    st = { step: 0, t: 0, tau: 0, hist: { R0: [], PR0: [], XT: [] }, newHist: [], newPulses: [], newDet: [], spent: 0, status: 'alive', end: null,
           jm, m0: sim.mass(jm), lastTrap: null, behind: false, transit: true, travMass: dM, E0: sim.matterEnergy(X_ENERGY)[1], through: null,
           probes: [], nProbe: 0 };
    kicks = [];
  }

  function stepTransit() {
    const T = LIVE.transit, t = st.step * sim.dt;
    if (st.step % every === 0) { const fl = sim.inflow(T.xs); ctl.read(t, fl); st.newDet.push(fl); }
    if (st.step % cevery === 0 && cfg.controlOn) {
      const r = ctl.decide(t, st.hist, cfg.detector);
      if (r) {
        const i = Math.max(0, Math.min(Math.round((t - T.d) / OUT), st.hist.XT.length - 1));
        sim.injectPair(r.u * (1 + cfg.eps), T.d, W_PULSE, st.hist.XT[i]);     // launched around the throat as last read
        st.spent += Math.abs(r.u); st.newPulses.push({ t, kind: 'control', a: r.u, ff: r.ff });
      }
    }
    const before = sim.throat();
    if (st.step % every === 0) { st.hist.R0.push(before.R); st.hist.PR0.push(before.Rt); st.hist.XT.push(before.x); st.newHist.push(before.R); }
    if (st.through === null && st.travMass > 0 && t >= T.x0 + 20) { const e = sim.matterEnergy(X_ENERGY); st.through = [e[0] / st.E0, e[1] / st.E0]; }
    sim.rk4(); st.step++; st.t = st.step * sim.dt;
    const th = sim.throat();
    for (const p of st.probes) if (p.state === 'fly') {      // test particles ride the spacetime just computed
      sim.probeStep(p, sim.dt);
      if (p.x - th.x < -(X_VIEW + 0.5)) { p.state = 'through'; p.tEnd = st.t; }
      else if (p.x - th.x > X_VIEW + 0.5 && p.px > 0) { p.state = 'back'; p.tEnd = st.t; }
    }
    st.tau += 0.5 * sim.dt * (Math.exp(before.S) + Math.exp(th.S));
    if (!(Math.abs(th.R - before.R) < 0.1)) return finish(st.lastTrap ? 'collapse' : 'breakdown');
    if (!(th.R > R_GONE)) return finish('collapse');
    if (th.R > R_OPEN) return finish('expand');
    if (st.step % (4 * every) === 0) {
      const trap = sim.trapped(th.j, 20, TRAP_TOL);
      if (trap) st.lastTrap = trap;
      const H = sim.constraints(10)[0];
      if (!(H < H_BAD) && (st.lastTrap || th.R < 0.5 || th.R > 2)) return finish(st.lastTrap ? 'collapse' : 'breakdown');
    }
  }

  function frameTransit() {
    const T = LIVE.transit, th = sim.throat(), [H, M] = sim.constraints(10), trap = sim.trapped(th.j, 20, TRAP_TOL);
    const U = sim.U, BG = sim.BG, I0 = sim.I0, jt = th.j, stride = 2, m = Math.floor(Math.round(X_VIEW / DX) / stride) + 1;
    // drift speed of the throat in these coordinates, from the last 2 time units of readings
    const X = st.hist.XT, nb = Math.min(40, X.length - 1), beta = nb > 0 ? (X[X.length - 1] - X[X.length - 1 - nb]) / (nb * OUT) : 0;
    const side = (s) => {
      const o = { x: new Float32Array(m), R: new Float32Array(m), Rbg: new Float32Array(m), ell: new Float32Array(m), eF: new Float32Array(m), eG: new Float32Array(m), dF: new Float32Array(m), G: new Float32Array(m), S: new Float32Array(m) };
      let l = 0;
      for (let i = 0, k = 0; k < m; i++) {
        const j = jt + s * i, b = I0 + s * i;                      // b: the same distance from the centre in the starting state
        if (i > 0) l += 0.5 * DX * (Math.exp(U[1][j]) + Math.exp(U[1][j - s]));
        if (i % stride) continue;
        const dFx = (U[2][j + 1] - BG[2][b + 1] - U[2][j - 1] + BG[2][b - 1]) / (2 * DX);
        const bFx = (BG[2][b + 1] - BG[2][b - 1]) / (2 * DX), pf = U[6][j] + beta * bFx;   // remove the part due to the drift alone
        const Gx = (U[3][j + 1] - U[3][j - 1]) / (2 * DX);
        o.x[k] = i * DX; o.R[k] = U[0][j]; o.Rbg[k] = BG[0][b]; o.ell[k] = l;
        o.eF[k] = dFx * dFx + pf * pf; o.eG[k] = Gx * Gx + U[7][j] * U[7][j];
        o.dF[k] = U[2][j] - BG[2][b]; o.G[k] = U[3][j]; o.S[k] = U[1][j]; k++;
      }
      return o;
    };
    const r = side(1), lft = side(-1), est = ctl.estimate;
    const msg = { type: 'frame', t: st.t, tau: st.tau, R0: th.R, lapse: Math.exp(th.S), H, M, mass: sim.mass(st.jm) - st.m0, spent: st.spent,
                  trap, anti: null, status: st.status, end: st.end, behind: st.behind, started: st.t >= cfg.d + 10, pace: st.pace || 1,
                  hist: st.newHist, pulses: st.newPulses, det: st.newDet, x: r.x, R: r.R, Rbg: r.Rbg, ell: r.ell, eF: r.eF, eG: r.eG, dF: r.dF, G: r.G, S: r.S, left: lft,
                  transit: { mass: st.travMass, arrive: T.x0, xs: T.xs, detector: cfg.detector, estMass: est.dM, estArrive: est.ta, through: st.through, xt: th.x, saturated: ctl.saturated },
                  probes: st.probes.map((p) => ({ id: p.id, s: p.x - sim.x[jt], phi: p.phi, tau: p.tau, age: (p.tEnd || st.t) - p.t0, v: p.v, b: p.b, state: p.state, En: p.En, L: p.L })) };
    st.newHist = []; st.newPulses = []; st.newDet = [];
    postMessage(msg);
  }

  function reset() {
    if (cfg.mode === 'transit') return resetTransit();
    sim = createSim({ dx: DX, L, outflow: true });
    sim.initial([], LIVE.trim);
    const D = LIVE.distances.find((q) => q.d === cfg.d);
    ctl = createController(D, { tmOn: 10 });
    every = Math.round(OUT / sim.dt); cevery = Math.round(CTRL / sim.dt);
    const jm = sim.jAt(X_MASS);
    st = { step: 0, t: 0, tau: 0, hist: { R0: [], PR0: [] }, newHist: [], newPulses: [], spent: 0, status: 'alive', end: null,
           jm, m0: sim.mass(jm), lastTrap: null, behind: false };
    kicks = [];
  }

  function stepOnce() {
    if (st.transit) return stepTransit();
    const t = st.step * sim.dt;
    if (st.step % cevery === 0) {
      while (kicks.length) { const k = kicks.shift(); sim.inject(k.field, k.amp, X_KICK, W_KICK); st.kickT = t; st.newPulses.push({ t, kind: k.field === 'normal' ? 'matter' : 'burst', a: k.amp }); }
      if (cfg.controlOn) {
        const u = ctl.decide(t, st.hist);
        if (u !== null) {
          sim.inject('phantom', u * (1 + cfg.eps), cfg.d, W_PULSE);   // the real response differs from the model by eps
          st.spent += Math.abs(u); st.newPulses.push({ t, kind: 'control', a: u });
        }
      }
    }
    if (st.step % every === 0) {
      const th = sim.throat();
      st.hist.R0.push(th.R); st.hist.PR0.push(th.Rt); st.newHist.push(th.R);
    }
    const before = sim.throat(), s0 = before.S;
    sim.rk4(); st.step++; st.t = st.step * sim.dt;
    const th = sim.throat();
    st.tau += 0.5 * sim.dt * (Math.exp(s0) + Math.exp(th.S));
    // a jump this large in one step (dt = 0.0125) is the solution blowing up at a singularity, not physics
    if (!(Math.abs(th.R - before.R) < 0.1)) return finish(st.lastTrap ? 'collapse' : 'breakdown');
    if (!(th.R > R_GONE)) return finish('collapse');
    if (th.R > R_OPEN) return finish('expand');
    if (st.step % (4 * every) === 0) {
      const hz = sim.horizons(20);
      if (hz.trap) st.lastTrap = hz.trap;
      const H = sim.constraints(10)[0];
      if (!(H < H_BAD)) return finish(st.lastTrap ? 'collapse' : 'breakdown');
    }
  }

  function finish(kind) {
    st.status = kind;
    st.end = { kind, t: st.t, horizon: st.lastTrap ? st.lastTrap.R : null, mass: sim.mass(st.jm) - st.m0, R0: sim.throat().R };
  }

  function frame() {
    if (st.transit) return frameTransit();
    const th = sim.throat(), [H, M] = sim.constraints(10), hz = sim.horizons(20);
    const p = sim.profile(X_VIEW, 1), n = p.x.length, U = sim.U, J0 = sim.J0;
    const stride = 2, m = Math.floor((n - 1) / stride) + 1;
    const x = new Float32Array(m), R = new Float32Array(m), Rbg = new Float32Array(m), ell = new Float32Array(m), eF = new Float32Array(m), eG = new Float32Array(m), dF = new Float32Array(m), G = new Float32Array(m);
    let l = 0;
    for (let i = 0, k = 0; i < n; i++) {
      if (i > 0) l += 0.5 * DX * (Math.exp(p.S[i]) + Math.exp(p.S[i - 1]));
      if (i % stride) continue;
      const j = J0 + i;
      const dFx = (U[2][j + 1] - sim.BG[2][j + 1] - U[2][j - 1] + sim.BG[2][j - 1]) / (2 * DX);
      const Gx = (U[3][j + 1] - U[3][j - 1]) / (2 * DX);
      x[k] = p.x[i]; R[k] = p.R[i]; Rbg[k] = sim.BG[0][j]; ell[k] = l;
      eF[k] = dFx * dFx + U[6][j] * U[6][j]; eG[k] = Gx * Gx + U[7][j] * U[7][j];
      dF[k] = p.dF[i]; G[k] = p.G[i]; k++;
    }
    const msg = { type: 'frame', t: st.t, tau: st.tau, R0: th.R, lapse: Math.exp(th.S), H, M, mass: sim.mass(st.jm) - st.m0, spent: st.spent,
                  trap: hz.trap, anti: hz.anti, status: st.status, end: st.end, behind: st.behind, started: st.t >= cfg.d + 10, pace: st.pace || 1,
                  hist: st.newHist, pulses: st.newPulses, x, R, Rbg, ell, eF, eG, dF, G };
    st.newHist = []; st.newPulses = [];
    postMessage(msg, [x.buffer, R.buffer, Rbg.buffer, ell.buffer, eF.buffer, eG.buffer, dF.buffer, G.buffer]);
  }

  function paceTarget() {
    if (!cfg.auto) return 1;
    const t = st.t;
    let near = false, waiting = false;
    if (st.transit) {
      const T = LIVE.transit, X = st.hist.XT, xt = X.length ? X[X.length - 1] : 0;
      if (st.travMass > 0) { if (Math.abs(T.x0 - t) < NEAR) near = true; else if (t < T.x0) waiting = true; }
      for (const p of st.probes) if (p.state === 'fly') { if (Math.abs(p.x - xt) < NEAR) near = true; else waiting = true; }
    } else {
      if (st.kickT !== undefined) { const s = X_KICK - (t - st.kickT); if (Math.abs(s) < NEAR) near = true; else if (s > 0) waiting = true; }
    }
    if (cfg.controlOn && t < cfg.d + 10 && !near) waiting = true;        // the controller has not started yet
    if (Math.abs(sim.throat().R - 1) > 0.05) near = true;                 // the throat itself is on the move
    return near ? 1 : waiting ? PACE_WAIT : PACE_IDLE;
  }

  function tick() {
    const now = performance.now();
    let wall = Math.min(0.25, (now - lastWall) / 1000); lastWall = now;
    if (!cfg.paused && st.status === 'alive') {
      // 'due' is the simulated time asked for so far.  Stepping only when a whole step is due keeps the rate at
      // the requested speed (stepping at least once per tick ran slow settings about twice too fast).
      if (!(st.due >= st.t)) st.due = st.t;
      pace += (paceTarget() - pace) * Math.min(1, wall * 5);               // ease between rates over a few tenths of a second
      st.pace = pace;
      st.due += wall * Math.min(SPEED_MAX, cfg.speed * pace);
      const t0 = performance.now();
      while (st.t + sim.dt <= st.due && st.status === 'alive' && performance.now() - t0 < 28) stepOnce();
      st.behind = st.t < st.due - 0.05;
      if (st.due - st.t > 0.5) st.due = st.t + 0.5;          // too slow to keep up: do not bank the shortfall
    } else if (kicks.length && st.status === 'alive') {
      stepOnce();
    }
    if (now - lastPost > 30) { lastPost = now; frame(); }
    setTimeout(tick, 4);
  }

  onmessage = (e) => {
    const m = e.data;
    if (m.type === 'init') { LIVE = m.live; Object.assign(cfg, m.cfg || {}); reset(); lastWall = performance.now(); tick(); }
    else if (m.type === 'set') {
      const restartCtl = m.cfg.controlOn === true && !cfg.controlOn;
      const newD = m.cfg.d !== undefined && m.cfg.d !== cfg.d;
      Object.assign(cfg, m.cfg);
      if (newD) { cfg.mode = 'sym'; reset(); } else if (restartCtl && !st.transit) ctl.restart();
    }
    else if (m.type === 'reset') { cfg.mode = 'sym'; reset(); }
    else if (m.type === 'transit') { Object.assign(cfg, m.cfg, { mode: 'transit', d: 1 }); reset(); }
    else if (m.type === 'probe') {
      if (st.transit && st.status === 'alive') {
        const p = sim.probeLaunch(sim.throat().x + X_PROBE, m.v, m.b, Math.PI / 2);
        Object.assign(p, { id: ++st.nProbe, t0: st.t, state: 'fly', b: m.b });
        st.probes.push(p); if (st.probes.length > MAX_PROBES) st.probes.shift();
      }
    }
    else if (m.type === 'kick') { if (st.status === 'alive' && !st.transit) kicks.push({ field: m.field, amp: m.amp }); }
  };
})();
