/* Runs the Einstein-scalar solver and the controller off the main thread.  Prepended with solver.js. */
(function () {
  'use strict';
  const DX = 0.025, L = 90, W_PULSE = 0.3, X_KICK = 6, W_KICK = 1, X_VIEW = 12, X_MASS = 30;
  const OUT = 0.05, CTRL = 0.25, R_GONE = 0.05, R_OPEN = 8, H_BAD = 3e-3;
  let LIVE = null, cfg = { d: 1, controlOn: true, eps: 0, speed: 3, paused: false };
  let sim, ctl, every, cevery, st, kicks = [], lastWall = 0, lastPost = 0;

  function reset() {
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
    const t = st.step * sim.dt;
    if (st.step % cevery === 0) {
      while (kicks.length) { const k = kicks.shift(); sim.inject(k.field, k.amp, X_KICK, W_KICK); st.newPulses.push({ t, kind: k.field === 'normal' ? 'matter' : 'burst', a: k.amp }); }
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
                  trap: hz.trap, anti: hz.anti, status: st.status, end: st.end, behind: st.behind, started: st.t >= cfg.d + 10,
                  hist: st.newHist, pulses: st.newPulses, x, R, Rbg, ell, eF, eG, dF, G };
    st.newHist = []; st.newPulses = [];
    postMessage(msg, [x.buffer, R.buffer, Rbg.buffer, ell.buffer, eF.buffer, eG.buffer, dF.buffer, G.buffer]);
  }

  function tick() {
    const now = performance.now();
    let wall = Math.min(0.25, (now - lastWall) / 1000); lastWall = now;
    if (!cfg.paused && st.status === 'alive') {
      const target = st.t + wall * cfg.speed;
      const t0 = performance.now();
      while (st.t < target && st.status === 'alive' && performance.now() - t0 < 28) stepOnce();
      st.behind = st.t < target - 0.05;
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
      if (newD) reset(); else if (restartCtl) ctl.restart();
    }
    else if (m.type === 'reset') { reset(); }
    else if (m.type === 'kick') { if (st.status === 'alive') kicks.push({ field: m.field, amp: m.amp }); }
  };
})();
