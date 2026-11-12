const { createSim, createController } = require('./solver.js');
const ref = require('../runs/viz_reference.json');
const VM = require('../viz_models.json');
const byD = {}; VM.distances.forEach((D) => { byD[D.d] = D; });

function run(dx, L, T, opts) {
  const sim = createSim({ dx, L, sponge: opts.sponge, outflow: opts.outflow, pyDx: opts.pyDx !== false });
  sim.initial(opts.pulses || [], opts.trim || 0);
  const every = Math.round(0.05 / sim.dt), cevery = Math.round(0.25 / sim.dt);
  const ctl = opts.d ? createController(byD[opts.d]) : null;
  const hist = { t: [], R0: [], PR0: [], S0: [], H: [] }, launch = [];
  const nsteps = Math.round(T / sim.dt);
  let outcome = 'alive', step = 0;
  const lo = opts.lo || 0.5, hi = opts.hi || 2.0;
  while (true) {
    const t = step * sim.dt;
    if (step % cevery === 0) {
      if (opts.disturb && Math.abs(t - opts.disturb[0]) < 1e-9) sim.inject(...opts.disturb.slice(1));
      if (ctl) { const u = ctl.decide(t, hist); if (u !== null) { launch.push([t, u]); sim.inject('phantom', u, opts.d, 0.3); } }
    }
    if (step % every === 0) {
      const th = sim.throat(); hist.t.push(t); hist.R0.push(th.R); hist.PR0.push(th.Rt); hist.S0.push(th.S);
      if (opts.constraints) hist.H.push(sim.constraints(10)[0]);
    }
    if (step >= nsteps) break;
    sim.rk4(); step++;
    const r0 = sim.throat().R;
    if (!(r0 >= lo)) { outcome = 'collapse'; break; }
    if (r0 > hi) { outcome = 'expand'; break; }
  }
  return { outcome, lifetime: step * sim.dt, hist, launch };
}

function compare(name, js, py) {
  const n = Math.min(js.hist.R0.length, py.R0.length);
  let md = 0, iAt = 0;
  for (let i = 0; i < n; i++) { const dd = Math.abs(js.hist.R0[i] - py.R0[i]); if (dd > md) { md = dd; iAt = i; } }
  let la = 0; const nl = Math.min(js.launch.length, py.launch.length);
  for (let i = 0; i < nl; i++) la = Math.max(la, Math.abs(js.launch[i][1] - py.launch[i][1]) / Math.max(1e-12, Math.abs(py.launch[i][1])));
  // compare only while the throat is near rest, before round-off growth dominates
  let mdNear = 0; for (let i = 0; i < n; i++) { if (Math.abs(py.R0[i] - 1) > 0.01) break; mdNear = Math.max(mdNear, Math.abs(js.hist.R0[i] - py.R0[i])); }
  console.log(`${name.padEnd(18)} js: ${js.outcome} at ${js.lifetime.toFixed(2)} | py: ${py.outcome} at ${py.lifetime.toFixed(2)} | max|dR0| while near rest ${mdNear.toExponential(2)}, overall ${md.toExponential(2)} at t=${(iAt*0.05).toFixed(2)} | launches ${js.launch.length}/${py.launch.length}, max rel diff ${la.toExponential(2)}`);
}

const t0 = Date.now();
compare('free_normal_pulse', run(0.05, 60, 12, { pulses: [['normal', 0.02, 6, 1]] }), ref.free_normal_pulse);
compare('free_balanced', run(0.05, 60, 45, { trim: ref.trim }), ref.free_balanced);
compare('lqg_d1', run(0.05, 60, 40, { trim: ref.trim, d: 1 }), ref.lqg_d1);
compare('lqg_d2', run(0.05, 60, 40, { trim: ref.trim, d: 2 }), ref.lqg_d2);
compare('lqg_d1_matter', run(0.05, 60, 40, { trim: ref.trim, d: 1, disturb: [20, 'normal', 0.002, 6, 1] }), ref.lqg_d1_matter);
console.log('wall time', ((Date.now() - t0) / 1000).toFixed(1), 's');
