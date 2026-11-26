/* Checks the full-domain port against the Python detector runs (results_transit_sensed_scan_1side_track*.json).
 *   node live/validate_transit.js [mass ...]
 * The grids differ (Python: no outer boundary within the run; here: outgoing-wave edges at |x| = 60), so
 * agreement is expected to a few percent, not to round-off. */
const { createSimFull, createTransitController } = require('./solver_full.js');
const M = require('../viz_transit.json');
const fs = require('fs');
const py = {};
['', '_refine'].forEach((s) => { JSON.parse(fs.readFileSync(__dirname + '/../results_transit_sensed_scan_1side_track' + s + '.json')).rows.forEach((r) => { py[r.traveller_mass.toPrecision(3)] = r; }); });

function run(dMt, T, useDetector) {
  const sim = createSimFull({ dx: 0.025, L: 60 });
  const dM = sim.initialWithTraveller(M.trim, dMt, M.x0, M.width);
  const ctl = createTransitController(M, { tmOn: 10 });
  const every = Math.round(0.05 / sim.dt), cevery = Math.round(0.25 / sim.dt);
  const hist = { R0: [], PR0: [], XT: [] };
  let step = 0, outcome = 'alive', maxOff = 0, E0 = sim.matterEnergy(6), E1 = null, maxH = 0;
  const nsteps = Math.round(T / sim.dt);
  while (true) {
    const t = step * sim.dt;
    if (step % every === 0) ctl.read(t, sim.inflow(M.xs));
    if (step % cevery === 0) {
      const r = ctl.decide(t, hist, useDetector);
      if (r) { const i = Math.max(0, Math.min(Math.round((t - M.d) / 0.05), hist.XT.length - 1)); sim.injectPair(r.u, M.d, 0.3, hist.XT[i]); }
      if (step % (4 * cevery) === 0) maxH = Math.max(maxH, sim.constraints(10)[0]);
    }
    if (step % every === 0) { const th = sim.throat(); hist.R0.push(th.R); hist.PR0.push(th.Rt); hist.XT.push(th.x); if (t >= M.x0 - 2) maxOff = Math.max(maxOff, Math.abs(th.R - 1)); }
    if (E1 === null && t >= M.x0 + 20) E1 = sim.matterEnergy(6);
    if (step >= nsteps) break;
    sim.rk4(); step++;
    const r0 = sim.throat().R;
    if (!(r0 >= 0.5)) { outcome = 'collapse'; break; }
    if (r0 > 2) { outcome = 'expand'; break; }
  }
  return { dM, outcome, lifetime: step * sim.dt, maxOff, est: ctl.estimate, transmitted: E1 ? E1[0] / E0[1] : null, shift: hist.XT[hist.XT.length - 1], maxH };
}

const masses = process.argv.slice(2).map(Number);
(masses.length ? masses : [5e-3, 1.25e-2]).forEach((m) => {
  const t0 = Date.now(), r = run(m, 90, true), p = py[m.toPrecision(3)];
  console.log(`mass ${r.dM.toExponential(3)}: ${r.outcome} at ${r.lifetime.toFixed(1)}, max offset ${r.maxOff.toExponential(2)}, est mass ${r.est.dM.toExponential(3)} arrival ${r.est.ta.toFixed(3)}, transmitted ${r.transmitted === null ? '-' : r.transmitted.toFixed(4)}, throat at ${r.shift.toFixed(3)}, max H ${r.maxH.toExponential(1)}  [${((Date.now() - t0) / 1000).toFixed(0)} s]`);
  if (p) console.log(`   Python:        ${p.outcome} at ${p.lifetime.toFixed(1)}, max offset ${p.max_offset_after_arrival.toExponential(2)}, est mass ${p.est_mass.toExponential(3)} arrival ${p.est_arrival.toFixed(3)}`);
});
