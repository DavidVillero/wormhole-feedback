/* Checks the probe (test-particle) integrator on the static Ellis wormhole, where the answers are known:
 *   radial fall is a straight line in (t, x); the energy En is conserved; a probe aimed to miss the centre
 *   by b turns back at areal radius R = b when b > 1 and passes through when b < 1.
 *   node live/validate_probe.js */
const { createSimFull } = require('./solver_full.js');
const sim = createSimFull({ dx: 0.025, L: 30 });
sim.initial(0);
function fly(v, b, T) {
  const p = sim.probeLaunch(10, v, b, 0); let E0 = null, dE = 0, xmin = Infinity, tThroat = null;
  for (let k = 0; k * sim.dt < T; k++) {
    sim.probeStep(p, sim.dt); if (E0 === null) E0 = p.En; dE = Math.max(dE, Math.abs(p.En / E0 - 1));
    if (p.x < xmin) xmin = p.x; if (tThroat === null && p.x <= 0) tThroat = (k + 1) * sim.dt;
    if (Math.abs(p.x) > 11) break;
  }
  return { p, dE, xmin, tThroat };
}
let r = fly(0.9, 0, 40);
console.log(`radial, v = 0.9: reaches the throat at t = ${r.tThroat.toFixed(3)} (exact ${(10 / 0.9).toFixed(3)}); its clock reads ${r.p.tau.toFixed(3)} at exit (exact ${(21 / 0.9 * Math.sqrt(1 - 0.81)).toFixed(3)} for 21 units of path); energy drift ${r.dE.toExponential(1)}`);
r = fly(0.9, 0.5, 60);
console.log(`b = 0.5: ends at x = ${r.p.x.toFixed(2)} (passes through), turned by ${(r.p.phi * 180 / Math.PI).toFixed(1)} degrees, energy drift ${r.dE.toExponential(1)}`);
r = fly(0.9, 1.5, 60);
console.log(`b = 1.5: ends at x = ${r.p.x.toFixed(2)} (turns back), closest x = ${r.xmin.toFixed(4)} (exact ${Math.sqrt(1.5 * 1.5 - 1).toFixed(4)}), energy drift ${r.dE.toExponential(1)}`);
r = fly(0.3, 0.99, 200); const r2 = fly(0.3, 1.01, 200);
console.log(`b = 0.99: ends at x = ${r.p.x.toFixed(1)};  b = 1.01: ends at x = ${r2.p.x.toFixed(1)}  (the dividing line is b = 1, the throat radius)`);
