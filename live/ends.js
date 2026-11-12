const { createSim } = require('./solver.js');
const trim = require('../runs/viz_reference2.json').trim;
function go(name, setup, T) {
  const sim = createSim({ dx: 0.025, L: 90, outflow: true, pyDx: true });
  sim.initial([], trim); setup(sim);
  const jm = sim.jAt(30); let t = 0, last = 0;
  console.log('--', name);
  for (let s = 0; t < T; s++) {
    sim.rk4(); t = (s + 1) * sim.dt;
    const th = sim.throat();
    if (t - last >= 1 || !(th.R > 0.03) || th.R > 6) {
      last = t;
      const [H, M] = sim.constraints(10), hz = sim.horizons(20);
      console.log(`t=${t.toFixed(2)} R0=${th.R.toFixed(4)} lapse=${Math.exp(th.S).toFixed(3)} H=${H.toExponential(1)} M=${M.toExponential(1)} mass(x=30)=${sim.mass(jm).toFixed(5)} trapped=${hz.trap ? 'R=' + hz.trap.R.toFixed(3) + ' x=' + hz.trap.x.toFixed(2) : '-'} anti=${hz.anti ? 'R=' + hz.anti.R.toFixed(3) : '-'}`);
      if (!(th.R > 0.03) || th.R > 6) break;
    }
  }
}
go('matter dropped in (0.02 from x=6)', (s) => s.inject('normal', 0.02, 6, 1), 40);
go('phantom burst (+0.01 from x=6)', (s) => s.inject('phantom', 0.01, 6, 1), 40);
