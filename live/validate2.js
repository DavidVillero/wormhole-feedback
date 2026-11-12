const fs = require('fs');
const src = fs.readFileSync('validate.js', 'utf8').split('const t0 = Date.now();')[0].replace("const ref = require('/mnt/user-data/uploads/hey/outputs/gravity-step/runs/viz_reference.json');", '');
eval(src);
const ref2 = require('/mnt/user-data/uploads/hey/outputs/gravity-step/runs/viz_reference2.json');
const t0 = Date.now();
compare('d1 (dx .025)', run(0.025, 60, 50, { trim: ref2.trim, d: 1 }), ref2.d1);
compare('d1_matter', run(0.025, 60, 50, { trim: ref2.trim, d: 1, disturb: [20, 'normal', 0.003, 6, 1] }), ref2.d1_matter);
compare('d1.5', run(0.025, 60, 50, { trim: ref2.trim, d: 1.5 }), ref2['d1.5']);
console.log('wall time for 150 time units at dx 0.025:', ((Date.now() - t0) / 1000).toFixed(1), 's');
// long run with a sponge on a short grid, against the python run on a long grid (lqg_d1_wr1: L = 167, no sponge)
const py = (() => { const { execSync } = require('child_process');
  return JSON.parse(execSync(`python3 -c "import numpy as np, json; R=np.load('/mnt/user-data/uploads/hey/outputs/gravity-step/runs/lqg_d1_wr1.npz'); print(json.dumps({'t':R['t'].tolist(),'R0':R['R0'].tolist()}))"`).toString()); })();
for (const [L, of] of [[60, true], [90, true], [170, false]]) {
  const r = run(0.025, L, 150, { trim: ref2.trim, d: 1, outflow: of, constraints: true });
  let md = 0; const n = Math.min(r.hist.R0.length, py.R0.length);
  const rows = [];
  for (let i = 0; i < n; i++) { md = Math.max(md, Math.abs(r.hist.R0[i] - py.R0[i])); if (i % 500 === 0) rows.push(`t=${(i*0.05).toFixed(0)} js ${r.hist.R0[i].toFixed(6)} py ${py.R0[i].toFixed(6)} H ${r.hist.H[i].toExponential(1)}`); }
  console.log('L=' + L + (of ? ' outflow' : ' fixed edge'), ':', r.outcome, r.lifetime.toFixed(1), 'max|dR0| vs long-grid python', md.toExponential(2)); rows.forEach((s) => console.log('    ' + s));
}
