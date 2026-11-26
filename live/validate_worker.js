/* Drives worker.js under node with a fake clock: a symmetric run, then a transit run to t = 60.
 *   node live/validate_worker.js */
const fs = require('fs'), path = require('path');
const rd = (f) => fs.readFileSync(path.join(__dirname, f), 'utf8');
const html = rd('../index.html');
const a = html.indexOf('const LIVE = ') + 13, b = html.indexOf('</script>', a);
const LIVE = JSON.parse(html.slice(a, b).trim().replace(/;$/, ''));
if (!LIVE.transit) LIVE.transit = JSON.parse(rd('../viz_transit.json'));
let now = 0, pending = null; const frames = [];
const ctx = {};
new Function('postMessage', 'performance', 'setTimeout', 'ctx', rd('solver.js').replace(/if \(typeof module[^\n]*\n/, '') + '\n' + rd('solver_full.js').replace(/if \(typeof module[^\n]*\n/, '') + '\n' + rd('worker.js').replace(/\bonmessage = /, 'ctx.onmessage = '))
  ((m) => frames.push(m), { now: () => now }, (fn) => { pending = fn; }, ctx);
const pump = (tEnd) => { let last = null; for (let k = 0; k < 200000; k++) { now += 250; const fn = pending; pending = null; fn(); last = frames[frames.length - 1]; if (frames.length > 50) frames.splice(0, frames.length - 5); if (last && (last.t >= tEnd || last.status !== 'alive')) break; } return last; };
ctx.onmessage({ data: { type: 'init', live: LIVE, cfg: { speed: 12 } } });
let f = pump(15);
console.log('symmetric: t', f.t.toFixed(1), 'R0-1', (f.R0 - 1).toExponential(2), f.status);
ctx.onmessage({ data: { type: 'transit', cfg: { travMass: 5e-3, detector: true } } });
f = pump(60);
console.log('transit:   t', f.t.toFixed(1), 'R0-1', (f.R0 - 1).toExponential(2), f.status, JSON.stringify(f.transit), 'left arrays', f.left.R.length, 'H', f.H.toExponential(1));
ctx.onmessage({ data: { type: 'reset' } });
f = pump(5);
console.log('back to symmetric: t', f.t.toFixed(1), f.status, 'transit' in f);
