(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const css = getComputedStyle(document.documentElement);
  const COL = {};
  ['fg', 'muted', 'phantom', 'matter', 'grid', 'ok', 'crit', 'line', 'bg', 'violet', 'warn'].forEach((k) => { COL[k] = css.getPropertyValue('--' + k).trim(); });
  const DISTS = LIVE.distances.map((D) => D.d);
  const S = { d: 1, controlOn: true, eps: 0, speed: 3, paused: false, mag: Math.pow(10, 1.5), kick: Math.pow(10, -3.5),
              f: null, hr: [], hu: [], tLast: 0, nPulses: 0, kicks: [] };
  const HN = 600;                  // strip-chart samples (one per 0.05 time units = 30 time units)

  // ---------- solver in a worker (falls back to this thread) ----------
  let worker = null;
  const src = $('solver-src').textContent + '\n' + $('worker-src').textContent;
  try {
    worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
  } catch (e) {
    // no workers here: run the same code on this thread
    const ctx = { onmessage: null };
    new Function('postMessage', 'ctx', src.replace(/\bonmessage = /, 'ctx.onmessage = '))((m) => onFrame({ data: m }), ctx);
    worker = { postMessage: (m) => setTimeout(() => ctx.onmessage({ data: m }), 0) };
  }
  function send(m) { worker.postMessage(m); }
  if (worker.addEventListener) worker.addEventListener('message', onFrame);
  function cfg() { return { d: S.d, controlOn: S.controlOn, eps: S.eps, speed: S.speed, paused: S.paused }; }

  function onFrame(e) {
    const f = e.data;
    if (f.type !== 'frame') return;
    if (f.t < S.tLast - 1e-9) { S.hr = []; S.hu = []; S.nPulses = 0; if (S.f) S.f = null; }          // a new run started
    S.tLast = f.t;
    f.hist.forEach((r) => { S.hr.push(r - 1); S.hu.push(0); });
    f.pulses.forEach((p) => {
      if (p.kind === 'control') { S.nPulses++; if (S.hu.length) S.hu[S.hu.length - 1] = p.a; }
      else S.kicks.push({ t: f.t, kind: p.kind });
    });
    while (S.hr.length > HN) { S.hr.shift(); S.hu.shift(); }
    if (f.status !== 'alive' && S.f && f.t >= S.tLast - 1e-9) {
      // the step that hits a singularity produces garbage: keep showing the last valid spacetime
      const good = S.f.status === 'alive' ? S.f : S.f.good;
      if (good) { S.f = Object.assign({}, good, { status: f.status, end: f.end, t: f.t, good }); return; }
    }
    S.f = f;
  }

  // ---------- spacetime view ----------
  const scene = $('scene'), g = scene.getContext('2d');
  const W = scene.width, H = scene.height, CX = W / 2, CY = H / 2 - 8, TH = 0.62;
  const cosT = Math.cos(TH), sinT = Math.sin(TH);
  function geometry(f) {
    // magnify small changes of R from its starting profile; fall back to true scale as changes grow
    let mx = 0;
    for (let i = 0; i < f.R.length; i++) mx = Math.max(mx, Math.abs(f.R[i] - f.Rbg[i]));
    const meff = 1 + (S.mag - 1) / (1 + Math.pow(mx * S.mag / 0.25, 2));
    const n = f.R.length, Rd = new Float32Array(n), z = new Float32Array(n);
    for (let i = 0; i < n; i++) Rd[i] = Math.max(0.005, f.Rbg[i] + meff * (f.R[i] - f.Rbg[i]));
    let flat = false;
    for (let i = 1; i < n; i++) {
      const dl = Math.max(1e-6, f.ell[i] - f.ell[i - 1]), s = (Rd[i] - Rd[i - 1]) / dl;
      if (Math.abs(s) > 1) flat = true;
      z[i] = z[i - 1] + dl * Math.sqrt(Math.max(0, 1 - s * s));
    }
    return { Rd, z, meff, flat };
  }
  function drawScene(now) {
    g.clearRect(0, 0, W, H);
    const f = S.f; if (!f) return;
    const geo = geometry(f), n = f.R.length;
    const R0d = geo.Rd[0];
    const zoom = Math.min(1, 1.15 / Math.max(R0d, 0.2)) * (R0d > 1 ? 1 : 1);
    const SC = 112 * Math.min(1, 1.6 / Math.max(1, R0d));
    let imax = n - 1;
    while (imax > 4 && (geo.Rd[imax] * SC > H * 0.47 || geo.z[imax] * SC * cosT > W * 0.47)) imax--;
    const P = (i, side, phi) => {
      const z = side * geo.z[i], r = geo.Rd[i], sd = r * Math.sin(phi);
      return [CX + (z * cosT + sd * sinT) * SC, CY - r * Math.cos(phi) * SC];
    };
    const ring = (i, side, col, width, alpha, dash) => {
      for (let half = 0; half < 2; half++) {
        g.beginPath();
        for (let k = 0; k <= 32; k++) { const phi = (half ? 0 : Math.PI) + (k / 32) * Math.PI; const p = P(i, side, phi); if (k === 0) g.moveTo(p[0], p[1]); else g.lineTo(p[0], p[1]); }
        g.globalAlpha = alpha * (half ? 1 : 0.45); g.strokeStyle = col; g.lineWidth = width; g.setLineDash(dash || []); g.stroke();
      }
      g.setLineDash([]); g.globalAlpha = 1;
    };
    // meridians
    for (let j = 0; j < 18; j++) {
      const phi = (j / 18) * 2 * Math.PI;
      g.beginPath();
      for (let s = -1; s <= 1; s += 2) for (let k = 0; k <= imax; k += 2) { const i = s < 0 ? imax - k : k; const p = P(i, s, phi); if (s < 0 && k === 0) g.moveTo(p[0], p[1]); else g.lineTo(p[0], p[1]); }
      g.globalAlpha = Math.sin(phi) > 0 ? 0.32 : 0.13; g.strokeStyle = COL.grid; g.lineWidth = 1; g.stroke();
    }
    g.globalAlpha = 1;
    for (let i = 6; i <= imax; i += 6) [-1, 1].forEach((s) => ring(i, s, COL.grid, 1, 0.7 - 0.45 * i / imax));
    // horizon: shade the trapped region
    if (f.trap) {
      const ih = Math.min(imax, Math.round(f.trap.x / 0.05));
      for (let i = 0; i <= ih; i += 2) [-1, 1].forEach((s) => ring(i, s, '#000', 3, 0.55));
      [-1, 1].forEach((s) => { g.shadowColor = COL.crit; g.shadowBlur = 14; ring(ih, s, COL.crit, 3, 1); g.shadowBlur = 0; });
    }
    // fields: rings brightened by local energy of the phantom and matter pulses
    // brightness relative to the strongest pulse in view (2 decades of range), dimmed when even that is weak
    let mF = 0, mG = 0;
    for (let i = 1; i <= imax; i++) { mF = Math.max(mF, f.eF[i]); mG = Math.max(mG, f.eG[i]); }
    const glowF = (e) => e <= 0 ? 0 : Math.max(0, Math.min(1, (Math.log10(e / mF) + 2) / 2)) * Math.max(0.25, Math.min(1, (Math.log10(mF + 1e-30) + 14) / 8));
    const glowG = (e) => e <= 0 ? 0 : Math.max(0, Math.min(1, (Math.log10(e / mG) + 2) / 2)) * Math.max(0.25, Math.min(1, (Math.log10(mG + 1e-30) + 14) / 8));
    for (let i = 1; i <= imax; i += 2) {
      const aF = mF > 1e-16 ? glowF(f.eF[i]) : 0, aG = mG > 1e-16 ? glowG(f.eG[i]) : 0;
      if (aF > 0.05) [-1, 1].forEach((s) => ring(i, s, COL.phantom, 1 + 3 * aF, 0.25 + 0.75 * aF));
      if (aG > 0.05) [-1, 1].forEach((s) => ring(i, s, COL.matter, 1 + 3 * aG, 0.25 + 0.75 * aG));
    }
    // controller stations
    const ic = Math.round(S.d / 0.05);
    if (ic <= imax) [-1, 1].forEach((s) => ring(ic, s, COL.fg, 1.6, S.controlOn ? 0.85 : 0.3, [7, 7]));
    // throat
    const alive = f.status === 'alive';
    const tc = !alive ? (f.status === 'collapse' ? COL.crit : COL.fg) : (!S.controlOn ? COL.muted : (Math.abs(f.R0 - 1) < 0.02 ? COL.ok : COL.warn));
    if (!f.trap) { g.shadowColor = tc; g.shadowBlur = 12; ring(0, 1, tc, 3, 1); g.shadowBlur = 0; }
    $('maglabel').innerHTML = (geo.meff > 1.5 ? 'small changes drawn ×' + Math.round(geo.meff) : 'drawn to scale') + (f.behind ? '<br><span class="slow">running slower than requested</span>' : '') + (geo.flat ? '<br>space too curved to draw exactly' : '');
    if (!alive && f.end) {
      g.fillStyle = COL.fg; g.textAlign = 'center';
      const msg = { collapse: 'Collapsed into a black hole', expand: 'Blown open', breakdown: 'The numerical solution broke down' }[f.end.kind];
      g.font = '600 30px "Space Grotesk", system-ui, sans-serif'; g.fillText(msg, CX, H - 96);
      g.font = '400 19px "IBM Plex Sans", system-ui, sans-serif'; g.fillStyle = COL.muted;
      const sub = f.end.kind === 'collapse'
        ? 'at t = ' + f.end.t.toFixed(1) + (f.end.horizon ? ', horizon radius ' + f.end.horizon.toFixed(2) : '') + '. A singularity formed, so the solver stops here.'
        : f.end.kind === 'expand' ? 'throat radius passed 8 at t = ' + f.end.t.toFixed(1) + '. Press Start over to try again.'
        : 'constraint residual grew too large at t = ' + f.end.t.toFixed(1) + '.';
      g.fillText(sub, CX, H - 66);
    }
  }

  // ---------- charts ----------
  const charts = ['c-r', 'c-u', 'c-p'].map((id) => { const c = $(id); return { c, x: c.getContext('2d') }; });
  let PX = 1, X0 = 52;
  function fit() { PX = Math.max(1, Math.min(3, window.devicePixelRatio || 1)); charts.forEach(({ c }) => { const w = Math.max(200, c.clientWidth); c.width = Math.round(w * PX); c.height = Math.round(150 * PX); }); X0 = Math.round(50 * PX); }
  fit(); window.addEventListener('resize', fit);
  const sci = (v) => { if (v === 0) return '0'; const e = Math.floor(Math.log10(Math.abs(v))); const m = v / Math.pow(10, e); return (Math.abs(m - Math.round(m)) < 1e-9 ? Math.round(m) : m.toFixed(1)) + 'e' + e; };
  function frameBox(c, ctx) {
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.strokeStyle = COL.line; ctx.lineWidth = PX; ctx.strokeRect(X0 - 0.5, 8.5, c.width - X0 - 9, c.height - 22 * PX - 8);
    ctx.fillStyle = COL.muted; ctx.font = (11 * PX) + 'px "IBM Plex Mono", ui-monospace, monospace';
    return { x0: X0, w: c.width - X0 - 10, top: 9, hgt: c.height - 22 * PX - 9 };
  }
  function timeTicks(c, ctx, b) {
    const t1 = S.tLast, t0 = t1 - HN * 0.05; ctx.textAlign = 'center';
    for (let t = Math.ceil(Math.max(0, t0) / 5) * 5; t <= t1; t += 5) {
      const x = b.x0 + ((t - t0) / (HN * 0.05)) * b.w;
      ctx.fillText(String(t), x, c.height - 6 * PX);
      ctx.globalAlpha = 0.3; ctx.beginPath(); ctx.moveTo(x, b.top); ctx.lineTo(x, b.top + b.hgt); ctx.stroke(); ctx.globalAlpha = 1;
    }
  }
  function drawCharts() {
    const n = S.hr.length;
    { // throat radius
      const { c, x: ctx } = charts[0], b = frameBox(c, ctx), mid = b.top + b.hgt / 2;
      timeTicks(c, ctx, b);
      let mx = 1e-6; for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(S.hr[i]));
      const e = Math.pow(10, Math.floor(Math.log10(mx))), q = mx / e, sc = (q <= 1 ? 1 : q <= 2 ? 2 : q <= 5 ? 5 : 10) * e;
      const yAt = (v) => mid - (v / sc) * (b.hgt / 2 - 4), xAt = (i) => b.x0 + ((HN - n + i) / HN) * b.w;
      ctx.strokeStyle = COL.line; ctx.beginPath(); ctx.moveTo(b.x0, mid); ctx.lineTo(b.x0 + b.w, mid); ctx.stroke();
      ctx.textAlign = 'right'; ctx.fillText('+' + sci(sc), b.x0 - 5 * PX, b.top + 10 * PX); ctx.fillText('0', b.x0 - 5 * PX, mid + 4 * PX); ctx.fillText('−' + sci(sc), b.x0 - 5 * PX, b.top + b.hgt - 2);
      ctx.strokeStyle = COL.fg; ctx.lineWidth = 1.6 * PX; ctx.beginPath();
      for (let i = 0; i < n; i++) { const y = Math.max(b.top, Math.min(b.top + b.hgt, yAt(S.hr[i]))); if (i === 0) ctx.moveTo(xAt(i), y); else ctx.lineTo(xAt(i), y); }
      ctx.stroke();
    }
    { // control pulses, signed log scale 1e-10 .. 1e-1
      const { c, x: ctx } = charts[1], b = frameBox(c, ctx), mid = b.top + b.hgt / 2;
      timeTicks(c, ctx, b);
      const lg = (v) => { const m = Math.abs(v); if (m < 1e-10) return 0; return Math.sign(v) * Math.min(1, (Math.log10(m) + 10) / 9); };
      ctx.strokeStyle = COL.line; ctx.beginPath(); ctx.moveTo(b.x0, mid); ctx.lineTo(b.x0 + b.w, mid); ctx.stroke();
      ctx.textAlign = 'right';
      [[1e-1, '1e-1'], [1e-6, '1e-6']].forEach(([v, t]) => { ctx.fillText('+' + t, b.x0 - 5 * PX, mid - lg(v) * (b.hgt / 2 - 4) + 4 * PX); ctx.fillText('−' + t, b.x0 - 5 * PX, mid + lg(v) * (b.hgt / 2 - 4) + 4 * PX); });
      const bw = Math.max(PX * 1.5, (b.w / HN) * 3);
      for (let i = 0; i < n; i++) {
        const v = S.hu[i]; if (v === 0) continue;
        const hh = lg(v) * (b.hgt / 2 - 4), x = b.x0 + ((HN - n + i) / HN) * b.w;
        ctx.fillStyle = v > 0 ? COL.phantom : COL.matter; ctx.fillRect(x - bw / 2, hh > 0 ? mid - hh : mid, bw, Math.abs(hh));
      }
    }
    { // profiles
      const { c, x: ctx } = charts[2], b = frameBox(c, ctx), mid = b.top + b.hgt / 2, f = S.f;
      if (!f) return;
      const xmax = 12, xAt = (xx) => b.x0 + (xx / xmax) * b.w;
      ctx.textAlign = 'center';
      for (let t = 0; t <= xmax; t += 2) { const x = xAt(t); ctx.fillText(String(t), x, c.height - 6 * PX); ctx.globalAlpha = 0.3; ctx.beginPath(); ctx.moveTo(x, b.top); ctx.lineTo(x, b.top + b.hgt); ctx.stroke(); ctx.globalAlpha = 1; }
      let mx = 1e-9; for (let i = 0; i < f.x.length; i++) mx = Math.max(mx, Math.abs(f.dF[i]), Math.abs(f.G[i]));
      const e = Math.pow(10, Math.floor(Math.log10(mx))), q = mx / e, sc = (q <= 1 ? 1 : q <= 2 ? 2 : q <= 5 ? 5 : 10) * e;
      ctx.textAlign = 'right'; ctx.fillText('+' + sci(sc), b.x0 - 5 * PX, b.top + 10 * PX); ctx.fillText('0', b.x0 - 5 * PX, mid + 4 * PX); ctx.fillText('−' + sci(sc), b.x0 - 5 * PX, b.top + b.hgt - 2);
      ctx.strokeStyle = COL.line; ctx.beginPath(); ctx.moveTo(b.x0, mid); ctx.lineTo(b.x0 + b.w, mid); ctx.stroke();
      const yAt = (v) => mid - (v / sc) * (b.hgt / 2 - 4);
      [[f.dF, COL.phantom], [f.G, COL.matter]].forEach(([arr, col]) => {
        ctx.strokeStyle = col; ctx.lineWidth = 1.6 * PX; ctx.beginPath();
        for (let i = 0; i < f.x.length; i++) { const p = [xAt(f.x[i]), yAt(arr[i])]; if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); }
        ctx.stroke();
      });
      if (f.trap) { ctx.fillStyle = COL.crit; ctx.globalAlpha = 0.25; ctx.fillRect(b.x0, b.top, xAt(f.trap.x) - b.x0, b.hgt); ctx.globalAlpha = 1; }
      ctx.setLineDash([5 * PX, 5 * PX]); ctx.strokeStyle = COL.fg; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(xAt(S.d), b.top); ctx.lineTo(xAt(S.d), b.top + b.hgt); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
  }

  // ---------- readouts and controls ----------
  const pct = (v, dg) => (v * 100).toFixed(dg === undefined ? (Math.abs(v) < 0.01 ? 2 : 1) : dg) + '%';
  const fmt = (v) => (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(1) : v.toFixed(4));
  function paintStatic() {
    const D = LIVE.distances.find((q) => q.d === S.d), mg = D.margin;
    $('dist-out').textContent = S.d + ' throat radi' + (S.d === 1 ? 'us' : 'i');
    [...$('dist').children].forEach((b) => b.setAttribute('aria-pressed', String(Number(b.textContent) === S.d)));
    const tol = Math.min(Math.abs(mg.gain_lo), mg.gain_hi);
    $('margin-text').textContent = 'This controller survives a model error of up to ±' + pct(tol) + ' (loop delay ' + mg.loop_delay.toFixed(1) + '). You set ' + (S.eps >= 0 ? '+' : '−') + pct(Math.abs(S.eps), 1) + '.';
    $('margin-text').style.color = Math.abs(S.eps) > tol ? COL.crit : COL.fg;
    $('margin-best').textContent = 'No controller could survive more than ±' + pct(mg.best) + ' at this distance.';
    $('bar-tol').style.width = Math.min(100, tol * 1000) + '%';
    $('bar-mark').style.left = 'calc(' + (50 + S.eps * 500) + '% - 1px)';
    $('err-out').textContent = (S.eps >= 0 ? '+' : '−') + pct(Math.abs(S.eps), 1);
    $('kick-out').textContent = S.kick.toExponential(1);
    $('speed-out').textContent = S.speed.toFixed(1) + ' time units/s';
    $('mag-out').textContent = '×' + (S.mag < 10 ? S.mag.toFixed(1) : Math.round(S.mag));
    $('ctl-on').setAttribute('aria-pressed', String(S.controlOn)); $('ctl-off').setAttribute('aria-pressed', String(!S.controlOn));
    $('pause').textContent = S.paused ? 'Play' : 'Pause';
  }
  function paintLive() {
    const f = S.f; if (!f) return;
    $('ro-t').textContent = f.t.toFixed(1); $('ro-tau').textContent = 'throat clock ' + f.tau.toFixed(1);
    const dr = f.R0 - 1;
    $('ro-r').textContent = Math.abs(dr) < 1e-3 ? '1 ' + (dr >= 0 ? '+ ' : '− ') + Math.abs(dr).toExponential(1) : f.R0.toFixed(4);
    $('ro-lapse').textContent = 'clock rate there ×' + f.lapse.toFixed(3);
    $('ro-m').textContent = fmt(f.mass);
    $('ro-u').textContent = fmt(f.spent); $('ro-np').textContent = S.nPulses + ' pulses';
    $('ro-h').textContent = f.trap ? 'R = ' + f.trap.R.toFixed(3) : 'none';
    $('ro-h2').textContent = f.trap ? 'black hole forming' : (f.anti ? 'throat expanding away' : 'no trapped surface');
    const H = Math.max(f.H, f.M); $('ro-c').textContent = H.toExponential(1); $('ro-c-box').dataset.ok = String(H < 1e-3);
    const st = $('state');
    const tOn = S.d + 10;
    if (f.status !== 'alive') { st.dataset.s = 'lost'; st.textContent = { collapse: 'Black hole', expand: 'Blown open', breakdown: 'Solution broke down' }[f.status]; }
    else if (!S.controlOn) { st.dataset.s = 'free'; st.textContent = 'Uncontrolled'; }
    else if (f.t < tOn) { st.dataset.s = 'waiting'; st.textContent = 'Controller starts at t = ' + tOn; }
    else if (Math.abs(dr) < 0.02) { st.dataset.s = 'holding'; st.textContent = 'Holding'; }
    else { st.dataset.s = 'drifting'; st.textContent = 'Losing it'; }
  }
  DISTS.forEach((d) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = String(d); b.addEventListener('click', () => { S.d = d; send({ type: 'set', cfg: cfg() }); paintStatic(); }); $('dist').appendChild(b); });
  $('ctl-on').addEventListener('click', () => { S.controlOn = true; send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('ctl-off').addEventListener('click', () => { S.controlOn = false; send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('err').addEventListener('input', (e) => { S.eps = Number(e.target.value) / 100; send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('kick').addEventListener('input', (e) => { S.kick = Math.pow(10, Number(e.target.value)); paintStatic(); });
  $('speed').addEventListener('input', (e) => { S.speed = Number(e.target.value); send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('mag').addEventListener('input', (e) => { S.mag = Math.pow(10, Number(e.target.value)); paintStatic(); });
  $('kick-m').addEventListener('click', () => send({ type: 'kick', field: 'normal', amp: S.kick }));
  $('kick-p').addEventListener('click', () => send({ type: 'kick', field: 'phantom', amp: S.kick }));
  $('pause').addEventListener('click', () => { S.paused = !S.paused; send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('reset').addEventListener('click', () => { S.hr = []; S.hu = []; S.nPulses = 0; send({ type: 'reset' }); });
  const tb = $('margin-table').tBodies[0];
  LIVE.distances.forEach((D) => {
    const tr = document.createElement('tr'), tol = Math.min(Math.abs(D.margin.gain_lo), D.margin.gain_hi);
    [String(D.d), D.margin.loop_delay.toFixed(1), '±' + pct(tol), '±' + pct(D.margin.best)].forEach((t) => { const td = document.createElement('td'); td.textContent = t; tr.appendChild(td); });
    tb.appendChild(tr);
  });

  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) S.speed = 1;
  paintStatic();
  send({ type: 'init', live: LIVE, cfg: cfg() });
  function loop(now) { drawScene(now); drawCharts(); paintLive(); requestAnimationFrame(loop); }
  window.__live = { S, send };
  requestAnimationFrame(loop);
})();
