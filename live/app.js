(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const css = getComputedStyle(document.documentElement);
  const COL = {};
  ['fg', 'muted', 'phantom', 'matter', 'grid', 'ok', 'crit', 'line', 'bg', 'violet', 'warn'].forEach((k) => { COL[k] = css.getPropertyValue('--' + k).trim(); });
  const DISTS = LIVE.distances.map((D) => D.d);
  const S = { d: 1, controlOn: true, eps: 0, speed: 1, paused: false, mag: Math.pow(10, 1.5), kick: Math.pow(10, -3.5),
              f: null, hr: [], hu: [], hd: [], tLast: 0, nPulses: 0, kicks: [], travMass: Math.pow(10, -2.3), detector: true,
              pv: 0.8, pb: 0.6, shell: false, auto: true, yaw: 0.62, pitch: 0, trails: {} };
  const HN = 600;                  // strip-chart samples (one per 0.05 time units = 30 time units)

  // ---------- solver in a worker (falls back to this thread) ----------
  let worker = null;
  const src = $('solver-src').textContent + '\n' + $('solver-full-src').textContent + '\n' + $('worker-src').textContent;
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
  function cfg() { return { d: S.d, controlOn: S.controlOn, eps: S.eps, speed: S.speed, paused: S.paused, auto: S.auto }; }

  function onFrame(e) {
    const f = e.data;
    if (f.type !== 'frame') return;
    if (f.t < S.tLast - 1e-9) { S.hr = []; S.hu = []; S.hd = []; S.nPulses = 0; S.trails = {}; if (S.f) S.f = null; }          // a new run started
    S.tLast = f.t;
    f.hist.forEach((r, k) => { S.hr.push(r - 1); S.hu.push(0); S.hd.push(f.det ? Math.max(0, f.det[k] || 0) : 0); });
    f.pulses.forEach((p) => {
      if (p.kind === 'control') { S.nPulses++; if (S.hu.length) S.hu[S.hu.length - 1] = p.a; }
      else S.kicks.push({ t: f.t, kind: p.kind });
    });
    while (S.hr.length > HN) { S.hr.shift(); S.hu.shift(); S.hd.shift(); }
    if (f.probes) {                  // remember where each probe in flight has been
      const live = {};
      f.probes.forEach((q) => { if (q.state !== 'fly') return; const tr = S.trails[q.id] || []; const l = tr[tr.length - 1]; if (!l || Math.abs(l[0] - q.s) + Math.abs(l[1] - q.phi) > 0.02) tr.push([q.s, q.phi]); if (tr.length > 600) tr.shift(); live[q.id] = tr; });
      S.trails = live;
    }
    if (f.status !== 'alive' && S.f && f.t >= S.tLast - 1e-9) {
      // the step that hits a singularity produces garbage: keep showing the last valid spacetime
      const good = S.f.status === 'alive' ? S.f : S.f.good;
      if (good) { S.f = Object.assign({}, good, { status: f.status, end: f.end, t: f.t, good }); return; }
    }
    S.f = f;
  }

  // ---------- spacetime view ----------
  const scene = $('scene'), g = scene.getContext('2d');
  const W = scene.width, H = scene.height, CX = W / 2, CY = H / 2 - 8;
  let cosT, sinT, cosP, sinP;
  function setView() { cosT = Math.cos(S.yaw); sinT = Math.sin(S.yaw); cosP = Math.cos(S.pitch); sinP = Math.sin(S.pitch); }
  setView();
  { // drag to turn the picture (sideways: about the vertical; up and down: tilt)
    let last = null;
    let moved = 0;
    scene.addEventListener('pointerdown', (e) => { last = [e.clientX, e.clientY]; scene.setPointerCapture(e.pointerId); scene.style.cursor = 'grabbing'; });
    scene.addEventListener('pointermove', (e) => {
      if (!last) return;
      S.yaw = Math.max(0.12, Math.min(1.45, S.yaw + (e.clientX - last[0]) * 0.006)); S.pitch = Math.max(-0.9, Math.min(0.9, S.pitch + (e.clientY - last[1]) * 0.006));
      moved += Math.abs(e.clientX - last[0]) + Math.abs(e.clientY - last[1]); if (moved > 30) $('turn').classList.add('used');     // the hint has done its job
      last = [e.clientX, e.clientY]; setView();
    });
    const end = () => { last = null; scene.style.cursor = ''; };
    scene.addEventListener('pointerup', end); scene.addEventListener('pointercancel', end);
  }
  function geometry(f) {
    // magnify small changes of R from its starting profile; fall back to true scale as changes grow
    // sides: +1 is x > 0 (right), -1 is x < 0; a mirror-symmetric run carries one set of arrays for both
    const sides = { 1: f, '-1': f.left || f };
    let mx = 0;
    [f, f.left].forEach((a) => { if (a) for (let i = 0; i < a.R.length; i++) mx = Math.max(mx, Math.abs(a.R[i] - a.Rbg[i])); });
    const meff = 1 + (S.mag - 1) / (1 + Math.pow(mx * S.mag / 0.25, 2));
    let flat = false;
    const one = (a) => {
      const n = a.R.length, Rd = new Float32Array(n), z = new Float32Array(n);
      for (let i = 0; i < n; i++) Rd[i] = Math.max(0.005, a.Rbg[i] + meff * (a.R[i] - a.Rbg[i]));
      for (let i = 1; i < n; i++) {
        const dl = Math.max(1e-6, a.ell[i] - a.ell[i - 1]), s = (Rd[i] - Rd[i - 1]) / dl;
        if (Math.abs(s) > 1) flat = true;
        z[i] = z[i - 1] + dl * Math.sqrt(Math.max(0, 1 - s * s));
      }
      return { Rd, z };
    };
    const gr = one(f), gl = f.left ? one(f.left) : gr;
    return { 1: gr, '-1': gl, sides, Rd: gr.Rd, z: gr.z, meff, flat };
  }
  function drawScene(now) {
    g.clearRect(0, 0, W, H);
    const f = S.f; if (!f) return;
    const geo = geometry(f), n = f.R.length;
    const R0d = geo.Rd[0];
    const zoom = Math.min(1, 1.15 / Math.max(R0d, 0.2)) * (R0d > 1 ? 1 : 1);
    const SC = 112 * Math.min(1, 1.6 / Math.max(1, R0d));
    let imax = n - 1;
    while (imax > 4 && [1, -1].some((s) => geo[s].Rd[imax] * SC > H * 0.47 || (geo[s].z[imax] * cosT + geo[s].Rd[imax] * sinT) * SC > W * 0.49)) imax--;
    const P = (i, side, phi) => {
      const z = side * geo[side].z[i], r = geo[side].Rd[i], sd = r * Math.sin(phi);
      return [CX + (z * cosT + sd * sinT) * SC, CY - (r * Math.cos(phi) * cosP - (sd * cosT - z * sinT) * sinP) * SC];
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
    const SD = f.left ? [1, -1] : [1];
    SD.forEach((s) => { const a = geo.sides[s]; for (let i = 1; i <= imax; i++) { mF = Math.max(mF, a.eF[i]); mG = Math.max(mG, a.eG[i]); } });
    const glowF = (e) => e <= 0 ? 0 : Math.max(0, Math.min(1, (Math.log10(e / mF) + 2) / 2)) * Math.max(0.25, Math.min(1, (Math.log10(mF + 1e-30) + 14) / 8));
    const glowG = (e) => e <= 0 ? 0 : Math.max(0, Math.min(1, (Math.log10(e / mG) + 2) / 2)) * Math.max(0.25, Math.min(1, (Math.log10(mG + 1e-30) + 14) / 8));
    const calm = S.tab === 'probe';      // keep the fields in the background so the probe stands out
    for (let i = 1; i <= imax; i += 2) [-1, 1].forEach((s) => {
      const a = geo.sides[s];
      const aF = mF > 1e-16 ? glowF(a.eF[i]) : 0, aG = mG > 1e-16 ? glowG(a.eG[i]) : 0;
      if (aF > 0.05) ring(i, s, COL.phantom, 1 + (calm ? 1 : 3) * aF, (0.25 + 0.75 * aF) * (calm ? 0.4 : 1));
      if (aG > 0.05) ring(i, s, COL.matter, 1 + 3 * aG, 0.25 + 0.75 * aG);
    });
    // controller stations, and the detector rings during a traveller run
    const ic = Math.round(S.d / 0.05);
    if (ic <= imax) [-1, 1].forEach((s) => ring(ic, s, COL.fg, 1.6, S.controlOn ? 0.85 : 0.3, [7, 7]));
    if (ic <= imax && S.tab === 'probe') [-1, 1].forEach((s) => { for (let k = 0; k < 8; k++) { const q = P(ic, s, (k + 0.5) * Math.PI / 4); g.beginPath(); g.arc(q[0], q[1], 4.5, 0, 2 * Math.PI); g.fillStyle = COL.fg; g.globalAlpha = (S.controlOn ? 1 : 0.35) * (Math.sin((k + 0.5) * Math.PI / 4) > 0 ? 1 : 0.5); g.fill(); } g.globalAlpha = 1; });
    if (f.transit) { const id = Math.round(f.transit.xs / 0.05); if (id <= imax) [-1, 1].forEach((s) => ring(id, s, COL.matter, 1.4, f.transit.detector && S.controlOn ? 0.6 : 0.2, [3, 6])); }
    // throat
    const alive = f.status === 'alive';
    const tc = !alive ? (f.status === 'collapse' ? COL.crit : COL.fg) : (!S.controlOn ? COL.muted : (Math.abs(f.R0 - 1) < 0.02 ? COL.ok : COL.warn));
    if (!f.trap) { g.shadowColor = tc; g.shadowBlur = 12; ring(0, 1, tc, 3, 1); g.shadowBlur = 0; }
    // probes: a point on the surface at distance s from the throat and angle phi, with its trail
    const PS = (sr, phi) => { const side = sr >= 0 ? 1 : -1, q = Math.abs(sr) / 0.05, i = Math.floor(q); if (i >= imax) return null; const a = P(i, side, phi), b = P(i + 1, side, phi), w = q - i; return [a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1])]; };
    (f.probes || []).forEach((q) => {
      if (q.state !== 'fly') return;
      const tr = S.trails[q.id] || [];
      g.strokeStyle = '#fff'; g.lineWidth = 2; g.globalAlpha = 0.55; g.beginPath(); let pen = false;
      tr.forEach(([sr, ph]) => { const pt = PS(sr, ph); if (!pt) { pen = false; return; } if (pen) g.lineTo(pt[0], pt[1]); else g.moveTo(pt[0], pt[1]); pen = true; });
      g.stroke(); g.globalAlpha = 1;
      const pt = PS(q.s, q.phi); if (!pt) return;
      g.shadowColor = '#fff'; g.shadowBlur = 14; g.fillStyle = '#fff'; g.globalAlpha = Math.sin(q.phi) > 0 ? 1 : 0.6;
      g.beginPath(); g.arc(pt[0], pt[1], 7, 0, 2 * Math.PI); g.fill(); g.shadowBlur = 0; g.globalAlpha = 1;
    });
    $('maglabel').innerHTML = (geo.meff > 1.5 ? 'small changes drawn ×' + Math.round(geo.meff) : 'drawn to scale') + (S.auto && f.pace > 1.3 && f.status === 'alive' && !S.paused ? '<br>fast-forward ×' + f.pace.toFixed(f.pace < 3 ? 1 : 0) : '') + (f.behind ? '<br><span class="slow">running slower than requested</span>' : '') + (geo.flat ? '<br>space too curved to draw exactly' : '');
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
      let md = 0; for (let i = 0; i < n; i++) md = Math.max(md, S.hd[i]);
      if (md > 0) {              // matter passing the detector, scaled to its own peak
        ctx.strokeStyle = COL.violet; ctx.lineWidth = 1.6 * PX; ctx.globalAlpha = 0.9; ctx.beginPath();
        for (let i = 0; i < n; i++) { const x = b.x0 + ((HN - n + i) / HN) * b.w, y = mid - (S.hd[i] / md) * (b.hgt / 2 - 6); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
        ctx.stroke(); ctx.globalAlpha = 1;
      }
    }
    { // profiles
      const { c, x: ctx } = charts[2], b = frameBox(c, ctx), mid = b.top + b.hgt / 2, f = S.f;
      if (!f) return;
      if (S.tab === 'probe' && f.left && f.S) {
        // Energy of the latest probe against position.  The hill is the least energy a body with this probe's angular
        // momentum needs to be at each point: e^S sqrt(1 + L^2/R^2), in units of its rest energy.  The line is the energy
        // it has.  It turns back where the line meets the hill.
        const q = f.probes && f.probes.length ? f.probes[f.probes.length - 1] : null;
        if (q && q.En > 0) S.lastProbe = { En: q.En, L: q.L, s: q.s, state: q.state };
        const lp = S.lastProbe, xmax = 12, xAt = (xx) => b.x0 + ((xx + xmax) / (2 * xmax)) * b.w;
        ctx.textAlign = 'center';
        for (let t = -xmax; t <= xmax; t += 4) { const x = xAt(t); ctx.fillText(String(t), x, c.height - 6 * PX); ctx.globalAlpha = 0.3; ctx.beginPath(); ctx.moveTo(x, b.top); ctx.lineTo(x, b.top + b.hgt); ctx.stroke(); ctx.globalAlpha = 1; }
        if (!lp) { ctx.fillText('Launch a probe to see its energy', b.x0 + b.w / 2, mid); return; }
        const hill = (a, i) => Math.exp(a.S[i]) * Math.sqrt(1 + lp.L * lp.L / (a.R[i] * a.R[i]));
        let top = lp.En; for (let i = 0; i < f.x.length; i++) top = Math.max(top, hill(f, i), hill(f.left, i));
        const y0 = 1, y1 = 1 + (top - 1) * 1.15 + 1e-3, yAt = (v) => b.top + b.hgt - 4 - ((v - y0) / (y1 - y0)) * (b.hgt - 10);
        ctx.textAlign = 'right'; ctx.fillText(top.toFixed(2), b.x0 - 5 * PX, yAt(top) + 4 * PX); ctx.fillText('1', b.x0 - 5 * PX, yAt(1) + 2 * PX);
        ctx.beginPath(); ctx.moveTo(xAt(-xmax), yAt(1));
        for (let i = f.left.x.length - 1; i >= 0; i--) ctx.lineTo(xAt(-f.left.x[i]), yAt(hill(f.left, i)));
        for (let i = 0; i < f.x.length; i++) ctx.lineTo(xAt(f.x[i]), yAt(hill(f, i)));
        ctx.lineTo(xAt(xmax), yAt(1)); ctx.closePath();
        ctx.fillStyle = COL.muted; ctx.globalAlpha = 0.28; ctx.fill(); ctx.globalAlpha = 0.9; ctx.strokeStyle = COL.muted; ctx.lineWidth = 1.6 * PX; ctx.stroke(); ctx.globalAlpha = 1;
        const passes = lp.En > Math.max(hill(f, 0), hill(f.left, 0));
        ctx.strokeStyle = passes ? COL.ok : COL.warn; ctx.lineWidth = 1.6 * PX; ctx.beginPath(); ctx.moveTo(b.x0, yAt(lp.En)); ctx.lineTo(b.x0 + b.w, yAt(lp.En)); ctx.stroke();
        if (lp.state === 'fly' && Math.abs(lp.s) <= xmax) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(xAt(lp.s), yAt(lp.En), 5 * PX, 0, 2 * Math.PI); ctx.fill(); }
        ctx.textAlign = 'left'; ctx.fillStyle = passes ? COL.ok : COL.warn;
        ctx.fillText(passes ? 'enough energy to clear the throat' : 'not enough: it turns back', b.x0 + 6 * PX, b.top + 13 * PX);
        return;
      }
      const xmax = 12, two = !!f.left, xAt = two ? (xx) => b.x0 + ((xx + xmax) / (2 * xmax)) * b.w : (xx) => b.x0 + (xx / xmax) * b.w;
      ctx.textAlign = 'center';
      for (let t = two ? -xmax : 0; t <= xmax; t += two ? 4 : 2) { const x = xAt(t); ctx.fillText(String(t), x, c.height - 6 * PX); ctx.globalAlpha = 0.3; ctx.beginPath(); ctx.moveTo(x, b.top); ctx.lineTo(x, b.top + b.hgt); ctx.stroke(); ctx.globalAlpha = 1; }
      let mx = 1e-9; [f, f.left].forEach((a) => { if (a) for (let i = 0; i < a.x.length; i++) mx = Math.max(mx, Math.abs(a.dF[i]), Math.abs(a.G[i])); });
      const e = Math.pow(10, Math.floor(Math.log10(mx))), q = mx / e, sc = (q <= 1 ? 1 : q <= 2 ? 2 : q <= 5 ? 5 : 10) * e;
      ctx.textAlign = 'right'; ctx.fillText('+' + sci(sc), b.x0 - 5 * PX, b.top + 10 * PX); ctx.fillText('0', b.x0 - 5 * PX, mid + 4 * PX); ctx.fillText('−' + sci(sc), b.x0 - 5 * PX, b.top + b.hgt - 2);
      ctx.strokeStyle = COL.line; ctx.beginPath(); ctx.moveTo(b.x0, mid); ctx.lineTo(b.x0 + b.w, mid); ctx.stroke();
      const yAt = (v) => mid - (v / sc) * (b.hgt / 2 - 4);
      (two ? [[f, 1], [f.left, -1]] : [[f, 1]]).forEach(([a, sgn]) => [['dF', COL.phantom], ['G', COL.matter]].forEach(([key, col]) => {
        const arr = a[key];
        ctx.strokeStyle = col; ctx.lineWidth = 1.6 * PX; ctx.beginPath();
        for (let i = 0; i < a.x.length; i++) { const p = [xAt(sgn * a.x[i]), yAt(arr[i])]; if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); }
        ctx.stroke();
      }));
      if (f.trap) { ctx.fillStyle = COL.crit; ctx.globalAlpha = 0.25; const x0 = two ? xAt(-f.trap.x) : b.x0; ctx.fillRect(x0, b.top, xAt(f.trap.x) - x0, b.hgt); ctx.globalAlpha = 1; }
      ctx.setLineDash([5 * PX, 5 * PX]); ctx.strokeStyle = COL.fg; ctx.globalAlpha = 0.6;
      (two ? [S.d, -S.d] : [S.d]).forEach((q) => { ctx.beginPath(); ctx.moveTo(xAt(q), b.top); ctx.lineTo(xAt(q), b.top + b.hgt); ctx.stroke(); });
      ctx.setLineDash([]); ctx.globalAlpha = 1;
      (f.probes || []).forEach((q) => { if (q.state !== 'fly' || Math.abs(q.s) > xmax) return; ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(xAt(q.s), mid, 4 * PX, 0, 2 * Math.PI); ctx.fill(); });
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
    $('trav-out').textContent = S.travMass.toFixed(4) + ' throat radii';
    $('det-on').setAttribute('aria-pressed', String(S.detector)); $('det-off').setAttribute('aria-pressed', String(!S.detector));
    $('pv-out').textContent = Math.round(S.pv * 100) + '% of light speed';
    $('pb-out').textContent = S.pb.toFixed(2) + (S.pb < 1 ? ' (inside the mouth)' : ' (outside the mouth)');
    $('shell-on').setAttribute('aria-pressed', String(S.shell)); $('shell-off').setAttribute('aria-pressed', String(!S.shell));
    $('speed-out').textContent = S.speed.toFixed(1) + ' time units/s';
    $('pace-auto').setAttribute('aria-pressed', String(S.auto)); $('pace-man').setAttribute('aria-pressed', String(!S.auto));
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
    const tr = f.transit || null, kind = !tr ? 'sym' : (S.tab === 'probe' ? 'probe' : 'trav');
    if (S.kind !== kind) {
      S.kind = kind;
      $('ro-m-lab').textContent = { sym: 'Mass seen from far away', trav: 'Came out the far side', probe: 'Latest probe' }[kind];
      $('kick-m').disabled = $('kick-p').disabled = !!tr; $('leg-det').hidden = !tr; $('leg-probe').hidden = kind !== 'probe';
      $('c-u-cap').textContent = tr ? 'Bars: field amplitude of each pulse (up opens, down shuts, logarithmic). Violet line: matter passing the detector.' : 'Field amplitude of each pulse. Up opens, down shuts. Logarithmic.';
      $('c-p-title').textContent = kind === 'probe' ? 'Probe energy' : 'Inside the spacetime now';
      $('c-p-cap').textContent = kind === 'probe' ? 'Grey hill: the energy a body needs to be at each point, given how far off-centre the probe was aimed. Line: the energy the probe has, in units of its rest energy.' : tr ? 'Fields on both sides of the throat; the traveller comes in from the right. After a crossing the exotic field settles into a new, lopsided shape.' : 'Pulse fields against distance from the throat; red marks a horizon.';
      if (!tr) $('ro-m-sub').textContent = 'change since the start';
    }
    if (kind === 'probe') {
      const q = f.probes && f.probes.length ? f.probes[f.probes.length - 1] : null;
      $('ro-m').textContent = !q ? 'none yet' : { fly: 'in flight', through: 'out the far side', back: 'swung back' }[q.state];
      $('ro-m-sub').textContent = !q ? 'press Launch probe' : 'its clock: ' + q.tau.toFixed(1) + ' of our ' + q.age.toFixed(1);
    } else if (tr) {
      $('ro-m').textContent = tr.through ? pct(tr.through[0], 1) : (f.t < tr.arrive ? 'on its way' : 'crossing');
      $('ro-m-sub').textContent = tr.through ? pct(tr.through[1], 1) + ' bounced back' : (tr.detector && tr.estMass > 1e-5 ? 'detector reads mass ' + tr.estMass.toFixed(4) : 'traveller mass ' + tr.mass.toFixed(4));
    } else $('ro-m').textContent = fmt(f.mass);
    $('ro-u').textContent = fmt(f.spent); $('ro-np').textContent = S.nPulses + ' pulses';
    $('ro-h').textContent = f.trap ? 'R = ' + f.trap.R.toFixed(3) : 'none';
    $('ro-h2').textContent = f.trap ? 'black hole forming' : (f.anti ? 'throat expanding away' : 'no trapped surface');
    const H = Math.max(f.H, f.M); $('ro-c').textContent = H.toExponential(1); $('ro-c-box').dataset.ok = String(H < 1e-3);
    const st = $('state');
    const tOn = S.d + 10;
    if (f.status !== 'alive') { st.dataset.s = 'lost'; st.textContent = { collapse: 'Black hole', expand: 'Blown open', breakdown: 'Solution broke down' }[f.status]; }
    else if (!S.controlOn) { st.dataset.s = 'free'; st.textContent = 'Uncontrolled'; }
    else if (f.t < tOn) { st.dataset.s = 'waiting'; st.textContent = 'Controller starts at t = ' + tOn; }
    else if (tr && tr.mass > 0 && f.t < tr.arrive - 2 && Math.abs(dr) < 0.02) { st.dataset.s = 'holding'; st.textContent = 'Holding. Traveller arrives at t = ' + tr.arrive; }
    else if (tr && tr.mass > 0 && f.t < tr.arrive + 4 && Math.abs(dr) < 0.02) { st.dataset.s = 'holding'; st.textContent = 'Traveller crossing'; }
    else if (Math.abs(dr) < 0.02) { st.dataset.s = 'holding'; st.textContent = 'Holding'; }
    else { st.dataset.s = 'drifting'; st.textContent = 'Losing it'; }
  }
  DISTS.forEach((d) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = String(d); b.addEventListener('click', () => { S.d = d; send({ type: 'set', cfg: cfg() }); paintStatic(); }); $('dist').appendChild(b); });
  $('ctl-on').addEventListener('click', () => { S.controlOn = true; send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('ctl-off').addEventListener('click', () => { S.controlOn = false; send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('err').addEventListener('input', (e) => { S.eps = Number(e.target.value) / 100; send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('kick').addEventListener('input', (e) => { S.kick = Math.pow(10, Number(e.target.value)); paintStatic(); });
  $('speed').addEventListener('input', (e) => { S.speed = Number(e.target.value); send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('pace-auto').addEventListener('click', () => { S.auto = true; send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('pace-man').addEventListener('click', () => { S.auto = false; send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('mag').addEventListener('input', (e) => { S.mag = Math.pow(10, Number(e.target.value)); paintStatic(); });
  $('trav').addEventListener('input', (e) => { S.travMass = Math.pow(10, Number(e.target.value)); paintStatic(); });
  $('det-on').addEventListener('click', () => { S.detector = true; paintStatic(); });
  $('det-off').addEventListener('click', () => { S.detector = false; paintStatic(); });
  const launch = () => send({ type: 'probe', v: S.pv, b: S.pb });
  function startRun() {            // a fresh run of whichever experiment the open tab shows
    S.hr = []; S.hu = []; S.hd = []; S.nPulses = 0; S.trails = {}; S.lastProbe = null;
    if (S.tab === 'trav') { S.d = 1; send({ type: 'transit', cfg: Object.assign(cfg(), { travMass: S.travMass, detector: S.detector }) }); }
    else if (S.tab === 'probe') { S.d = 1; send({ type: 'transit', cfg: Object.assign(cfg(), { travMass: S.shell ? 5e-3 : 0, detector: true }) }); launch(); }
    else send({ type: 'reset' });
    paintStatic();
  }
  const TABS = { hold: 'hold-open', trav: 'traveller', probe: 'probe' };
  function setTab(tab, start) {
    S.tab = tab; document.documentElement.dataset.tab = tab;
    Object.keys(TABS).forEach((k) => $('tab-' + k).setAttribute('aria-selected', String(k === tab)));
    if (start) startRun();
  }
  const tabOfHash = () => Object.keys(TABS).find((k) => location.hash === '#' + TABS[k]) || 'hold';
  Object.keys(TABS).forEach((k) => $('tab-' + k).addEventListener('click', () => { if (S.tab !== k) { setTab(k, true); try { location.hash = TABS[k]; } catch (e) {} } }));
  window.addEventListener('hashchange', () => { const t = tabOfHash(); if (t !== S.tab) setTab(t, true); });
  $('pv').addEventListener('input', (e) => { S.pv = Number(e.target.value); paintStatic(); });
  $('pb').addEventListener('input', (e) => { S.pb = Number(e.target.value); paintStatic(); });
  $('launch').addEventListener('click', launch);
  $('shell-on').addEventListener('click', () => { if (!S.shell) { S.shell = true; startRun(); } });
  $('shell-off').addEventListener('click', () => { if (S.shell) { S.shell = false; startRun(); } });
  $('send').addEventListener('click', startRun);
  $('kick-m').addEventListener('click', () => send({ type: 'kick', field: 'normal', amp: S.kick }));
  $('kick-p').addEventListener('click', () => send({ type: 'kick', field: 'phantom', amp: S.kick }));
  $('pause').addEventListener('click', () => { S.paused = !S.paused; send({ type: 'set', cfg: cfg() }); paintStatic(); });
  $('reset').addEventListener('click', startRun);
  const tb = $('margin-table').tBodies[0];
  LIVE.distances.forEach((D) => {
    const tr = document.createElement('tr'), tol = Math.min(Math.abs(D.margin.gain_lo), D.margin.gain_hi);
    [String(D.d), D.margin.loop_delay.toFixed(1), '±' + pct(tol), '±' + pct(D.margin.best)].forEach((t) => { const td = document.createElement('td'); td.textContent = t; tr.appendChild(td); });
    tb.appendChild(tr);
  });

  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) S.speed = 1;
  setTab(tabOfHash(), false);
  paintStatic();
  send({ type: 'init', live: LIVE, cfg: cfg() });
  if (S.tab !== 'hold') startRun();
  function loop(now) { drawScene(now); drawCharts(); paintLive(); requestAnimationFrame(loop); }
  window.__live = { S, send };
  requestAnimationFrame(loop);
})();
