'use strict';

/* Scientific plots modelled on GNU Radio's Qt GUI sinks (frequency, waterfall,
   time, constellation): framed plot area, dotted major grid, labelled axes with
   units, legend toggles, crosshair readout, drag-to-zoom and double-click reset.
   Pure canvas, no dependencies. */

const Plot = (() => {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const M = { l: 58, r: 14, t: 12, b: 40 };
  const font = px => `${px}px ${css('--sans') || 'system-ui, sans-serif'}`;

  /* 1-2-5 tick steps, the same family Qwt uses in GNU Radio sinks. */
  function ticks(lo, hi, target = 6) {
    if (!(hi > lo)) return [lo];
    const raw = (hi - lo) / target; const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => raw <= s);
    const out = []; for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  function niceDomain(lo, hi, pad = 0) {
    const span = hi - lo || Math.abs(hi) || 1; lo -= span * pad; hi += span * pad;
    const t = ticks(lo, hi); const step = t.length > 1 ? t[1] - t[0] : span;
    return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
  }
  /* Engineering units for an axis, chosen from its largest magnitude. */
  function unitFor(kind, lo, hi) {
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    if (kind === 'Hz') return m >= 1e9 ? ['GHz', 1e9] : m >= 1e6 ? ['MHz', 1e6] : m >= 1e3 ? ['kHz', 1e3] : ['Hz', 1];
    if (kind === 's') return m >= 1 ? ['s', 1] : m >= 1e-3 ? ['ms', 1e-3] : ['µs', 1e-6];
    return [kind || '', 1];
  }
  function fmtTick(v, step) {
    const d = Math.max(0, Math.min(6, -Math.floor(Math.log10(Math.abs(step) || 1) + 1e-9)));
    return v.toFixed(d).replace('-', '−');
  }
  function fmtVal(v, unit, scale, digits = 4) { return `${Number((v / scale).toPrecision(digits))} ${unit}`.trim().replace('-', '−'); }

  function setup(c) {
    const r = c.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    c.width = Math.max(1, Math.round(r.width * dpr)); c.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: r.height };
  }

  function theme() {
    return {
      bg: css('--plot-bg'), frame: css('--plot-frame'), grid: css('--plot-grid'), text: css('--plot-text'), title: css('--ink-2'),
      series: [css('--trace-0'), css('--trace-1'), css('--trace-2'), css('--trace-3')], span: css('--plot-span'), mark: css('--ink'),
    };
  }

  /* Axes, grid and labels for a plot area; returns mapping functions. */
  function frame(ctx, w, h, spec, T, right = M.r, ox = 0) {
    const L = M.l + ox, R = w - right, Tp = M.t, B = h - M.b;
    const [xu, xs] = unitFor(spec.x.unit, spec.x.domain[0], spec.x.domain[1]);
    const [yu, ys] = unitFor(spec.y.unit, spec.y.domain[0], spec.y.domain[1]);
    const X = v => L + (v - spec.x.domain[0]) / (spec.x.domain[1] - spec.x.domain[0]) * (R - L);
    const Y = v => B - (v - spec.y.domain[0]) / (spec.y.domain[1] - spec.y.domain[0]) * (B - Tp);
    ctx.fillStyle = T.bg; ctx.fillRect(L, Tp, R - L, B - Tp);
    ctx.font = font(10.5); ctx.fillStyle = T.text;
    const span = (d, s) => [Math.min(d[0], d[1]) / s, Math.max(d[0], d[1]) / s];
    const xt = ticks(...span(spec.x.domain, xs), Math.max(3, Math.floor((R - L) / 70)));
    const yt = ticks(...span(spec.y.domain, ys), Math.max(3, Math.floor((B - Tp) / 34)));
    ctx.strokeStyle = T.grid; ctx.lineWidth = 1; ctx.setLineDash([1, 3]);
    xt.forEach(v => { const x = Math.round(X(v * xs)) + .5; if (x < L || x > R) return; ctx.beginPath(); ctx.moveTo(x, Tp); ctx.lineTo(x, B); ctx.stroke(); });
    yt.forEach(v => { const y = Math.round(Y(v * ys)) + .5; if (y < Tp || y > B) return; ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(R, y); ctx.stroke(); });
    ctx.setLineDash([]);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const xstep = xt.length > 1 ? xt[1] - xt[0] : 1; const ystep = yt.length > 1 ? yt[1] - yt[0] : 1;
    xt.forEach(v => { const x = X(v * xs); if (x < L - 1 || x > R + 1) return; ctx.fillText(fmtTick(v, xstep), x, B + 6); ctx.beginPath(); ctx.moveTo(Math.round(x) + .5, B); ctx.lineTo(Math.round(x) + .5, B + 4); ctx.strokeStyle = T.frame; ctx.stroke(); });
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    yt.forEach(v => { const y = Y(v * ys); if (y < Tp - 1 || y > B + 1) return; ctx.fillText(fmtTick(v, ystep), L - 7, y); ctx.beginPath(); ctx.moveTo(L - 4, Math.round(y) + .5); ctx.lineTo(L, Math.round(y) + .5); ctx.strokeStyle = T.frame; ctx.stroke(); });
    ctx.fillStyle = T.title; ctx.font = font(11);
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(`${spec.x.label}${xu ? ` (${xu})` : ''}`, (L + R) / 2, h - 4);
    ctx.save(); ctx.translate(L - 45, (Tp + B) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle';
    ctx.fillText(`${spec.y.label}${yu ? ` (${yu})` : ''}`, 0, 0); ctx.restore();
    ctx.strokeStyle = T.frame; ctx.lineWidth = 1; ctx.strokeRect(L + .5, Tp + .5, R - L - 1, B - Tp - 1);
    return { L, R, T: Tp, B, X, Y, xu, xs, yu, ys };
  }

  const colormaps = {
    /* GNU Radio waterfall default "Multi-Color". */
    multi: ['#000000', '#00007f', '#0000ff', '#00ffff', '#ffff00', '#ff0000'],
    whitehot: ['#000000', '#ffffff'],
    blackhot: ['#ffffff', '#000000'],
  };
  function lut(name) {
    const stops = colormaps[name].map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)));
    const out = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i += 1) {
      const t = i / 255 * (stops.length - 1); const k = Math.min(stops.length - 2, Math.floor(t)); const u = t - k;
      for (let ch = 0; ch < 3; ch += 1) out[i * 3 + ch] = stops[k][ch] + (stops[k + 1][ch] - stops[k][ch]) * u;
    }
    return out;
  }

  /* ---- interaction: crosshair readout, x drag-zoom, double-click reset ---- */
  function interact(c) {
    if (c._bound) return; c._bound = true;
    const redraw = ov => c._redraw && c._redraw(ov);
    const tip = document.getElementById('tip');
    let drag = null;
    const local = e => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const inv = (px, py) => { const g = c._geom; if (!g) return null; const sp = c._spec; const x = sp.x.domain[0] + (px - g.L) / (g.R - g.L) * (sp.x.domain[1] - sp.x.domain[0]); const y = sp.y.domain[0] + (g.B - py) / (g.B - g.T) * (sp.y.domain[1] - sp.y.domain[0]); return { x, y, inside: px >= g.L && px <= g.R && py >= g.T && py <= g.B }; };
    c.addEventListener('pointerdown', e => { if (!c._spec.zoomX) return; const [px] = local(e); const p = inv(px, local(e)[1]); if (p?.inside) { drag = { x0: px, x1: px }; c.setPointerCapture(e.pointerId); } });
    c.addEventListener('pointermove', e => {
      const [px, py] = local(e); const p = inv(px, py);
      if (drag) { drag.x1 = px; redraw({ cross: null, band: [drag.x0, drag.x1] }); tip.hidden = true; return; }
      if (!p || !p.inside) { tip.hidden = true; redraw({}); return; }
      const text = c._spec.readout ? c._spec.readout(p.x, p.y, c._geom) : null;
      redraw({ cross: [px, py] });
      if (text) { tip.textContent = text; tip.style.left = `${e.clientX}px`; tip.style.top = `${e.clientY}px`; tip.hidden = false; }
    });
    c.addEventListener('pointerup', () => {
      if (!drag) return; const a = Math.min(drag.x0, drag.x1), b = Math.max(drag.x0, drag.x1); drag = null;
      if (b - a > 6) { const lo = inv(a, 0).x, hi = inv(b, 0).x; c._zoom = [lo, hi]; }
      redraw({});
    });
    c.addEventListener('pointerleave', () => { if (!drag) { tip.hidden = true; redraw({}); } });
    c.addEventListener('dblclick', () => { c._zoom = null; redraw({}); });
  }
  function overlay(ctx, g, T, ov) {
    if (ov.cross) {
      const [x, y] = ov.cross; ctx.strokeStyle = T.mark; ctx.globalAlpha = .45; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(Math.round(x) + .5, g.T); ctx.lineTo(Math.round(x) + .5, g.B); ctx.moveTo(g.L, Math.round(y) + .5); ctx.lineTo(g.R, Math.round(y) + .5); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    if (ov.band) { const a = Math.max(g.L, Math.min(...ov.band)), b = Math.min(g.R, Math.max(...ov.band)); ctx.fillStyle = T.span; ctx.globalAlpha = .5; ctx.fillRect(a, g.T, b - a, g.B - g.T); ctx.globalAlpha = 1; }
  }

  /* ---- line plot (frequency sink, time sink) ---- */
  function line(c, spec) {
    c._spec = spec;
    const draw = (ov = {}) => {
      const { ctx, w, h } = setup(c); const T = theme(); ctx.clearRect(0, 0, w, h);
      const sp = { ...spec, x: { ...spec.x, domain: c._zoom || spec.x.domain } }; c._spec = { ...spec, x: sp.x };
      const g = frame(ctx, w, h, sp, T); c._geom = g;
      ctx.save(); ctx.beginPath(); ctx.rect(g.L + 1, g.T + 1, g.R - g.L - 2, g.B - g.T - 2); ctx.clip();
      (spec.vspans || []).forEach(s => { ctx.fillStyle = T.span; ctx.fillRect(g.X(s[0]), g.T, Math.max(1, g.X(s[1]) - g.X(s[0])), g.B - g.T); });
      (spec.hlines || []).forEach(l => { ctx.strokeStyle = l.color || T.series[1]; ctx.setLineDash(l.dash || [6, 4]); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(g.L, g.Y(l.y)); ctx.lineTo(g.R, g.Y(l.y)); ctx.stroke(); ctx.setLineDash([]); if (l.label) { ctx.fillStyle = l.color || T.series[1]; ctx.font = font(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(l.label, g.R - 6, g.Y(l.y) - 3); } });
      spec.series.forEach((s, k) => {
        if (s.hidden) return;
        const color = s.color || T.series[k % 4]; ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = s.width || 1.25;
        if (s.dots) { s.y.forEach((v, i) => ctx.fillRect(g.X(s.x[i]) - 1.5, g.Y(v) - 1.5, 3, 3)); if (!s.joined) return; ctx.globalAlpha = .35; }
        ctx.beginPath(); s.y.forEach((v, i) => { const x = g.X(s.x[i]), y = g.Y(Math.max(spec.y.domain[0] - 1e3, v)); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke(); ctx.globalAlpha = 1;
      });
      ctx.restore();
      overlay(ctx, g, T, ov);
    };
    c._redraw = draw; interact(c); draw();
  }

  /* ---- constellation sink ---- */
  function scatter(c, spec) {
    let m = 0; spec.points.forEach(p => { m = Math.max(m, Math.abs(p.i), Math.abs(p.q)); }); (spec.ideal || []).forEach(p => { m = Math.max(m, Math.abs(p.i), Math.abs(p.q)); });
    const d = niceDomain(-(m || 1), m || 1, 0.05);
    const lim = Math.max(Math.abs(d[0]), Math.abs(d[1]));
    spec = { ...spec, x: { ...spec.x, domain: [-lim, lim] }, y: { ...spec.y, domain: [-lim, lim] } };
    spec.readout = spec.readout || ((x, y) => `I ${x.toPrecision(3)}, Q ${y.toPrecision(3)}, |r| ${Math.hypot(x, y).toPrecision(3)}, ∠ ${(Math.atan2(y, x) * 180 / Math.PI).toFixed(1)}°`);
    c._spec = spec;
    const draw = (ov = {}) => {
      const { ctx, w, h } = setup(c); const T = theme(); ctx.clearRect(0, 0, w, h);
      /* Equal aspect: square plot area centred horizontally. */
      const side = Math.min(w - M.l - M.r, h - M.t - M.b); const extra = w - M.l - M.r - side;
      const g = frame(ctx, w, h, spec, T, M.r + extra / 2, extra / 2); c._geom = g;
      ctx.strokeStyle = T.frame; ctx.globalAlpha = .6; ctx.beginPath(); ctx.moveTo(g.X(0), g.T); ctx.lineTo(g.X(0), g.B); ctx.moveTo(g.L, g.Y(0)); ctx.lineTo(g.R, g.Y(0)); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = T.series[0]; ctx.globalAlpha = Math.max(.35, Math.min(.9, 200 / spec.points.length));
      spec.points.forEach(p => ctx.fillRect(g.X(p.i) - 1.5, g.Y(p.q) - 1.5, 3, 3)); ctx.globalAlpha = 1;
      if (spec.ideal) { ctx.strokeStyle = T.series[1]; ctx.lineWidth = 1.5; spec.ideal.forEach(p => { const x = g.X(p.i), y = g.Y(p.q); ctx.beginPath(); ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke(); }); }
      overlay(ctx, g, T, ov);
    };
    c._redraw = draw; interact(c); draw();
  }

  /* ---- waterfall sink with colour bar ---- */
  function image(c, spec) {
    c._spec = spec;
    const draw = (ov = {}) => {
      const { ctx, w, h } = setup(c); const T = theme(); ctx.clearRect(0, 0, w, h);
      const bar = 62; const g = frame(ctx, w, h, spec, T, M.r + bar); c._geom = g;
      const rows = spec.rows; const cols = rows[0].length; const [z0, z1] = spec.z.domain; const map = lut(spec.colormap || 'multi');
      const off = document.createElement('canvas'); off.width = cols; off.height = rows.length; const o = off.getContext('2d'); const img = o.createImageData(cols, rows.length);
      /* Row 0 is the earliest frame; GNU Radio scrolls newest at the top, so time runs downward here like a recording read top to bottom. */
      rows.forEach((row, y) => row.forEach((v, x) => { const k = Math.round(Math.max(0, Math.min(1, (v - z0) / (z1 - z0))) * 255); const p = (y * cols + x) * 4; img.data[p] = map[k * 3]; img.data[p + 1] = map[k * 3 + 1]; img.data[p + 2] = map[k * 3 + 2]; img.data[p + 3] = 255; }));
      o.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = false;
      const yTop = g.Y(spec.y.domain[1]), yBot = g.Y(spec.y.domain[0]);
      ctx.drawImage(off, g.L + 1, Math.min(yTop, yBot) + 1, g.R - g.L - 2, Math.abs(yBot - yTop) - 2);
      (spec.marks || []).forEach(s => { ctx.fillStyle = T.series[1]; const a = g.Y(s[0]), b = g.Y(s[1]); ctx.fillRect(g.L + 1, Math.min(a, b), 3, Math.max(2, Math.abs(b - a))); });
      ctx.strokeStyle = T.frame; ctx.strokeRect(g.L + .5, g.T + .5, g.R - g.L - 1, g.B - g.T - 1);
      /* colour bar */
      const bx = w - M.r - 16, bw = 12; const steps = 128;
      for (let i = 0; i < steps; i += 1) { const k = Math.round(i / (steps - 1) * 255); ctx.fillStyle = `rgb(${map[k * 3]},${map[k * 3 + 1]},${map[k * 3 + 2]})`; const y0 = g.B - (i + 1) / steps * (g.B - g.T); ctx.fillRect(bx, y0, bw, (g.B - g.T) / steps + 1); }
      ctx.strokeStyle = T.frame; ctx.strokeRect(bx + .5, g.T + .5, bw, g.B - g.T - 1);
      ctx.fillStyle = T.text; ctx.font = font(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      const zt = ticks(z0, z1, 5); zt.forEach(v => { const y = g.B - (v - z0) / (z1 - z0) * (g.B - g.T); ctx.fillText(fmtTick(v, zt[1] - zt[0] || 1), bx - 4, y); });
      ctx.save(); ctx.translate(w - 4, (g.T + g.B) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = T.title; ctx.font = font(11); ctx.fillText(spec.z.label, 0, 0); ctx.restore();
      overlay(ctx, g, T, ov);
    };
    c._redraw = draw; interact(c); draw();
  }

  return { line, scatter, image, niceDomain, unitFor, fmtVal, colormaps: Object.keys(colormaps) };
})();
