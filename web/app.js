'use strict';

/* DEmod analyst UI. Talks only to the local comparison API the analyst points it at.
   Every number shown is either returned by that API or labelled as a browser-side check. */

const UI_VERSION = '0.4.0';
const MAX_BYTES = 16 * 1024 * 1024;
const SAMPLE_BYTES = { s8: 2, cu8: 2, s16le: 4, s16be: 4, f32le: 8, f32be: 8 };
const SOURCE_LABEL = { analyst_hypothesis: 'My hypothesis', sigmf_metadata: 'SigMF sidecar', capture_log: 'Capture log', unavailable: 'Not supplied', automatic_classifier: 'Classifier', analyst_override: 'Analyst setting' };
const DENOISE_LABEL = { raw: 'Raw', dc_only: 'DC only', dc_and_impulse: 'DC and impulses' };
const RX_SUPPORTED = ['bpsk', 'qpsk', '2fsk'];
const EXAMPLE_JSON = 'examples/synthetic-qpsk-burst.json';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const state = {
  view: 'capture', filter: 'all', branch: 'raw', denoise: 'raw',
  file: null, bytes: null, truncated: false, shaBrowser: null,
  analysis: null, analysisSettings: null, demod: null, demodSettings: null,
  example: null, api: { ok: false, info: null },
};

/* ---------- small helpers ---------- */
function esc(v) { return String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function parseNum(text) {
  const s = String(text ?? '').trim().replace(/[,_\s]/g, '');
  if (!s) return null;
  const m = s.match(/^([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)([kKmMgG])?(?:hz|Hz|HZ)?$/);
  if (!m) return NaN;
  const mult = { k: 1e3, m: 1e6, g: 1e9 }[(m[2] || '').toLowerCase()] || 1;
  return Number(m[1]) * mult;
}
function sig(v, n = 3) { return Number(v.toPrecision(n)).toString(); }
function fmtHz(v, signed = false) {
  if (v == null || !Number.isFinite(v)) return 'Unknown';
  const a = Math.abs(v); const sign = signed && v > 0 ? '+' : v < 0 ? '−' : '';
  if (a >= 1e9) return `${sign}${sig(a / 1e9, 6)} GHz`;
  if (a >= 1e6) return `${sign}${sig(a / 1e6, 6)} MHz`;
  if (a >= 1e3) return `${sign}${sig(a / 1e3, 4)} kHz`;
  return `${sign}${sig(a, 3)} Hz`;
}
function fmtRate(v) { if (!v) return 'Unknown'; return fmtHz(v).replace('Hz', 'S/s'); }
function fmtBytes(n) { if (n >= 1048576) return `${sig(n / 1048576, 3)} MB`; if (n >= 1024) return `${sig(n / 1024, 3)} KB`; return `${n} bytes`; }
function fmtTime(s) { if (!Number.isFinite(s)) return '?'; if (s >= 1) return `${sig(s, 4)} s`; if (s >= 1e-3) return `${sig(s * 1e3, 4)} ms`; return `${sig(s * 1e6, 4)} µs`; }
function fmtNum(v, d = 4) { return v == null || !Number.isFinite(v) ? 'n/a' : Number(v).toFixed(d).replace(/\.?0+$/, '') || '0'; }
function pct(v) { return `${(v * 100).toFixed(1)}%`; }
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function rgb(c) {
  if (c.startsWith('#')) { const h = c.length === 4 ? c.slice(1).split('').map(x => x + x).join('') : c.slice(1); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
  const m = c.match(/\d+/g); return m ? m.slice(0, 3).map(Number) : [0, 0, 0];
}
function el(html) { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; }
function show(node, on) { node.hidden = !on; }

/* ---------- API connection ---------- */
const apiInput = $('#api-url');
function defaultApi() {
  if (location.pathname.startsWith('/ui/') && /^https?:$/.test(location.protocol)) return location.origin;
  try { const saved = localStorage.getItem('demod.api'); if (saved) return saved; } catch (_) { /* storage optional */ }
  return 'http://127.0.0.1:8787';
}
function apiBase() { return apiInput.value.trim().replace(/\/+$/, ''); }
async function checkApi() {
  $('#api-status').textContent = 'Checking';
  $('#api-dot').className = 'dot';
  const ctrl = new AbortController(); const timer = setTimeout(() => ctrl.abort(), 2500);
  try {
    const res = await fetch(`${apiBase()}/health`, { cache: 'no-store', signal: ctrl.signal });
    const info = await res.json();
    state.api = { ok: res.ok && info.status === 'ok', info };
  } catch (_) { state.api = { ok: false, info: null }; }
  clearTimeout(timer);
  const ok = state.api.ok;
  $('#api-dot').className = `dot ${ok ? 'on' : 'off'}`;
  $('#api-status').textContent = ok ? 'Connected' : 'Not reachable';
  const hosted = location.protocol === 'https:' && !/^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
  $('#api-help').innerHTML = ok
    ? `Captures go to this address only. It keeps nothing after replying. Limit ${fmtBytes(state.api.info.max_input_bytes || MAX_BYTES)} per run.`
    : `Start it with <code>python src/local_comparison_api.py</code>.${hosted ? ' From this hosted page your browser may ask to allow local network access, and Safari blocks it. Opening <code>127.0.0.1:8787/ui/</code> avoids both.' : ''}`;
  try { localStorage.setItem('demod.api', apiBase()); } catch (_) { /* optional */ }
  updateButtons();
}

function captureHeaders(s) {
  const h = { 'Content-Type': 'application/octet-stream', 'X-DEmod-IQ-Format': s.iq_format, 'X-DEmod-Sample-Rate': String(s.sample_rate_hz), 'X-DEmod-Metadata-Source': s.metadata_source };
  if (s.centre_frequency_hz != null) h['X-DEmod-Centre-Frequency'] = String(s.centre_frequency_hz);
  if (s.gain_db != null) h['X-DEmod-Gain'] = String(s.gain_db);
  return h;
}
async function post(path, headers) {
  let res;
  try { res = await fetch(`${apiBase()}${path}`, { method: 'POST', headers, body: state.bytes }); }
  catch (_) { throw new Error(`Couldn't reach the local API at ${apiBase()}. Start it with python src/local_comparison_api.py, then check the address in the sidebar.`); }
  let body; try { body = await res.json(); } catch (_) { throw new Error(`The API answered ${res.status} without JSON.`); }
  if (!res.ok) throw new Error(body.error || `The API answered ${res.status}.`);
  return body;
}

/* ---------- settings form ---------- */
const F = { format: $('#p-format'), rate: $('#p-rate'), centre: $('#p-centre'), gain: $('#p-gain'), source: $('#p-source'), mod: $('#p-mod'), sps: $('#p-sps'), timing: $('#p-timing'), cfo: $('#p-cfo') };
let sigmfLoaded = false;

function settings() {
  return {
    iq_format: F.format.value, sample_rate_hz: parseNum(F.rate.value), centre_frequency_hz: parseNum(F.centre.value),
    gain_db: parseNum(F.gain.value), metadata_source: F.source.value, denoise_profile: state.denoise,
    modulation: F.mod.value, samples_per_symbol: parseNum(F.sps.value), timing_offset: parseNum(F.timing.value) ?? 0, carrier_offset_hz: parseNum(F.cfo.value) ?? 0,
  };
}
function captureProblems(s) {
  const p = [];
  if (!(s.sample_rate_hz > 0)) p.push('Enter a positive sample rate.');
  if (Number.isNaN(s.centre_frequency_hz)) p.push('Centre frequency should be a number, or leave it empty.');
  if (Number.isNaN(s.gain_db)) p.push('Gain should be a number, or leave it empty.');
  return p;
}
function setDenoise(v) { state.denoise = v; $$('#p-denoise button').forEach(b => b.setAttribute('aria-checked', String(b.dataset.value === v))); updateSummary(); }

function updateSummary() {
  const s = settings();
  F.rate.setAttribute('aria-invalid', String(F.rate.value !== '' && !(s.sample_rate_hz > 0)));
  F.centre.setAttribute('aria-invalid', String(Number.isNaN(s.centre_frequency_hz)));
  F.gain.setAttribute('aria-invalid', String(Number.isNaN(s.gain_db)));
  const out = $('#capture-summary');
  if (!state.file) { out.textContent = 'No file loaded.'; updateButtons(); return; }
  const size = state.bytes.byteLength; const per = SAMPLE_BYTES[s.iq_format]; const n = Math.floor(size / per);
  let text = `${state.file.name}, ${fmtBytes(state.file.size)}. Read as ${s.iq_format}, that is ${n.toLocaleString()} complex samples`;
  text += s.sample_rate_hz > 0 ? `, or ${fmtTime(n / s.sample_rate_hz)} at ${fmtRate(s.sample_rate_hz)}.` : '.';
  if (size % per) text += ` ${size % per} trailing bytes don't fill a sample, so the format may be wrong.`;
  if (state.truncated) text += ` Only the first ${fmtBytes(size)} will be sent, and the hash covers that slice only.`;
  if (state.analysisSettings && JSON.stringify(pick(state.analysisSettings)) !== JSON.stringify(pick(s))) text += ' Settings changed since the last run, so analyse again to update the evidence.';
  out.textContent = text;
  updateButtons();
}
const pick = s => [s.iq_format, s.sample_rate_hz, s.centre_frequency_hz, s.gain_db, s.metadata_source, s.denoise_profile];

function updateButtons() {
  const s = settings();
  $('#run-analysis').disabled = !state.bytes || captureProblems(s).length > 0 || !state.api.ok;
  $('#run-analysis').title = !state.api.ok ? 'Connect the local API first' : '';
  $('#run-demod').disabled = !state.bytes || !(s.sample_rate_hz > 0) || !state.api.ok;
  $('#export').disabled = !state.analysis && !state.demod;
}

/* SigMF values are labelled as such until the analyst edits one. */
[F.format, F.rate, F.centre].forEach(input => input.addEventListener('input', () => {
  if (sigmfLoaded && F.source.value === 'sigmf_metadata') {
    F.source.value = 'analyst_hypothesis'; sigmfLoaded = false;
    flash('#capture-error', 'You edited a SigMF value, so the source is now "My hypothesis".', true);
  }
}));
Object.values(F).forEach(input => ['input', 'change'].forEach(evt => input.addEventListener(evt, () => { syncBound(); updateSummary(); renderRxHints(); })));

function syncBound() { $$('[data-bind]').forEach(b => { const src = document.getElementById(b.dataset.bind); if (b !== document.activeElement) b.value = src.value; }); }
$$('[data-bind]').forEach(b => ['input', 'change'].forEach(evt => b.addEventListener(evt, () => { const src = document.getElementById(b.dataset.bind); src.value = b.value; src.dispatchEvent(new Event(evt)); })));

function flash(sel, msg, soft = false) {
  const node = $(sel); node.textContent = msg; node.hidden = !msg;
  node.style.background = soft ? 'var(--surface)' : ''; node.style.color = soft ? 'var(--ink-2)' : '';
}

/* ---------- loading a capture ---------- */
async function sha256(buf) {
  if (!crypto?.subtle) return null;
  const d = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, '0')).join('');
}
async function loadBytes(file, buffer, extra = {}) {
  state.file = { name: file.name, size: file.size, ...extra };
  state.truncated = file.size > MAX_BYTES;
  state.bytes = buffer;
  state.shaBrowser = await sha256(buffer);
  state.analysis = null; state.analysisSettings = null; state.demod = null; state.demodSettings = null;
  $('#drop').classList.add('loaded');
  $('#drop-title').textContent = file.name;
  $('#drop-detail').textContent = extra.note || `${fmtBytes(file.size)}. Click to choose a different file.`;
  flash('#capture-error', '');
  renderAll(); updateSummary();
}
async function onFile(file) {
  if (!file) return;
  if (!file.size) { flash('#capture-error', `${file.name} is empty.`); return; }
  state.example = null;
  const buffer = await file.slice(0, Math.min(file.size, MAX_BYTES)).arrayBuffer();
  await loadBytes(file, buffer);
}
$('#file-input').addEventListener('change', e => onFile(e.target.files[0]));
const drop = $('#drop');
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, () => drop.classList.add('over')));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, () => drop.classList.remove('over')));

/* SigMF sidecar: fills interpretation fields and marks them as SigMF-sourced. */
const SIGMF_TYPES = { ci16_le: 's16le', ci16_be: 's16be', ci8: 's8', cu8: 'cu8', cf32_le: 'f32le', cf32_be: 'f32be' };
$('#sigmf-label').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#sigmf-input').click(); } });
$('#sigmf-input').addEventListener('change', async e => {
  const file = e.target.files[0]; if (!file) return;
  try {
    const doc = JSON.parse(await file.text()); const g = doc.global || {}; const c = (doc.captures || [])[0] || {};
    const type = g['core:datatype'];
    if (type && !SIGMF_TYPES[type]) throw new Error(`SigMF datatype ${type} isn't one the API reads. Supported: ${Object.keys(SIGMF_TYPES).join(', ')}.`);
    if (type) F.format.value = SIGMF_TYPES[type];
    if (g['core:sample_rate']) F.rate.value = g['core:sample_rate'];
    if (c['core:frequency'] != null) F.centre.value = c['core:frequency'];
    F.source.value = 'sigmf_metadata'; sigmfLoaded = true;
    flash('#capture-error', `Filled from ${file.name}: ${[type && `format ${SIGMF_TYPES[type]}`, g['core:sample_rate'] && `rate ${fmtRate(g['core:sample_rate'])}`, c['core:frequency'] != null && `centre ${fmtHz(c['core:frequency'])}`].filter(Boolean).join(', ') || 'no usable fields'}.`, true);
    syncBound(); updateSummary();
  } catch (err) { flash('#capture-error', err instanceof SyntaxError ? `${file.name} isn't valid JSON.` : err.message); }
  e.target.value = '';
});

/* Browser-generated QPSK with known truth, for checking the estimators end to end. */
$('#make-test').addEventListener('click', async () => {
  const fs = 250000, sps = 8, cfo = 1500, amp = 8000;
  const quiet = [6000, 4000, 6000], bursts = [1000, 750];
  const total = quiet.reduce((a, b) => a + b) + bursts.reduce((a, b) => a + b) * sps;
  const out = new Int16Array(total * 2); let n = 0; let seed = Date.now() % 2147483647;
  const rand = () => (seed = seed * 16807 % 2147483647) / 2147483647;
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const push = (i, q) => { const ph = 2 * Math.PI * cfo * n / fs; const c = Math.cos(ph), s = Math.sin(ph); out[2 * n] = Math.round(((i * c - q * s) + 0.05 * gauss()) * amp); out[2 * n + 1] = Math.round(((i * s + q * c) + 0.05 * gauss()) * amp); n += 1; };
  quiet.forEach((q, k) => {
    for (let j = 0; j < q; j += 1) push(0, 0);
    if (k < bursts.length) for (let b = 0; b < bursts[k]; b += 1) { const i = (rand() < 0.5 ? -1 : 1) / Math.SQRT2, qq = (rand() < 0.5 ? -1 : 1) / Math.SQRT2; for (let j = 0; j < sps; j += 1) push(i, qq); }
  });
  const buffer = out.buffer; const name = `test-qpsk-sps${sps}-${Date.now().toString(36)}.s16le.iq`;
  state.example = null;
  F.format.value = 's16le'; F.rate.value = String(fs); F.source.value = 'analyst_hypothesis'; F.mod.value = 'qpsk'; F.sps.value = String(sps); F.timing.value = '0'; F.cfo.value = '0'; sigmfLoaded = false;
  syncBound();
  await loadBytes({ name, size: buffer.byteLength }, buffer, { generated: true, truth: { modulation: 'qpsk', samples_per_symbol: sps, carrier_offset_hz: cfo, sample_rate_hz: fs, bursts: bursts.length, noise_std: 0.05 }, note: `Made in this browser. QPSK, ${sps} samples per symbol, ${fmtHz(cfo, true)} carrier offset, two bursts. Known truth, so you can check the estimates.` });
});

/* Recorded example run: lets the hosted page show real API output without a local API. */
$('#open-example').addEventListener('click', async () => {
  try {
    const ex = await (await fetch(EXAMPLE_JSON)).json();
    const iq = await (await fetch(ex.example.file_url)).arrayBuffer();
    const s = ex.example.settings;
    F.format.value = s.iq_format; F.rate.value = s.sample_rate_hz; F.centre.value = s.centre_frequency_hz ?? ''; F.gain.value = s.gain_db ?? ''; F.source.value = s.metadata_source;
    F.mod.value = s.modulation; F.sps.value = s.samples_per_symbol; F.timing.value = s.timing_offset; F.cfo.value = s.carrier_offset_hz; sigmfLoaded = false;
    setDenoise(s.denoise_profile); syncBound();
    await loadBytes({ name: ex.example.file_name, size: iq.byteLength }, iq, { note: 'Example capture. Synthetic, not a real signal.' });
    state.example = ex.example;
    state.analysis = ex.analyse; state.analysisSettings = settings();
    state.demod = ex.demodulate; state.demodSettings = settings();
    renderAll(); updateSummary(); go('evidence');
  } catch (_) { flash('#capture-error', `Couldn't load the example from ${EXAMPLE_JSON}.`); }
});

/* ---------- running ---------- */
async function busy(btn, label, fn) {
  const old = btn.textContent; btn.textContent = label; btn.classList.add('busy'); btn.disabled = true;
  try { await fn(); } finally { btn.textContent = old; btn.classList.remove('busy'); updateButtons(); }
}
$('#run-analysis').addEventListener('click', () => busy($('#run-analysis'), 'Analysing', async () => {
  const s = settings(); const problems = captureProblems(s);
  if (problems.length) { flash('#capture-error', problems.join(' ')); return; }
  flash('#capture-error', '');
  try {
    state.analysis = await post('/analyse', { ...captureHeaders(s), 'X-DEmod-Denoise-Profile': s.denoise_profile });
    state.analysisSettings = s; state.example = null; state.demod = null; state.demodSettings = null;
    renderAll(); updateSummary(); go('evidence');
  } catch (err) { flash('#capture-error', err.message); }
}));

$('#run-demod').addEventListener('click', () => busy($('#run-demod'), 'Running', async () => {
  const s = settings(); const p = captureProblems(s).filter(x => !x.startsWith('Enter a positive'));
  if (!Number.isInteger(s.samples_per_symbol) || s.samples_per_symbol < 1) p.push('Samples per symbol must be a whole number of at least 1.');
  else if (!Number.isInteger(s.timing_offset) || s.timing_offset < 0 || s.timing_offset >= s.samples_per_symbol) p.push(`Timing offset must be a whole number from 0 to ${s.samples_per_symbol - 1}.`);
  if (!Number.isFinite(s.carrier_offset_hz)) p.push('Carrier offset should be a number in Hz.');
  if (!(s.sample_rate_hz > 0)) p.push('Set a sample rate on the Capture page.');
  if (p.length) { flash('#rx-error', p.join(' ')); return; }
  flash('#rx-error', '');
  try {
    state.demod = await post('/demodulate', { ...captureHeaders(s), 'X-DEmod-Modulation': s.modulation, 'X-DEmod-Samples-Per-Symbol': String(s.samples_per_symbol), 'X-DEmod-Timing-Offset': String(s.timing_offset), 'X-DEmod-Carrier-Offset': String(s.carrier_offset_hz) });
    state.demodSettings = s; renderReceiver(); renderReport();
  } catch (err) { flash('#rx-error', err.message); }
}));

/* ---------- navigation ---------- */
function go(view) {
  if (!['capture', 'evidence', 'receiver', 'report'].includes(view)) view = 'capture';
  state.view = view;
  ['capture', 'evidence', 'receiver', 'report'].forEach(v => show($(`#view-${v}`), v === view));
  $$('.nav-link').forEach(a => { if (a.dataset.view === view) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  if (location.hash.slice(1) !== view) history.replaceState(null, '', `#${view}`);
  syncBound();
  requestAnimationFrame(layoutAll);
  window.scrollTo(0, 0);
}
window.addEventListener('hashchange', () => go(location.hash.slice(1)));

/* ---------- tiles and masonry ---------- */
const AV = { raw: ['raw', 'R'], derived: ['derived', 'D'], model: ['model', 'M'], est: ['est', 'E'], rx: ['rx', 'Rx'] };
function tile({ cat = '', title, src, av = 'raw', metric = '', body = '', pad = true, draw = null }) {
  const [cls, letter] = AV[av];
  const node = el(`<article class="tile" data-cat="${cat}"><div class="media${pad ? ' pad' : ''}">${body}</div><div class="cap"><span class="av ${cls}" aria-hidden="true">${letter}</span><span class="cap-text"><span class="cap-title">${esc(title)}</span><span class="cap-src">${esc(src)}</span></span><span class="cap-metric">${esc(metric)}</span></div></article>`);
  node._draw = draw;
  return node;
}
const feeds = { feed: [], 'rx-feed': [] };
function setTiles(id, tiles) { feeds[id] = tiles; layout(id); }
function layout(id) {
  const box = document.getElementById(id); if (!box || box.closest('.view').hidden) return;
  const w = box.clientWidth || box.parentElement.clientWidth;
  const n = w >= 1120 ? 3 : w >= 640 ? 2 : 1;
  box.innerHTML = '';
  const cols = Array.from({ length: n }, () => box.appendChild(el('<div class="col"></div>')));
  feeds[id].forEach(t => {
    const visible = id !== 'feed' || state.filter === 'all' || t.dataset.cat.split(' ').includes(state.filter);
    t.hidden = !visible; if (!visible) return;
    cols.reduce((a, b) => (b.offsetHeight < a.offsetHeight ? b : a)).appendChild(t);
  });
  feeds[id].forEach(t => { if (!t.hidden && t._draw) t._draw(); });
}
function layoutAll() { layout('feed'); layout('rx-feed'); }
let resizeTimer; window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(layoutAll, 120); });
matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', layoutAll);

/* ---------- canvas plots ---------- */
function canvas(ratio) { return `<canvas style="aspect-ratio:${ratio}"></canvas>`; }
function prep(c) {
  const r = c.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
  c.width = Math.max(1, Math.round(r.width * dpr)); c.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}
const tip = $('#tip');
function hover(c, fn) {
  c.addEventListener('pointermove', e => { const r = c.getBoundingClientRect(); const text = fn((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height); if (!text) { tip.hidden = true; return; } tip.textContent = text; tip.style.left = `${e.clientX}px`; tip.style.top = `${e.clientY}px`; tip.hidden = false; });
  c.addEventListener('pointerleave', () => { tip.hidden = true; });
}

function drawSpectrum(c, spec, centre, guide = null) {
  const { ctx, w, h } = prep(c); const f = spec.frequency_hz, p = spec.power_db;
  const floor = Math.max(-100, Math.floor(Math.min(...p) / 10) * 10); const top = 4, bottom = h - 6;
  const X = i => (i / (f.length - 1)) * w; const Y = v => top + (Math.min(0, v) / floor) * (bottom - top);
  ctx.strokeStyle = css('--surface-2'); ctx.lineWidth = 1;
  for (let d = 0; d >= floor; d -= 20) { ctx.beginPath(); ctx.moveTo(0, Y(d) + .5); ctx.lineTo(w, Y(d) + .5); ctx.stroke(); }
  ctx.beginPath(); ctx.moveTo(w / 2 + .5, top); ctx.lineTo(w / 2 + .5, bottom); ctx.stroke();
  ctx.beginPath(); p.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
  ctx.lineTo(w, bottom); ctx.lineTo(0, bottom); ctx.closePath(); ctx.fillStyle = css('--signal-soft'); ctx.fill();
  ctx.beginPath(); p.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.strokeStyle = css('--signal'); ctx.lineWidth = 1.25; ctx.stroke();
  if (guide != null) { ctx.strokeStyle = css('--ink'); ctx.beginPath(); ctx.moveTo(X(guide) + .5, top); ctx.lineTo(X(guide) + .5, bottom); ctx.stroke(); }
  if (!c._hover) {
    c._hover = true;
    hover(c, x => {
      const i = Math.max(0, Math.min(f.length - 1, Math.round(x * (f.length - 1)))); drawSpectrum(c, spec, centre, i);
      const abs = centre != null ? `, ${fmtHz(centre + f[i])} if the centre is right` : '';
      return `${fmtHz(f[i], true)}, ${p[i].toFixed(1)} dB${abs}`;
    });
    c.addEventListener('pointerleave', () => drawSpectrum(c, spec, centre));
  }
}

function colormap() {
  const stops = [css('--surface'), css('--signal-soft'), css('--signal'), css('--ink')].map(rgb);
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i += 1) {
    const t = Math.pow(i / 255, 2.2) * (stops.length - 1); const k = Math.min(stops.length - 2, Math.floor(t)); const u = t - k;
    for (let ch = 0; ch < 3; ch += 1) lut[i * 3 + ch] = stops[k][ch] + (stops[k + 1][ch] - stops[k][ch]) * u;
  }
  return lut;
}
function drawWaterfall(c, wf, segs, fs) {
  const { ctx, w, h } = prep(c); const rows = wf.power_db_relative; if (!rows.length) return;
  const cols = rows[0].length; const off = document.createElement('canvas'); off.width = cols; off.height = rows.length;
  const octx = off.getContext('2d'); const img = octx.createImageData(cols, rows.length); const lut = colormap();
  rows.forEach((row, y) => row.forEach((v, x) => { const k = Math.round(Math.max(0, Math.min(1, (v + 70) / 70)) * 255); const o = (y * cols + x) * 4; img.data[o] = lut[k * 3]; img.data[o + 1] = lut[k * 3 + 1]; img.data[o + 2] = lut[k * 3 + 2]; img.data[o + 3] = 255; }));
  octx.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, w, h);
  const t0 = wf.time_seconds[0] - wf.fft_size / 2 / fs; const t1 = wf.time_seconds[wf.time_seconds.length - 1] + wf.fft_size / 2 / fs;
  ctx.fillStyle = css('--warn');
  (segs || []).forEach(s => { const a = s.sample_start / fs, b = (s.sample_start + s.sample_count) / fs; if (b < t0 || a > t1) return; const ya = Math.max(0, (a - t0) / (t1 - t0)) * h, yb = Math.min(1, (b - t0) / (t1 - t0)) * h; ctx.fillRect(0, ya, 3, Math.max(2, yb - ya)); });
  if (!c._hover) {
    c._hover = true;
    hover(c, (x, y) => { const r = Math.min(rows.length - 1, Math.floor(y * rows.length)); const k = Math.min(cols - 1, Math.floor(x * cols)); const fr = wf.frequency_hz[k]; return `${fmtTime(wf.time_seconds[r])}, ${fmtHz(fr, true)}, ${rows[r][k].toFixed(1)} dB`; });
  }
}

function drawIQ(c, pts, ideal = null, colorBy = null) {
  const { ctx, w, h } = prep(c); const s = Math.min(w, h); const cx = w / 2, cy = h / 2;
  let m = 0; pts.forEach(p => { m = Math.max(m, Math.abs(p.i), Math.abs(p.q)); }); (ideal || []).forEach(p => { m = Math.max(m, Math.abs(p.i), Math.abs(p.q)); }); m = m * 1.12 || 1;
  const X = v => cx + (v / m) * (s / 2 - 10); const Y = v => cy - (v / m) * (s / 2 - 10);
  ctx.strokeStyle = css('--surface-2'); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx + .5, 8); ctx.lineTo(cx + .5, h - 8); ctx.moveTo(8, cy + .5); ctx.lineTo(w - 8, cy + .5); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, (s / 2 - 10) / 1.12, 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = css('--signal'); ctx.globalAlpha = Math.max(0.25, Math.min(0.8, 160 / pts.length + 0.2));
  pts.forEach(p => { ctx.fillRect(X(p.i) - 1.5, Y(p.q) - 1.5, 3, 3); });
  ctx.globalAlpha = 1;
  if (ideal) { ctx.strokeStyle = css('--ink'); ctx.lineWidth = 1.5; ideal.forEach(p => { ctx.beginPath(); ctx.arc(X(p.i), Y(p.q), 6, 0, Math.PI * 2); ctx.stroke(); }); }
}

function drawSeries(c, values) {
  const { ctx, w, h } = prep(c); const m = Math.max(...values.map(Math.abs)) * 1.15 || 1; const cy = h / 2;
  ctx.strokeStyle = css('--surface-2'); ctx.beginPath(); ctx.moveTo(0, cy + .5); ctx.lineTo(w, cy + .5); ctx.stroke();
  ctx.fillStyle = css('--signal');
  values.forEach((v, i) => { const x = 8 + (i / Math.max(1, values.length - 1)) * (w - 16); ctx.fillRect(x - 1.5, cy - (v / m) * (h / 2 - 8) - 1.5, 3, 3); });
}

/* ---------- evidence ---------- */
function branchData() {
  const a = state.analysis.analysis; return state.branch === 'derived' ? a.derived_branch : a.raw_branch;
}
function branchLabel() {
  return state.branch === 'derived' ? `Derived branch, ${DENOISE_LABEL[state.analysis.provenance.denoising_profile] || state.analysis.provenance.denoising_profile}` : 'Raw branch';
}

function renderEvidence() {
  const a = state.analysis; const has = Boolean(a);
  show($('#evidence-empty'), !has); show($('#evidence-bar'), has);
  const ex = state.example;
  show($('#example-notice'), Boolean(ex && has));
  if (ex) $('#example-notice').innerHTML = `Example run. ${esc(ex.description)} Recorded from ${esc(ex.recorded_from)}. <a href="${esc(ex.file_url)}" download>Download the capture</a> to reproduce it on your own API.`;
  if (!has) { $('#evidence-title').textContent = 'Evidence'; $('#evidence-sub').textContent = 'Analyse a capture to fill this page.'; $('#evidence-tags').innerHTML = ''; setTiles('feed', []); return; }

  const inp = a.input; const cap = inp.capture; const fs = cap.sample_rate_hz; const centre = cap.centre_frequency_hz;
  const rawVis = a.analysis.raw_branch.visualization;
  $('#evidence-title').textContent = state.file?.name || 'Capture';
  $('#evidence-sub').textContent = `${(rawVis?.input.sample_count ?? a.manual_dsp.sample_count).toLocaleString()} complex samples analysed by the local API. Run ${a.run_id.slice(0, 8)}.`;
  const tag = (k, v) => `<li>${esc(k)} <b>${esc(v)}</b></li>`;
  $('#evidence-tags').innerHTML = [
    tag('Format', inp.representation.iq_format), tag('Rate', fmtRate(fs)), tag('Centre', centre != null ? fmtHz(centre) : 'unknown'),
    tag('Gain', cap.gain_db != null ? `${cap.gain_db} dB` : 'unknown'), tag('Source', SOURCE_LABEL[inp.representation.iq_format_source] || inp.representation.iq_format_source),
    tag('Cleaning', DENOISE_LABEL[a.provenance.denoising_profile] || a.provenance.denoising_profile),
  ].join('');

  const rawOnly = a.provenance.denoising_profile === 'raw';
  if (rawOnly) state.branch = 'raw';
  $$('#branch button').forEach(b => { b.setAttribute('aria-checked', String(b.dataset.value === state.branch)); b.disabled = rawOnly && b.dataset.value === 'derived'; b.title = b.disabled ? 'Cleaning was set to Raw, so there is no separate derived branch' : ''; });

  const B = branchData(); const vis = B.visualization; const av = state.branch === 'derived' ? 'derived' : 'raw'; const src = branchLabel();
  const tiles = [];

  if (vis) {
    const spec = vis.spectrum;
    tiles.push(tile({ cat: 'spectrum', title: 'Power spectrum', src: `${src}, Hann ${spec.fft_size}-point FFT`, av, metric: `peak ${fmtHz(spec.peak_frequency_hz, true)}`, pad: false,
      body: `${canvas('16 / 9')}<div class="plot-axis"><span>${fmtHz(spec.frequency_hz[0], true)}</span><span>0 Hz offset</span><span>${fmtHz(spec.frequency_hz[spec.frequency_hz.length - 1], true)}</span></div>`,
      draw() { drawSpectrum($('canvas', this), spec, centre); } }));

    const wf = vis.waterfall; const segs = vis.segmentation.segments; const total = vis.input.sample_count / fs;
    const shown = wf.time_seconds.length ? wf.time_seconds[wf.time_seconds.length - 1] + wf.fft_size / 2 / fs : 0;
    tiles.push(tile({ cat: 'spectrum time', title: 'Waterfall', src: `${src}, STFT hop ${wf.hop_size}`, av, metric: shown < total * 0.98 ? `first ${fmtTime(shown)} of ${fmtTime(total)}` : fmtTime(total), pad: false,
      body: `${canvas('4 / 5')}<div class="plot-axis"><span>Time runs down. Orange marks energy segments.</span></div>`,
      draw() { drawWaterfall($('canvas', this), wf, segs, fs); } }));

    const pts = vis.constellation.sampled_points;
    tiles.push(tile({ cat: 'iq', title: 'IQ scatter', src: `${src}, DC-centred, no timing or carrier correction`, av, metric: `${pts.length} points`, pad: false,
      body: canvas('1 / 1'), draw() { drawIQ($('canvas', this), pts); } }));

    const seg = vis.segmentation;
    tiles.push(tile({ cat: 'time', title: 'Energy segments', src: `${src}, ${seg.method.replaceAll('_', ' ')}`, av, metric: `${seg.segments.length} candidate${seg.segments.length === 1 ? '' : 's'}`,
      body: `<div class="timeline">${seg.segments.map(s => `<i style="left:${(s.sample_start / vis.input.sample_count * 100).toFixed(2)}%;width:${Math.max(0.4, s.sample_count / vis.input.sample_count * 100).toFixed(2)}%"></i>`).join('')}</div>
        ${seg.segments.length ? `<ul class="list">${seg.segments.map((s, i) => `<li><span>Segment ${i + 1} at ${fmtTime(s.sample_start / fs)}</span><span>${fmtTime(s.duration_seconds)}, ${s.sample_count.toLocaleString()} samples</span></li>`).join('')}</ul>` : '<p class="note">Nothing crossed the threshold. That alone doesn\'t mean there is no signal.</p>'}
        <p class="note">${esc(seg.limitation)}</p>` }));
  } else {
    tiles.push(tile({ cat: 'spectrum time iq', title: 'Plots unavailable', src: 'The API ran without NumPy', av: 'est', body: '<p class="note" style="margin:0">Install NumPy on the machine running the API (<code>pip install -r requirements-dsp.txt</code>) to get FFT, waterfall, scatter and segment evidence.</p>' }));
  }

  /* Classifier */
  const cls = a.modulation_classification; const ranked = cls.ranked_candidates || []; const dmin = Math.min(...ranked.map(r => r.distance));
  const pred = cls.predicted_modulation; const noRx = !cls.abstained && pred && !RX_SUPPORTED.includes(pred);
  tiles.push(tile({ cat: 'model', title: 'Modulation classifier', src: 'Synthetic-trained feature centroids, raw input', av: 'model', metric: cls.abstained ? 'abstained' : `${pct(cls.confidence)} confidence`,
    body: `<div class="big">${cls.abstained ? 'Abstained' : esc(String(pred).toUpperCase())}${cls.abstained ? '' : `<small>${pct(cls.confidence)}</small>`}</div>
      <p class="note" style="margin-top:6px">${esc(cls.reason)}${noRx ? ` There is no ${esc(pred.toUpperCase())} receiver in this build.` : ''}</p>
      <div class="rows">${ranked.map((r, i) => `<div class="row${i === 0 && !cls.abstained ? ' lead' : ''}"><span>${esc(r.modulation.toUpperCase())}</span><span class="track"><span class="fill" style="width:${(r.distance ? Math.min(1, dmin / r.distance) * 100 : 100).toFixed(1)}%"></span></span><span>${r.distance.toFixed(2)}</span></div>`).join('')}</div>
      <p class="note">Bars show closeness to each class centroid, numbers are distance (lower is closer). ${esc(cls.model?.scope || '')}</p>` }));

  /* Manual versus learned */
  const cmp = a.automated_parameter_comparison;
  const nyq = fs / 2; const label = { dc_i: 'DC, I', dc_q: 'DC, Q', carrier_offset_hz: 'Carrier offset' };
  const fmtVal = (k, v) => (k === 'carrier_offset_hz' ? fmtHz(v, true) : fmtNum(v, 5));
  const rows = Object.entries(cmp).map(([k, v]) => {
    const flags = [v.agreement >= 1 ? '<span class="flag ok">Agree</span>' : '<span class="flag warn">Disagree</span>'];
    if (k === 'carrier_offset_hz' && Math.abs(v.learned) > nyq) flags.push(`<span class="flag warn">Model value is outside ±${fmtHz(nyq)}, impossible at this rate</span>`);
    return `<tr><td>${esc(label[k] || k)}<br>${flags.join(' ')}</td><td>${fmtVal(k, v.dsp)}</td><td>${fmtVal(k, v.learned)}</td></tr>`;
  }).join('');
  const disagreements = Object.values(cmp).filter(v => v.agreement < 1).length;
  tiles.push(tile({ cat: 'model estimates', title: 'Manual versus model', src: 'Named DSP estimators next to the synthetic-trained MLP, raw input', av: 'model', metric: disagreements ? `${disagreements} disagreement${disagreements > 1 ? 's' : ''}` : 'all agree',
    body: `<table class="cmp"><thead><tr><th>Parameter</th><th>Manual</th><th>Model</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="note">The model doesn't estimate ${esc((a.provenance.not_supported_by_model || []).join(', ').replaceAll('_', ' '))}. Where they disagree, trust neither until a reference confirms one.</p>` }));

  /* Manual estimates for the chosen branch */
  const mp = B.manual_parameters; const cfo = mp.coarse_carrier_offset_hz; const bw = mp.spectrum;
  tiles.push(tile({ cat: 'estimates', title: 'Manual estimates', src: `${src}, ${mp.method_version.replaceAll('_', ' ')}`, av: 'est', metric: `CFO ${fmtHz(cfo.carrier_offset_hz, true)}`,
    body: `<dl class="kv">
        <dt>Coarse carrier offset</dt><dd>${fmtHz(cfo.carrier_offset_hz, true)}</dd>
        <dt>DC offset</dt><dd>I ${fmtNum(mp.dc_iq.i, 5)}, Q ${fmtNum(mp.dc_iq.q, 5)}</dd>
        <dt>Occupied bandwidth, 99%</dt><dd>${fmtHz(bw?.occupied_bandwidth_99pct_hz)}</dd>
        <dt>Peak to median power</dt><dd>${fmtNum(mp.power.peak_to_median_db, 2)} dB</dd>
      </dl>
      <p class="note">Carrier offset method: ${esc(cfo.method.replaceAll('_', ' '))}. Valid for ${esc(cfo.validity)}.</p>
      <div class="chips"><button class="chip" data-set="p-cfo" data-value="${cfo.carrier_offset_hz.toFixed(1)}">Use ${fmtHz(cfo.carrier_offset_hz, true)} in the receiver</button></div>
      ${(mp.symbol_rate_candidates || []).length ? `<p class="note">Symbol-rate candidates from transition periodicity. These are guesses to try, not a recovered clock.</p>
      <div class="chips">${mp.symbol_rate_candidates.map(c => `<button class="chip" data-set="p-sps" data-value="${c.samples_per_symbol_candidate}" title="Correlation ${c.normalized_transition_correlation}">${c.samples_per_symbol_candidate} sps, ${fmtHz(c.symbol_rate_baud_candidate).replace('Hz', 'Bd')}</button>`).join('')}</div>` : ''}` }));

  /* Features */
  const ft = B.features; const hyp = ft.modulation_hypotheses || []; const hmax = Math.max(...hyp.map(x => x.score), 1e-9);
  tiles.push(tile({ cat: 'iq estimates', title: 'Signal features', src, av, metric: `crest ${fmtNum(ft.crest_factor, 2)}`,
    body: `<dl class="kv"><dt>RMS</dt><dd>${fmtNum(ft.rms, 5)}</dd><dt>Peak</dt><dd>${fmtNum(ft.peak, 5)}</dd><dt>Crest factor</dt><dd>${fmtNum(ft.crest_factor, 3)}</dd><dt>Amplitude variation</dt><dd>${fmtNum(ft.amplitude_cv, 3)}</dd><dt>Instantaneous frequency</dt><dd>${fmtHz(ft.instantaneous_frequency_hz.mean, true)} mean, ${fmtHz(ft.instantaneous_frequency_hz.std)} spread</dd></dl>
      ${hyp.length ? `<div class="rows">${hyp.map(x => `<div class="row"><span>${esc(x.family.replaceAll('_', ' ').replace('ofdm or multicarrier', 'OFDM').toUpperCase())}</span><span class="track"><span class="fill" style="width:${(x.score / hmax * 100).toFixed(1)}%"></span></span><span>${x.score.toFixed(2)}</span></div>`).join('')}</div>
      <p class="note">Family scores from a hand-written feature heuristic. Separate from the classifier, and often weaker.</p>` : ''}` }));

  /* Cleaning audit */
  const dn = a.analysis.denoising; const rf = a.analysis.raw_branch.features; const df = a.analysis.derived_branch.features;
  const act = dn.actions.map(x => {
    let detail = x.applied ? 'Applied' : `Off${x.reason ? `, ${x.reason.replaceAll('_', ' ')}` : ''}`;
    if (x.applied && x.removed) detail = `Removed I ${fmtNum(x.removed.i, 5)}, Q ${fmtNum(x.removed.q, 5)}`;
    if (x.applied && x.masked_count != null) detail = `${x.masked_count} samples masked`;
    return `<li><span>${esc(x.stage.replaceAll('_', ' '))}</span><span>${esc(detail)}</span></li>`;
  }).join('');
  tiles.push(tile({ cat: 'estimates', title: 'Raw and derived', src: dn.raw_preserved ? 'Raw samples kept. Derived branch is separate.' : 'Raw preservation not confirmed', av: 'derived', metric: DENOISE_LABEL[a.provenance.denoising_profile] || '',
    body: `<ul class="list">${act}</ul>
      <table class="cmp" style="margin-top:14px"><thead><tr><th></th><th>Raw</th><th>Derived</th></tr></thead><tbody>
      <tr><td>RMS</td><td>${fmtNum(rf.rms, 5)}</td><td>${fmtNum(df.rms, 5)}</td></tr>
      <tr><td>DC, I</td><td>${fmtNum(rf.dc_offset.i, 5)}</td><td>${fmtNum(df.dc_offset.i, 5)}</td></tr>
      <tr><td>DC, Q</td><td>${fmtNum(rf.dc_offset.q, 5)}</td><td>${fmtNum(df.dc_offset.q, 5)}</td></tr>
      <tr><td>Crest factor</td><td>${fmtNum(rf.crest_factor, 3)}</td><td>${fmtNum(df.crest_factor, 3)}</td></tr></tbody></table>
      <p class="note">${esc(a.analysis.comparison_rule)}</p>` }));

  /* Limits */
  const limits = [...(a.manual_dsp.limitations || []), ...(a.manual_parameter_estimation.limitations || []), a.provenance.scope].filter(Boolean);
  tiles.push(tile({ cat: 'model estimates', title: 'What this run can\'t tell you', src: 'Stated by the API', av: 'est', body: `<ul class="limits">${limits.map(l => `<li>${esc(l)}</li>`).join('')}</ul>` }));

  setTiles('feed', tiles);
}

$('#filters').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; state.filter = b.dataset.filter; $$('#filters button').forEach(x => x.setAttribute('aria-selected', String(x === b))); layout('feed'); });
$('#branch').addEventListener('click', e => { const b = e.target.closest('button'); if (!b || b.disabled) return; state.branch = b.dataset.value; renderEvidence(); });
$('#p-denoise').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setDenoise(b.dataset.value); });
document.addEventListener('click', e => {
  const chip = e.target.closest('[data-set]'); if (!chip) return;
  const target = document.getElementById(chip.dataset.set); target.value = chip.dataset.value; target.dispatchEvent(new Event('change'));
  chip.textContent = 'Set in receiver';
  setTimeout(() => renderEvidence(), 900);
});

/* ---------- receiver ---------- */
function renderRxHints() {
  const box = $('#rx-hints'); const a = state.analysis;
  if (!a) { box.innerHTML = ''; return; }
  const cls = a.modulation_classification; const mp = a.manual_parameter_estimation; const chips = [];
  if (!cls.abstained && RX_SUPPORTED.includes(cls.predicted_modulation)) chips.push(`<button class="chip" data-set="p-mod" data-value="${cls.predicted_modulation}">Classifier says ${cls.predicted_modulation.toUpperCase()}</button>`);
  chips.push(`<button class="chip" data-set="p-cfo" data-value="${mp.coarse_carrier_offset_hz.carrier_offset_hz.toFixed(1)}">CFO estimate ${fmtHz(mp.coarse_carrier_offset_hz.carrier_offset_hz, true)}</button>`);
  (mp.symbol_rate_candidates || []).slice(0, 4).forEach(c => chips.push(`<button class="chip" data-set="p-sps" data-value="${c.samples_per_symbol_candidate}">${c.samples_per_symbol_candidate} sps</button>`));
  box.innerHTML = `<span>From the analysis:</span>${chips.join('')}`;
}

function renderReceiver() {
  renderRxHints();
  const d = state.demod; show($('#rx-empty'), !d);
  $('#rx-empty-text').textContent = state.bytes ? 'Set the receiver and run it. Start from the analysis hints above if you have them.' : 'Load a capture first. The receiver uses the same bytes and interpretation.';
  if (!d) { setTiles('rx-feed', []); return; }
  const tiles = [];
  if (d.status === 'abstained') {
    const c = d.classification;
    tiles.push(tile({ title: 'No receiver selected', src: 'Automatic modulation choice', av: 'model', metric: 'abstained',
      body: `<div class="big">Abstained</div><p class="note">${esc(d.reason)}</p><p class="note">Classifier: ${c.abstained ? 'abstained' : `${esc(String(c.predicted_modulation).toUpperCase())} at ${pct(c.confidence)}`}. Pick a modulation above to run it anyway.</p>` }));
    setTiles('rx-feed', tiles); return;
  }
  const cfg = d.configuration; const mod = d.modulation; const fs = state.demodSettings?.sample_rate_hz;
  const dec = d.decisions_preview || [];
  if (mod === '2fsk') {
    const vals = dec.map(x => x.frequency_step_rad);
    tiles.push(tile({ title: 'Discriminator output', src: 'Phase step per symbol, first 256 symbols', av: 'rx', metric: '2-FSK', pad: false, body: canvas('16 / 9'), draw() { drawSeries($('canvas', this), vals); } }));
  } else {
    const rms = Math.sqrt(dec.reduce((s, x) => s + x.i * x.i + x.q * x.q, 0) / Math.max(1, dec.length)) || 1;
    const pts = dec.map(x => ({ i: x.i / rms, q: x.q / rms }));
    const ideal = mod === 'bpsk' ? [{ i: -1, q: 0 }, { i: 1, q: 0 }] : [-1, 1].flatMap(i => [-1, 1].map(q => ({ i: i * Math.SQRT1_2, q: q * Math.SQRT1_2 })));
    tiles.push(tile({ title: 'Symbol decisions', src: `First ${dec.length} symbols, RMS-normalised. Rings mark ideal points.`, av: 'rx', metric: mod.toUpperCase(), pad: false, body: canvas('1 / 1'), draw() { drawIQ($('canvas', this), pts, ideal); } }));
  }
  const segs = state.analysis?.analysis.raw_branch.visualization?.segmentation.segments || [];
  const evm = d.quality.evm_rms;
  tiles.push(tile({ title: 'Quality', src: 'Error vector magnitude after fixed integrate-and-dump', av: 'rx', metric: `${d.symbol_count.toLocaleString()} symbols`,
    body: `<div class="big">${evm == null ? 'n/a' : pct(evm)}<small>${evm == null ? 'EVM not defined for FSK' : 'EVM RMS'}</small></div>
      <p class="note">${esc(d.quality.meaning)}${segs.length > 0 && evm != null ? ' It covers the whole file, so quiet stretches between bursts push it up.' : ''}</p>
      <dl class="kv" style="margin-top:12px"><dt>Samples per symbol</dt><dd>${cfg.samples_per_symbol}</dd><dt>Timing offset</dt><dd>${cfg.timing_offset_samples} samples</dd><dt>Carrier offset removed</dt><dd>${fmtHz(cfg.carrier_offset_hz, true)}</dd>${fs ? `<dt>Implied symbol rate</dt><dd>${fmtHz(fs / cfg.samples_per_symbol).replace('Hz', 'Bd')}</dd>` : ''}<dt>Settings from</dt><dd>${esc(SOURCE_LABEL[cfg.parameter_source] || cfg.parameter_source)}</dd></dl>` }));

  const bits = d.bits_preview || ''; const grouped = bits.match(/.{1,8}/g) || [];
  tiles.push(tile({ title: 'Candidate hard decisions', src: 'Unframed, unchecked, not decoded data', av: 'rx', metric: `${bits.length} of ${d.bit_count.toLocaleString()} bits`,
    body: `<p class="bits">${grouped.map((g, i) => (i % 2 ? `<b>${g}</b>` : g)).join(' ')}</p>
      <p class="note">Packed into bytes, first ${d.packed_bytes_preview.length}:</p>
      <p class="bits">${d.packed_bytes_preview.map(b => b.toString(16).padStart(2, '0')).join(' ')}</p>
      <p class="note">No framing, CRC, FEC or descrambling ran. Phase ambiguity is unresolved, so the whole stream may be rotated or inverted.</p>` }));

  const g = d.gnu_radio_graph;
  if (g) tiles.push(tile({ title: 'GNU Radio flowgraph', src: g.engine.installed ? 'GNU Radio found on the API machine' : 'Descriptor only. GNU Radio isn\'t installed where the API runs.', av: 'est', metric: `${g.blocks.length} blocks`,
    body: `<div class="chain">${g.blocks.map((b, i) => `<div class="block"><i>${i + 1}</i><span>${esc(b.block)}<small>${esc(b.role)}${b.samples_per_symbol ? `, ${b.samples_per_symbol} sps` : ''}${b.frequency_offset_hz ? `, ${fmtHz(b.frequency_offset_hz, true)}` : ''}${b.constellation ? `, ${esc(b.constellation)}` : ''}</small></span></div>`).join('')}</div><p class="note">${esc(g.note)}</p>` }));

  tiles.push(tile({ title: 'Receiver limits', src: 'Stated by the API', av: 'est', body: `<ul class="limits">${(d.limitations || []).map(l => `<li>${esc(l)}</li>`).join('')}</ul>` }));
  setTiles('rx-feed', tiles);
}

/* ---------- report ---------- */
function renderReport() {
  const a = state.analysis, d = state.demod; const rows = [];
  const add = (k, v) => rows.push(`<dt>${esc(k)}</dt><dd>${v}</dd>`);
  if (state.file) {
    add('File', `${esc(state.file.name)}, ${fmtBytes(state.file.size)}${state.file.generated ? ', generated in this browser' : ''}${state.example ? ', example capture' : ''}`);
    add('Bytes sent', `${state.bytes.byteLength.toLocaleString()}${state.truncated ? ' (first slice only)' : ''}`);
    add('SHA-256, browser', state.shaBrowser ? `<code>${state.shaBrowser}</code>` : 'Unavailable on this page (needs a secure context)');
  }
  if (state.file?.truth) { const t = state.file.truth; add('Known truth', `${esc(t.modulation.toUpperCase())}, ${t.samples_per_symbol} samples per symbol, ${fmtHz(t.carrier_offset_hz, true)} carrier offset, ${fmtRate(t.sample_rate_hz)}, ${t.bursts} bursts, noise σ ${t.noise_std}`); }
  if (a) {
    const inp = a.input; const cap = inp.capture;
    const match = state.shaBrowser ? (state.shaBrowser === inp.sha256 ? ' <span class="flag ok">Matches the browser hash</span>' : ' <span class="flag warn">Differs from the browser hash</span>') : '';
    add('Run', `<code>${esc(a.run_id)}</code>`);
    add('SHA-256, API', `<code>${esc(inp.sha256)}</code>${match}`);
    add('IQ format', `${esc(inp.representation.iq_format)}, ${esc(SOURCE_LABEL[inp.representation.iq_format_source] || inp.representation.iq_format_source)}`);
    add('Sample rate', `${fmtRate(cap.sample_rate_hz)}, ${esc(SOURCE_LABEL[cap.sample_rate_source] || cap.sample_rate_source)}`);
    add('Centre frequency', cap.centre_frequency_hz != null ? `${fmtHz(cap.centre_frequency_hz)}, ${esc(SOURCE_LABEL[cap.centre_frequency_source] || cap.centre_frequency_source)}` : 'Not supplied');
    add('Gain', cap.gain_db != null ? `${cap.gain_db} dB, ${esc(SOURCE_LABEL[cap.gain_source] || cap.gain_source)}` : 'Not supplied');
    add('Raw data kept by the API', inp.raw_data_persisted || a.provenance.raw_data_persisted ? 'Yes' : 'No');
    add('Cleaning', esc(DENOISE_LABEL[a.provenance.denoising_profile] || a.provenance.denoising_profile));
    add('Manual branch', esc(a.provenance.manual_branch));
    add('Model branch', `${esc(a.provenance.automated_branch)}. ${a.provenance.training_examples} examples from <code>${esc(a.provenance.model_training_recipes)}</code>.`);
    add('Model scope', esc(a.provenance.scope));
  }
  if (d && d.status !== 'abstained') {
    const c = d.configuration;
    add('Receiver', `${esc(d.modulation.toUpperCase())}, ${c.samples_per_symbol} sps, offset ${c.timing_offset_samples}, ${fmtHz(c.carrier_offset_hz, true)} removed. Settings from ${esc(SOURCE_LABEL[c.parameter_source] || c.parameter_source)}.`);
    add('Receiver output', `${d.bit_count.toLocaleString()} candidate bits, not decoded data`);
  } else if (d) add('Receiver', `Abstained. ${esc(d.reason)}`);
  if (state.example) add('Recorded from', esc(state.example.recorded_from));
  $('#report').innerHTML = rows.join('') || '';
  $('#report-note').textContent = a || d ? `Export includes the full API responses${state.example ? ' (recorded example)' : ''}.` : 'No run yet.';
  $('#raw-json').textContent = a || d ? JSON.stringify({ analyse: a ? { ...a, analysis: '[plot arrays omitted here, included in the export]' } : null, demodulate: d }, null, 2) : 'None yet.';
  updateButtons();
}
$('#export').addEventListener('click', () => {
  const report = {
    report_type: 'demod_analyst_session', ui_version: UI_VERSION, exported_at: new Date().toISOString(),
    client: { file_name: state.file?.name, file_size: state.file?.size, bytes_sent: state.bytes?.byteLength, truncated_to_api_limit: state.truncated, sha256_browser: state.shaBrowser, generated_in_browser: Boolean(state.file?.generated), known_truth: state.file?.truth || null, api_url: apiBase(), example: state.example },
    settings: { analyse: state.analysisSettings, demodulate: state.demodSettings },
    analyse: state.analysis, demodulate: state.demod,
    labels: { candidate_bits: 'hard decisions from a fixed receiver; not decoded, validated or decrypted data' },
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: `demod-${(state.analysis?.run_id || 'session').slice(0, 8)}.json` });
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

/* ---------- boot ---------- */
function renderAll() { renderEvidence(); renderReceiver(); renderReport(); }
apiInput.value = defaultApi();
apiInput.addEventListener('change', checkApi);
$('#api-check').addEventListener('click', checkApi);
setDenoise('raw');
syncBound(); renderAll(); updateSummary();
go(location.hash.slice(1) || 'capture');
checkApi();
