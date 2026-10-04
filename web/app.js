'use strict';

/* DEmod analyst UI. Talks only to the local comparison API the analyst points it at.
   Every number shown is either returned by that API or labelled as a browser-side check. */

const UI_VERSION = '0.6.0';
const MAX_BYTES = 16 * 1024 * 1024;
const SAMPLE_BYTES = { s8: 2, cu8: 2, s16le: 4, s16be: 4, f32le: 8, f32be: 8 };
const SOURCE_LABEL = { analyst_hypothesis: 'My hypothesis', sigmf_metadata: 'SigMF sidecar', wav_header: 'WAV header', analyst_wav_interpretation: 'Declared WAV interpretation', capture_log: 'Capture log', unavailable: 'Not supplied', automatic_classifier: 'Classifier', analyst_override: 'Analyst setting' };
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
  wavHeader: null, isWav: false, audioUrl: null,
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
  const duration = parseNum($('#p-duration').value);
  if (duration > 0) h['X-DEmod-Recording-Duration'] = String(duration);
  if (state.isWav) { h['X-DEmod-WAV-Role'] = s.wav_role; h['X-DEmod-IF-Centre'] = String(s.if_centre_hz); }
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
const F = { format: $('#p-format'), rate: $('#p-rate'), centre: $('#p-centre'), gain: $('#p-gain'), source: $('#p-source'), mod: $('#p-mod'), sps: $('#p-sps'), timing: $('#p-timing'), cfo: $('#p-cfo'), wav: $('#p-wav-role'), ifcentre: $('#p-if-centre') };
let sigmfLoaded = false;

function settings() {
  return {
    iq_format: F.format.value, sample_rate_hz: parseNum(F.rate.value), centre_frequency_hz: parseNum(F.centre.value),
    gain_db: parseNum(F.gain.value), metadata_source: F.source.value, denoise_profile: state.denoise,
    modulation: F.mod.value, samples_per_symbol: parseNum(F.sps.value), timing_offset: parseNum(F.timing.value) ?? 0, carrier_offset_hz: parseNum(F.cfo.value) ?? 0,
    wav_role: F.wav.value, if_centre_hz: parseNum(F.ifcentre.value) ?? 0,
  };
}
function captureProblems(s) {
  const p = [];
  if (state.isWav && s.wav_role === 'unspecified') p.push('Choose whether this WAV contains stereo I/Q, a mono RF/IF waveform, or audio.');
  if (state.isWav && state.wavHeader && s.sample_rate_hz !== state.wavHeader.sample_rate_hz) p.push('Use the WAV header sample rate; resampling is a separate operation.');
  if (state.isWav && (!Number.isFinite(s.if_centre_hz) || Math.abs(s.if_centre_hz) >= s.sample_rate_hz / 2)) p.push('Recorded IF centre must lie inside the sampled Nyquist interval.');
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
  const size = state.bytes.byteLength; const per = SAMPLE_BYTES[s.iq_format]; const n = state.isWav ? (state.wavHeader?.frame_count || 0) : Math.floor(size / per);
  let text = state.isWav ? `${state.file.name}, ${fmtBytes(state.file.size)}. WAV header: ${n.toLocaleString()} frames, ${state.wavHeader?.channels || '?'} channels` : `${state.file.name}, ${fmtBytes(state.file.size)}. Read as ${s.iq_format}, that is ${n.toLocaleString()} complex samples`;
  text += s.sample_rate_hz > 0 ? `, or ${fmtTime(n / s.sample_rate_hz)} at ${fmtRate(s.sample_rate_hz)}.` : '.';
  if (!state.isWav && size % per) text += ` ${size % per} trailing bytes don't fill a sample, so the format may be wrong.`;
  if (state.truncated) text += ` Only the first ${fmtBytes(size)} will be sent, and the hash covers that slice only.`;
  if (state.analysisSettings && JSON.stringify(pick(state.analysisSettings)) !== JSON.stringify(pick(s))) text += ' Settings changed since the last run, so analyse again to update the evidence.';
  out.textContent = text;
  updateButtons();
}
const pick = s => [s.iq_format, s.sample_rate_hz, s.centre_frequency_hz, s.gain_db, s.metadata_source, s.denoise_profile, s.wav_role, s.if_centre_hz];

function updateButtons() {
  const s = settings();
  $('#run-analysis').disabled = !state.bytes || captureProblems(s).length > 0 || !state.api.ok;
  $('#run-analysis').title = !state.api.ok ? 'Connect the local API first' : '';
  $('#run-demod').disabled = !state.bytes || !(s.sample_rate_hz > 0) || !state.api.ok;
  const audio = state.isWav && s.wav_role === 'audio';
  $('#run-demod').disabled ||= audio;
  $('#run-auto-demod').disabled = !state.bytes || !state.api.ok || captureProblems(s).length > 0 || audio;
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
  state.wavHeader = null; state.isWav = false; F.rate.readOnly = false; F.format.disabled = false;
  show($('#wav-options'), false);
  if (state.audioUrl) { URL.revokeObjectURL(state.audioUrl); state.audioUrl = null; }
  $('#wav-player').removeAttribute('src');
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
  const magic = new Uint8Array(await file.slice(0,12).arrayBuffer());
  const wav = String.fromCharCode(...magic.slice(0,4)) === 'RIFF' && String.fromCharCode(...magic.slice(8,12)) === 'WAVE';
  if (file.name.toLowerCase().endsWith('.wav') && !wav) { flash('#capture-error', 'This file is not a supported RIFF/WAVE container.'); return; }
  if (wav && file.size > MAX_BYTES) { flash('#capture-error', `WAV files must fit the ${fmtBytes(MAX_BYTES)} demo limit; truncating their headers/payload would invalidate them.`); return; }
  state.example = null;
  $('#p-duration').value = ''; $('#rate-evidence').textContent = '';
  const buffer = await file.slice(0, Math.min(file.size, MAX_BYTES)).arrayBuffer();
  await loadBytes(file, buffer);
  state.isWav = wav;
  show($('#wav-options'), wav); F.wav.value = 'unspecified'; F.ifcentre.value = '0';
  if (wav) { state.audioUrl = URL.createObjectURL(new Blob([buffer], {type:'audio/wav'})); $('#wav-player').src = state.audioUrl; F.format.disabled = true; }
  if (wav && state.api.ok) {
    try {
      const r = await post('/rates', { 'Content-Type': 'application/octet-stream' });
      state.wavHeader = r.metadata;
      F.rate.value = r.metadata.sample_rate_hz; F.rate.readOnly = true;
      $('#wav-header-info').textContent = `${r.metadata.encoding}, ${r.metadata.bits_per_channel_sample} bits/channel sample, ${r.metadata.channels} channels, ${r.metadata.frame_count.toLocaleString()} frames, ${fmtRate(r.metadata.sample_rate_hz)}, ${fmtTime(r.metadata.duration_seconds)}. Header does not establish channel roles, RF centre or gain.`;
      $('#rate-evidence').textContent = 'Sample rate read from WAV header. Choose its waveform interpretation to analyze it.';
      updateSummary();
    } catch (err) { flash('#capture-error', err.message); }
  }
}
$('#file-input').addEventListener('change', e => onFile(e.target.files[0]));
$('#open-wav-example').addEventListener('click', async () => {
  try {
    const res = await fetch('examples/synthetic-qpsk-burst.wav');
    if (!res.ok) throw new Error('WAV example is unavailable.');
    await onFile(new File([await res.blob()], 'synthetic-qpsk-burst.wav', {type:'audio/wav'}));
    F.wav.value = 'stereo_iq'; F.mod.value = 'auto'; F.sps.value = ''; F.cfo.value = '0';
    $('#drop-detail').textContent = 'Synthetic QPSK demo wrapped in PCM WAV. Known channel order: channel 1 = I, channel 2 = Q.';
    updateSummary(); syncBound();
  } catch (err) { flash('#capture-error', err.message); }
});
$('#estimate-rates').addEventListener('click', async () => {
  if (!state.bytes) { $('#rate-evidence').textContent = 'Load a capture first.'; return; }
  try {
    const h = { 'Content-Type': 'application/octet-stream', 'X-DEmod-IQ-Format': F.format.value };
    const fs = parseNum(F.rate.value); if (fs > 0) h['X-DEmod-Sample-Rate'] = String(fs);
    if (state.isWav) { h['X-DEmod-WAV-Role'] = F.wav.value; h['X-DEmod-IF-Centre'] = String(parseNum(F.ifcentre.value) ?? 0); }
    const duration = parseNum($('#p-duration').value);
    if ($('#p-duration').value.trim() && !(duration > 0)) throw new Error('Recording duration must be positive.');
    if (duration > 0) h['X-DEmod-Recording-Duration'] = String(duration);
    const r = await post('/rates', h);
    if (r.metadata) {
      state.wavHeader = r.metadata; F.rate.value = r.metadata.sample_rate_hz; F.rate.readOnly = true;
      $('#rate-evidence').textContent = `Fs ${fmtRate(r.metadata.sample_rate_hz)} from WAV header; ${r.metadata.channels} channels.${r.rates?.learned ? ' Model SPS: ' + (r.rates.learned.abstained ? 'abstained' : r.rates.learned.samples_per_symbol) + '.' : ' Choose the waveform interpretation for RF rate estimates.'}`;
      updateSummary();
    } else {
      if (r.rates.sample_rate_source === 'sample_count / analyst_recording_duration') {
        F.rate.value = r.rates.absolute_sample_rate_hz; updateSummary();
      }
      const learned = r.rates.learned;
      const manual = r.rates.manual.candidates.slice(0, 3).map(c => `${fmtNum(c.samples_per_symbol, 2)} samples/symbol`).join(', ');
      $('#rate-evidence').textContent = `${r.rates.absolute_sample_rate_hz ? 'Fs ' + fmtRate(r.rates.absolute_sample_rate_hz) + ' (' + r.rates.sample_rate_source + '). ' : ''}DSP candidates: ${manual}. Model: ${learned && !learned.abstained ? learned.samples_per_symbol + ' samples/symbol' : 'abstained or unavailable'}. Absolute Fs needs metadata or a time reference. Model scope: simulated PSK/QAM, SPS 2/4/8/16.`;
    }
  } catch (err) { $('#rate-evidence').textContent = err.message; }
});
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

$('#run-auto-demod').addEventListener('click', () => busy($('#run-auto-demod'), 'Estimating and receiving', async () => {
  const s = settings(); const problems = captureProblems(s);
  if (!state.bytes) problems.push('Load a capture first.');
  if (problems.length) { flash('#rx-error', problems.join(' ')); return; }
  flash('#rx-error', '');
  try {
    state.demod = await post('/demodulate', { ...captureHeaders(s), 'X-DEmod-Modulation': 'auto' });
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
  const inst = box.appendChild(el('<div class="instruments"></div>'));
  const mas = box.appendChild(el('<div class="masonry"></div>'));
  const cols = Array.from({ length: n }, () => mas.appendChild(el('<div class="col"></div>')));
  feeds[id].forEach(t => {
    const visible = id !== 'feed' || state.filter === 'all' || t.dataset.cat.split(' ').includes(state.filter);
    t.hidden = !visible; if (!visible) return;
    if (t.classList.contains('instrument')) inst.appendChild(t);
    else cols.reduce((a, b) => (b.offsetHeight < a.offsetHeight ? b : a)).appendChild(t);
  });
  show(inst, inst.children.length > 0);
  feeds[id].forEach(t => { if (!t.hidden && t._draw) t._draw(); });
}
function layoutAll() { layout('feed'); layout('rx-feed'); }
let resizeTimer; window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(layoutAll, 120); });
matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', layoutAll);

/* ---------- instruments (GNU Radio style sinks, see plots.js) ---------- */
/* legend: [{label, trace, on}], controls: extra HTML, render(canvas, legendState) */
function instrument({ cat = '', title, src, av = 'raw', metric = '', wide = false, ratio = '16 / 9', legend = [], controls = '', note = '', render }) {
  const legendHtml = legend.map((l, i) => `<button type="button" class="lg" data-i="${i}" aria-pressed="${l.on !== false}"><i style="background:var(--trace-${l.trace})${l.dash ? ';height:0;border-top:2px dashed var(--trace-' + l.trace + ');background:none' : ''}"></i>${esc(l.label)}</button>`).join('');
  const node = tile({ cat, title, src, av, metric, pad: false,
    body: `<div class="scope"><canvas class="plot" style="aspect-ratio:${ratio}"></canvas><div class="scope-bar"><div class="legend">${legendHtml}</div><div class="scope-ctl">${controls}</div></div>${note ? `<p class="scope-note">${note}</p>` : ''}</div>` });
  node.classList.add('instrument'); if (wide) node.classList.add('wide');
  const st = { on: legend.map(l => l.on !== false) };
  node._st = st;
  node._draw = () => render($('canvas', node), st, node);
  node.addEventListener('click', e => {
    const b = e.target.closest('.lg'); if (!b) return;
    const i = Number(b.dataset.i); st.on[i] = !st.on[i]; b.setAttribute('aria-pressed', String(st.on[i])); node._draw();
  });
  return node;
}
const tip = $('#tip');
const tr = n => getComputedStyle(document.documentElement).getPropertyValue(`--trace-${n}`).trim();
function nearestIndex(arr, v) {
  let lo = 0, hi = arr.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (arr[mid] < v) lo = mid; else hi = mid; }
  return Math.abs(arr[lo] - v) <= Math.abs(arr[hi] - v) ? lo : hi;
}
function percentile(values, q) { const s = [...values].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.max(0, Math.round((s.length - 1) * q)))]; }
const fmtF = v => { const [u, k] = Plot.unitFor('Hz', v, v); return Plot.fmtVal(v, u, k, 9); };
const fmtFr = (v, res) => { const [u, k] = Plot.unitFor('Hz', v, v); const d = Math.max(0, Math.ceil(-Math.log10(res / k) - 1e-9)); return `${(v / k).toFixed(d).replace('-', '−')} ${u}`; };
const fmtT = v => { const [u, k] = Plot.unitFor('s', v, v); return Plot.fmtVal(v, u, k, 5); };


/* Evidence instruments: frequency sink, waterfall sink, constellation sink, two time sinks. */
function instruments(vis, fs, centre, src, av) {
  const out = []; const spec = vis.spectrum; const f = spec.frequency_hz;
  const hasWelch = Array.isArray(spec.welch_power_db);
  const traces = hasWelch
    ? [{ label: `Welch average, ${spec.welch_frames} frames`, y: spec.welch_power_db, trace: 0 }, { label: 'Max hold', y: spec.max_hold_db, trace: 1 }, { label: `Single FFT, first ${spec.fft_size} samples`, y: spec.power_db, trace: 2, on: false }]
    : [{ label: `Single FFT, first ${spec.fft_size} samples`, y: spec.power_db, trace: 0 }];
  const rfOk = centre != null;
  out.push(instrument({
    cat: 'spectrum', wide: true, ratio: '21 / 8', title: 'Frequency sink', av, src: `${src}, Hann ${spec.fft_size}-point${hasWelch ? ', 50% overlap' : ''}`,
    metric: `RBW ${fmtFr(fs / spec.fft_size * 1.5, 1)}`, legend: traces,
    controls: rfOk ? '<div class="seg small" role="radiogroup" aria-label="Frequency axis"><button type="button" role="radio" data-axis="bb" aria-checked="true">Baseband</button><button type="button" role="radio" data-axis="rf" aria-checked="false">RF</button></div>' : '<span class="muted">Baseband offset. Add a centre frequency for an RF axis.</span>',
    note: `${hasWelch ? 'dB relative to the peak of the Welch average. ' : 'dB relative to the FFT peak. '}RBW is the Hann equivalent noise bandwidth, 1.5 bins. Drag across the plot to zoom, double-click to reset.${rfOk ? ' The RF axis assumes the stated centre frequency is correct.' : ''}`,
    render(c, st, node) {
      const rf = rfOk && node.dataset.axis === 'rf'; const off = rf ? centre : 0;
      const shown = traces.filter((_, i) => st.on[i]); const all = (shown.length ? shown : traces).flatMap(t => t.y);
      const top = Math.max(...all); const bottom = Math.max(Math.min(...all), top - 110);
      Plot.line(c, {
        x: { label: rf ? 'Frequency' : 'Frequency offset', unit: 'Hz', domain: [f[0] + off, f[f.length - 1] + off] },
        y: { label: 'Relative power', unit: 'dB', domain: Plot.niceDomain(bottom, top, 0.04) }, zoomX: true,
        series: traces.map((t, i) => ({ x: f.map(v => v + off), y: t.y, color: tr(t.trace), hidden: !st.on[i], width: t.trace === 0 ? 1.4 : 1 })),
        readout: x => { const k = nearestIndex(f, x - off); return `${fmtFr(f[k] + off, fs / spec.fft_size)}  ${traces.filter((_, i) => st.on[i]).map(t => `${t.label.split(',')[0]} ${t.y[k].toFixed(1)} dB`).join('  ')}`; },
      });
    },
  }));
  const fNode = out[0];
  fNode.dataset.axis = 'bb';
  fNode.addEventListener('click', e => { const b = e.target.closest('button[data-axis]'); if (!b) return; fNode.dataset.axis = b.dataset.axis; $$('button[data-axis]', fNode).forEach(x => x.setAttribute('aria-checked', String(x === b))); const c = $('canvas', fNode); c._zoom = null; fNode._draw(); });

  /* Waterfall sink */
  const wf = vis.waterfall; const global = Array.isArray(wf.power_db_global) && wf.power_db_global.length;
  const rows = global ? wf.power_db_global : wf.power_db_relative;
  if (rows.length) {
    const half = wf.fft_size / 2 / fs; const t0 = wf.time_seconds[0] - half; const t1 = wf.time_seconds[wf.time_seconds.length - 1] + half;
    const total = vis.input.sample_count / fs; const wff = wf.frequency_hz; const df = (wff[wff.length - 1] - wff[0]) / (wff.length - 1);
    const flat = rows.flat(); const zlo = Math.floor(percentile(flat, 0.05) / 10) * 10; const zhi = 0;
    const marks = (vis.segmentation.segments || []).map(sg => [sg.sample_start / fs, (sg.sample_start + sg.sample_count) / fs]).filter(([x, y]) => y > t0 && x < t1).map(([x, y]) => [Math.max(x, t0), Math.min(y, t1)]);
    out.push(instrument({
      cat: 'spectrum time', ratio: '5 / 4', title: 'Waterfall sink', av, src: `${src}, STFT ${wf.fft_size}-point, hop ${wf.hop_size}`,
      metric: t1 < total * 0.98 ? `first ${fmtT(t1)} of ${fmtT(total)}` : fmtT(total),
      controls: `<label class="mini">Colour map <select data-cmap><option value="multi">Multi-colour</option><option value="whitehot">White hot</option><option value="blackhot">Black hot</option></select></label>`,
      note: `${global ? 'dB relative to the strongest bin across the frames shown, so quiet frames stay dark.' : 'Older API output: each frame is normalised to its own peak, so quiet frames look as bright as bursts.'} Time runs down. Red ticks on the left edge mark energy segments. Colour range autoscaled from the 5th percentile to the peak.`,
      render(c, st, node) {
        Plot.image(c, {
          x: { label: 'Frequency offset', unit: 'Hz', domain: [wff[0] - df / 2, wff[wff.length - 1] + df / 2] },
          y: { label: 'Time', unit: 's', domain: [t1, t0] }, z: { label: 'Power (dB rel.)', domain: [zlo, zhi] },
          rows, colormap: node.dataset.cmap || 'multi', marks,
          readout: (x, y) => { const k = nearestIndex(wff, x); const r = Math.max(0, Math.min(rows.length - 1, Math.floor((y - t0) / (t1 - t0) * rows.length))); return `${fmtFr(wff[k], df)}  t ${fmtT(y)}  ${rows[r][k].toFixed(1)} dB`; },
        });
      },
    }));
    const wNode = out[out.length - 1];
    wNode.addEventListener('change', e => { if (e.target.matches('[data-cmap]')) { wNode.dataset.cmap = e.target.value; wNode._draw(); } });
  }

  /* Constellation sink */
  const pts = vis.constellation.sampled_points;
  out.push(instrument({
    cat: 'iq', ratio: '5 / 4', title: 'Constellation sink', av, src: `${src}, every Nth sample, DC removed`, metric: `${pts.length} points`,
    legend: [{ label: 'Samples', trace: 0 }],
    note: 'Raw samples spread across the capture. No matched filter, timing or carrier correction, so a carrier offset smears PSK into a ring.',
    render(c, st) { Plot.scatter(c, { x: { label: 'In-phase' }, y: { label: 'Quadrature' }, points: st.on[0] ? pts : [] }); },
  }));

  /* Time sink: smoothed power envelope with the segmentation threshold */
  const env = vis.segmentation.envelope;
  if (env) {
    const segs = (vis.segmentation.segments || []).map(sg => [sg.sample_start / fs, (sg.sample_start + sg.sample_count) / fs]);
    const et = env.time_seconds;
    out.push(instrument({
      cat: 'time', wide: true, ratio: '21 / 7', title: 'Time sink, power envelope', av, src: `${src}, ${vis.segmentation.window_samples}-sample moving average, max-pooled`,
      metric: `${segs.length} segment${segs.length === 1 ? '' : 's'}`,
      legend: [{ label: 'Smoothed power', trace: 0 }, { label: 'Threshold', trace: 1, dash: true }, { label: 'Baseline', trace: 3, dash: true }],
      note: 'Shaded spans are the energy segments. Drag to zoom, double-click to reset.',
      render(c, st) {
        const ys = env.power_db; const lo = Math.max(Math.min(...ys), Math.max(...ys) - 80);
        const T = tr;
        Plot.line(c, {
          x: { label: 'Time', unit: 's', domain: [0, vis.input.sample_count / fs] },
          y: { label: 'Power', unit: 'dB', domain: Plot.niceDomain(Math.min(lo, env.baseline_db), Math.max(...ys, env.threshold_db), 0.05) }, zoomX: true,
          vspans: segs, series: [{ x: et, y: ys, color: T(0), hidden: !st.on[0] }],
          hlines: [st.on[1] && { y: env.threshold_db, color: T(1), label: 'threshold' }, st.on[2] && { y: env.baseline_db, color: T(3), dash: [2, 4], label: 'baseline' }].filter(Boolean),
          readout: (x, y) => { const k = nearestIndex(et, x); return `t ${fmtT(et[k])}  ${ys[k].toFixed(1)} dB`; },
        });
      },
    }));
  }

  /* Time sink: I and Q around the first burst */
  const tp = vis.time_preview;
  if (tp && tp.i.length) {
    const tt = tp.i.map((_, n) => (tp.start_sample + n) / fs);
    out.push(instrument({
      cat: 'time iq', wide: true, ratio: '21 / 7', title: 'Time sink, I and Q', av, src: `${src}, samples ${tp.start_sample.toLocaleString()} to ${(tp.start_sample + tp.sample_count - 1).toLocaleString()}, DC removed`,
      metric: fmtT(tp.sample_count / fs),
      legend: [{ label: 'Re (I)', trace: 0 }, { label: 'Im (Q)', trace: 1 }],
      note: `${esc(tp.placement.charAt(0).toUpperCase() + tp.placement.slice(1))}. Drag to zoom, double-click to reset.`,
      render(c, st) {
        const T = tr;
        const m = Math.max(...tp.i.map(Math.abs), ...tp.q.map(Math.abs)) || 1;
        Plot.line(c, {
          x: { label: 'Time', unit: 's', domain: [tt[0], tt[tt.length - 1]] }, y: { label: 'Amplitude', domain: Plot.niceDomain(-m, m, 0.05) }, zoomX: true,
          series: [{ x: tt, y: tp.i, color: T(0), hidden: !st.on[0], width: 1 }, { x: tt, y: tp.q, color: T(1), hidden: !st.on[1], width: 1 }],
          readout: x => { const k = nearestIndex(tt, x); return `n ${(tp.start_sample + k).toLocaleString()}  t ${fmtT(tt[k])}  I ${tp.i[k].toFixed(4)}  Q ${tp.q[k].toFixed(4)}`; },
        });
      },
    }));
  }
  return out;
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

  if (a.kind === 'audio') {
    state.branch = 'raw';
    $$('#branch button').forEach(b => { b.disabled = true; b.setAttribute('aria-checked', String(b.dataset.value === 'raw')); });
    $('#evidence-sub').textContent = `${a.audio_summary.sample_count.toLocaleString()} PCM audio frames. RF models are not applied to declared audio.`;
    const audioTiles = instruments(rawVis, fs, null, 'PCM audio, channel average for overview', 'raw').filter(t => !t.dataset.cat.includes('iq'));
    audioTiles.push(tile({title:'Audio overview',cat:'estimates',src:'WAV header and real-sample measurements',av:'raw',
      body:`<dl class="kv"><dt>Duration</dt><dd>${fmtTime(a.audio_summary.duration_seconds)}</dd><dt>RMS</dt><dd>${fmtNum(a.audio_summary.rms_level)}</dd><dt>Peak</dt><dd>${fmtNum(a.audio_summary.peak_level)}</dd></dl><p>Listen using the local player on Capture. This path analyzes already-demodulated audio; it cannot recover the original RF modulation.</p>`}));
    setTiles('feed',audioTiles); return;
  }

  const rawOnly = a.provenance.denoising_profile === 'raw';
  if (rawOnly) state.branch = 'raw';
  $$('#branch button').forEach(b => { b.setAttribute('aria-checked', String(b.dataset.value === state.branch)); b.disabled = rawOnly && b.dataset.value === 'derived'; b.title = b.disabled ? 'Cleaning was set to Raw, so there is no separate derived branch' : ''; });

  const B = branchData(); const vis = B.visualization; const av = state.branch === 'derived' ? 'derived' : 'raw'; const src = branchLabel();
  const tiles = [];
  if (inp.wav_header) tiles.push(tile({title:'WAV input and conversion',cat:'estimates',src:'Container facts plus declared waveform interpretation',av:'raw',
    body:`<p>${esc(inp.wav_header.encoding)}, ${inp.wav_header.bits_per_channel_sample} bits/channel sample; ${inp.wav_header.channels} channels; ${fmtRate(fs)} from WAV header.</p><p>${esc(inp.conversion.method)}. Original WAV SHA-256 is preserved. No resampling was performed.</p><p>Raw plots show the decoded/converted waveform before optional cleaning. Real-IF conversion is derived analytic I/Q, not independently measured quadrature.</p>`}));

  if (vis) {
    tiles.push(...instruments(vis, fs, centre, src, av));
    const seg = vis.segmentation;
    tiles.push(tile({ cat: 'time', title: 'Energy segments', src: `${src}, ${seg.method.replaceAll('_', ' ')}`, av, metric: `${seg.segments.length} candidate${seg.segments.length === 1 ? '' : 's'}`,
      body: seg.segments.length ? `<ul class="list">${seg.segments.map((sg, i) => `<li><span>Segment ${i + 1} at ${fmtTime(sg.sample_start / fs)}</span><span>${fmtTime(sg.duration_seconds)}, ${sg.sample_count.toLocaleString()} samples</span></li>`).join('')}</ul><p class="note">${esc(seg.limitation)}</p>` : `<p class="note" style="margin:0">Nothing crossed the threshold. That alone doesn't mean there is no signal. ${esc(seg.limitation)}</p>` }));
  } else {
    tiles.push(tile({ cat: 'spectrum time iq', title: 'Plots unavailable', src: 'The API ran without NumPy', av: 'est', body: '<p class="note" style="margin:0">Install NumPy on the machine running the API (<code>pip install -r requirements-dsp.txt</code>) to get FFT, waterfall, scatter and segment evidence.</p>' }));
  }

  /* Classifier: new DemodAMC returns probabilities, the legacy centroid returned distances. */
  const cls = a.modulation_classification; const ranked = cls.ranked_candidates || [];
  const isProb = ranked.length > 0 && ranked[0].probability != null;
  const dmin = isProb ? 0 : Math.min(...ranked.map(r => r.distance));
  const pred = cls.predicted_modulation; const noRx = !cls.abstained && pred && !RX_SUPPORTED.includes(pred);
  const label = m => (m === 'ofdm' ? 'OFDM' : m === 'am' ? 'AM' : m === 'fm' ? 'FM' : m === 'noise' ? 'Noise only' : m.toUpperCase());
  const legacy = cls.legacy_centroid;
  tiles.push(tile({ cat: 'model', title: 'Modulation classifier', av: 'model',
    src: isProb ? `DemodAMC, ${cls.chunks_used} chunks of 1,024 samples, carrier offset removed first` : 'Legacy feature centroids (new model files not found on the API machine)',
    metric: cls.abstained ? 'abstained' : `${pct(cls.confidence)} ${isProb ? 'probability' : 'margin'}`,
    body: `<div class="big">${cls.abstained ? 'Abstained' : esc(label(String(pred)))}${cls.abstained ? '' : `<small>${pct(cls.confidence)}</small>`}</div>
      <p class="note" style="margin-top:6px">${esc(cls.reason)}${noRx ? ` There is no ${esc(label(pred))} receiver in this build.` : ''}</p>
      <div class="rows">${ranked.map((r, i) => { const v = isProb ? r.probability : (r.distance ? Math.min(1, dmin / r.distance) : 1); return `<div class="row${i === 0 && !cls.abstained ? ' lead' : ''}"><span>${esc(label(r.modulation))}</span><span class="track"><span class="fill" style="width:${(v * 100).toFixed(1)}%"></span></span><span>${isProb ? pct(r.probability) : r.distance.toFixed(2)}</span></div>`; }).join('')}</div>
      <p class="note">${isProb ? (cls.model?.calibrated ? 'Probabilities from a temperature-scaled softmax fitted on held-out simulated data, averaged over chunks.' : 'Raw softmax probabilities, not calibrated, averaged over chunks.') : 'Bars show closeness to each class centroid, numbers are distance (lower is closer).'} ${esc(cls.model?.scope || '')}</p>
      ${legacy ? `<p class="note">The old centroid classifier said ${legacy.abstained ? 'nothing (abstained)' : esc(label(String(legacy.predicted_modulation)))} at ${pct(legacy.confidence)}.</p>` : ''}` }));

  /* Manual versus learned. Carrier offset now comes from SpecCFO and carries a confidence. */
  if (a.rate_estimation) {
    const r = a.rate_estimation; const model = r.learned;
    tiles.push(tile({ cat: 'model estimates', title: 'Symbol-rate evidence', av: 'model',
      src: 'Classical cyclic-line candidates versus trained feature softmax',
      metric: model && !model.abstained ? `${model.samples_per_symbol} samples/symbol` : 'Model abstained',
      body: `<p>Fs ${fmtRate(r.absolute_sample_rate_hz)} (${esc(r.sample_rate_source)}).</p><p>DSP candidates: ${r.manual.candidates.slice(0,3).map(c => esc(fmtNum(c.samples_per_symbol,2))).join(', ')} samples/symbol.</p><p>${esc(model?.scope || 'Model unavailable')}</p><p>Absolute Fs needs metadata or a time reference; candidates are hypotheses.</p>` }));
  }
  const cmp = a.automated_parameter_comparison;
  const nyq = fs / 2; const plabel = { dc_i: 'DC, I', dc_q: 'DC, Q', carrier_offset_hz: 'Carrier offset' };
  const fmtFine = v => (Math.abs(v) < 1e5 ? `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)} Hz` : fmtHz(v, true));
  const fmtVal = (k, v) => (k === 'carrier_offset_hz' ? fmtFine(v) : fmtNum(v, 5));
  const rows = Object.entries(cmp).map(([k, v]) => {
    const flags = [v.agreement >= 1 ? '<span class="flag ok">Agree</span>' : '<span class="flag warn">Disagree</span>'];
    const det = v.learned_detail;
    if (k === 'carrier_offset_hz' && Math.abs(v.learned) > nyq) flags.push(`<span class="flag warn">Model value is outside ±${fmtHz(nyq)}, impossible at this rate</span>`);
    if (det) { flags.push(`<span class="flag">${pct(det.confidence)} confidence</span>`); if (Math.abs(v.learned) > det.trained_range_hz[1]) flags.push('<span class="flag warn">Beyond the range it was trained on</span>'); }
    const extra = det ? `<br><span class="note" style="margin:0">${esc(det.method)}${det.refined_on_power ? `, line at x^${det.refined_on_power}` : ''}</span>` : '';
    return `<tr><td>${esc(plabel[k] || k)}<br>${flags.join(' ')}${extra}</td><td>${fmtVal(k, v.dsp)}</td><td>${fmtVal(k, v.learned)}</td></tr>`;
  }).join('');
  const disagreements = Object.values(cmp).filter(v => v.agreement < 1).length;
  const cfoRow = cmp.carrier_offset_hz; const legacyCfo = cfoRow?.legacy_tinymlp;
  const learnedBranch = Boolean(cfoRow?.learned_detail);
  tiles.push(tile({ cat: 'model estimates', title: 'Manual versus model', av: 'model',
    src: learnedBranch ? 'Named DSP estimators next to SpecCFO (carrier offset) and the DC regressor' : 'Named DSP estimators next to the synthetic-trained MLP, raw input',
    metric: disagreements ? `${disagreements} disagreement${disagreements > 1 ? 's' : ''}` : 'all agree',
    body: `<table class="cmp"><thead><tr><th>Parameter</th><th>Manual</th><th>Model</th></tr></thead><tbody>${rows}</tbody></table>
      ${a.learned_region ? `<p class="note">Model carrier offset measured on ${a.learned_region.sample_count.toLocaleString()} samples from ${a.learned_region.source === 'longest_energy_segment' ? 'the longest energy segment' : 'the start of the file'}, so silence doesn't dilute it.</p>` : ''}
      ${legacyCfo != null ? `<p class="note">The old MLP gave ${fmtHz(legacyCfo, true)} for the same offset. It was trained at a fixed 1 MS/s and isn't used for this value any more.</p>` : ''}
      <p class="note">The models don't estimate ${esc((a.provenance.not_supported_by_model || []).join(', ').replaceAll('_', ' '))}. Where they disagree, trust neither until a reference confirms one.</p>` }));

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
  if (!a || a.kind === 'audio') { box.innerHTML = a?.kind === 'audio' ? 'Declared audio uses the audio overview rather than an RF receiver.' : ''; return; }
  const cls = a.modulation_classification; const mp = a.manual_parameter_estimation; const chips = [];
  if (!cls.abstained && RX_SUPPORTED.includes(cls.predicted_modulation)) chips.push(`<button class="chip" data-set="p-mod" data-value="${cls.predicted_modulation}">Classifier says ${cls.predicted_modulation.toUpperCase()}</button>`);
  const learnedCfo = a.automated_parameter_comparison?.carrier_offset_hz?.learned_detail;
  if (learnedCfo) chips.push(`<button class="chip" data-set="p-cfo" data-value="${learnedCfo.carrier_offset_hz.toFixed(1)}" title="${pct(learnedCfo.confidence)} confidence">SpecCFO ${fmtHz(learnedCfo.carrier_offset_hz, true)}</button>`);
  chips.push(`<button class="chip" data-set="p-cfo" data-value="${mp.coarse_carrier_offset_hz.carrier_offset_hz.toFixed(1)}">Manual CFO ${fmtHz(mp.coarse_carrier_offset_hz.carrier_offset_hz, true)}</button>`);
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
      body: `<div class="big">Abstained</div><p class="note">${esc(d.reason)}</p>${c ? `<p class="note">Classifier: ${c.abstained ? 'abstained' : `${esc(String(c.predicted_modulation).toUpperCase())} at ${pct(c.confidence)}`}.</p>` : ''}<p class="note">Review the estimates and supply manual receiver settings above.</p>` }));
    setTiles('rx-feed', tiles); return;
  }
  const cfg = d.configuration; const mod = d.modulation; const fs = state.demodSettings?.sample_rate_hz;
  if (cfg.parameter_sources) tiles.push(tile({ title: 'Receiver parameter sources', src: 'Applied settings and evidence', av: 'rx',
    metric: d.input_region?.source || 'whole capture',
    body: `<dl class="kv">${Object.entries(cfg.parameter_sources).map(([k,v]) => `<dt>${esc(k.replaceAll('_',' '))}</dt><dd>${esc(v)}</dd>`).join('')}</dl><p class="note">${d.input_region ? `Input samples ${d.input_region.sample_start} through ${d.input_region.sample_start + d.input_region.sample_count - 1}.` : ''} Timing is a fixed offset, not a recovered symbol clock.</p>` }));
  const dec = d.decisions_preview || [];
  if (mod === '2fsk') {
    const vals = dec.map(x => x.frequency_step_rad);
    const idx = vals.map((_, n) => n); const m = Math.max(...vals.map(Math.abs)) || 1;
    tiles.push(instrument({ wide: true, ratio: '21 / 8', title: 'Time sink, discriminator', src: `Mean phase step per symbol, first ${vals.length} symbols`, av: 'rx', metric: '2-FSK',
      legend: [{ label: 'Symbol mean', trace: 0 }, { label: 'Slicer threshold', trace: 1, dash: true }],
      note: 'Above zero decides 1, below decides 0. Two well-separated rows mean the tones and symbol timing fit. Drag to zoom, double-click to reset.',
      render(c, st) { Plot.line(c, { x: { label: 'Symbol index', domain: [0, Math.max(1, vals.length - 1)] }, y: { label: 'Phase step', unit: 'rad/sample', domain: Plot.niceDomain(-m, m, 0.08) }, zoomX: true,
        series: [{ x: idx, y: vals, color: tr(0), dots: true, joined: true, hidden: !st.on[0] }], hlines: st.on[1] ? [{ y: 0, color: tr(1), label: 'threshold' }] : [],
        readout: x => { const k = Math.max(0, Math.min(vals.length - 1, Math.round(x))); return `symbol ${k}  ${vals[k].toFixed(4)} rad/sample  bit ${vals[k] >= 0 ? 1 : 0}`; } }); } }));
  } else {
    const rms = Math.sqrt(dec.reduce((s, x) => s + x.i * x.i + x.q * x.q, 0) / Math.max(1, dec.length)) || 1;
    const pts = dec.map(x => ({ i: x.i / rms, q: x.q / rms }));
    const ideal = mod === 'bpsk' ? [{ i: -1, q: 0 }, { i: 1, q: 0 }] : [-1, 1].flatMap(i => [-1, 1].map(q => ({ i: i * Math.SQRT1_2, q: q * Math.SQRT1_2 })));
    tiles.push(instrument({ wide: true, ratio: '21 / 9', title: 'Constellation sink, decisions', src: `First ${dec.length} integrate-and-dump outputs, RMS-normalised`, av: 'rx', metric: mod.toUpperCase(),
      legend: [{ label: 'Symbols', trace: 0 }, { label: 'Ideal points', trace: 1 }],
      note: 'Tight clusters on the crosses mean timing and carrier settings fit. A ring means residual carrier offset. A cloud at the centre means these symbols fall in a quiet stretch.',
      render(c, st) { Plot.scatter(c, { x: { label: 'In-phase' }, y: { label: 'Quadrature' }, points: st.on[0] ? pts : [], ideal: st.on[1] ? ideal : null }); } }));
  }
  const segs = state.analysis?.analysis.raw_branch.visualization?.segmentation.segments || [];
  const evm = d.quality.evm_rms;
  tiles.push(tile({ title: 'Quality', src: 'Error vector magnitude after fixed integrate-and-dump', av: 'rx', metric: `${d.symbol_count.toLocaleString()} symbols`,
    body: `<div class="big">${evm == null ? 'n/a' : pct(evm)}<small>${evm == null ? 'EVM not defined for FSK' : 'EVM RMS'}</small></div>
      <p class="note">${esc(d.quality.meaning)}${segs.length > 0 && evm != null ? (d.input_region?.source !== 'whole_capture' && d.input_region ? ' Measured on the selected analysis region.' : ' It covers the whole file, so quiet stretches between bursts push it up.') : ''}</p>
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
    if (inp.wav_header) {
      add('WAV header', `${inp.wav_header.channels} channels, ${inp.wav_header.bits_per_channel_sample} bits/channel sample, ${inp.wav_header.frame_count} frames; ${esc(inp.wav_header.encoding)}.`);
      add('WAV interpretation', `${esc(inp.wav_header.waveform_role)} (${esc(inp.wav_header.role_source)}). ${esc(inp.conversion.method)}.`);
    }
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
