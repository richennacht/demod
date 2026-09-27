const state = { file: null, analysed: false, demodulated: false, fec: false };
const tabs = [...document.querySelectorAll('.stage-link')];
const panels = [...document.querySelectorAll('.tab-panel')];
const pageTitle = document.getElementById('page-title');
const titles = { 'file-management': 'File management', dsp: 'DSP characterization', demodulation: 'Demodulation detector', fec: 'FEC & bit recovery', results: 'Results' };

function openTab(id) {
  tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.tab === id));
  panels.forEach(panel => panel.classList.toggle('active', panel.id === id));
  pageTitle.textContent = titles[id];
  window.location.hash = id;
}

function setFile(name, detail) {
  state.file = { name, detail };
  document.getElementById('selected-file').textContent = name;
  document.getElementById('selected-detail').textContent = detail;
  document.getElementById('run-analysis').disabled = false;
  document.querySelectorAll('.demo-file').forEach(button => button.classList.toggle('selected', button.dataset.file === name));
}

function runAnalysis() {
  if (!state.file) return;
  state.analysed = true;
  document.getElementById('dsp-state').textContent = 'Auto-characterized';
  document.getElementById('demod-state').textContent = 'Candidates ready';
  document.getElementById('results-subtitle').textContent = `Automated evidence report for ${state.file.name}.`;
  document.getElementById('result-mode').textContent = 'Auto result';
  document.getElementById('result-name').textContent = 'QPSK signal candidate';
  document.getElementById('result-description').textContent = 'The selected input was interpreted, characterised, and ranked using the current automatic settings.';
  document.getElementById('result-confidence').textContent = '86%';
  document.getElementById('result-format').textContent = 'Complex signed 16-bit LE';
  document.getElementById('result-input').textContent = state.file.name;
  document.getElementById('result-params').textContent = '+183.25 kHz · 24.1 kHz BW';
  document.getElementById('result-fec').textContent = 'Awaiting demodulation';
  openTab('results');
}

tabs.forEach(tab => tab.addEventListener('click', () => openTab(tab.dataset.tab)));
document.querySelectorAll('.demo-file').forEach(button => button.addEventListener('click', () => setFile(button.dataset.file, button.dataset.detail)));
document.getElementById('file-input').addEventListener('change', event => { const file = event.target.files[0]; if (file) setFile(file.name, `${file.type || 'Local binary capture'} · ${(file.size / 1024).toFixed(1)} KB`); });
document.getElementById('run-analysis').addEventListener('click', runAnalysis);
document.getElementById('apply-dsp').addEventListener('click', () => { state.analysed = true; document.getElementById('dsp-state').textContent = 'Characterization refreshed'; });
document.querySelectorAll('.hypothesis').forEach(item => item.addEventListener('click', () => { document.querySelectorAll('.hypothesis').forEach(other => other.classList.remove('selected')); item.classList.add('selected'); }));
document.getElementById('run-demod').addEventListener('click', () => { state.demodulated = true; document.getElementById('fec-state').textContent = 'Soft bits available'; document.getElementById('result-fec').textContent = 'Soft bits ready for bounded search'; openTab('fec'); });
document.getElementById('run-fec').addEventListener('click', () => { state.fec = true; document.getElementById('fec-state').textContent = 'Insufficient evidence'; document.getElementById('result-fec').textContent = 'No supported FEC identified'; });
document.querySelectorAll('[data-go]').forEach(button => button.addEventListener('click', () => openTab(button.dataset.go)));
document.getElementById('export-report').addEventListener('click', () => { const report = { version: '0.1.0-ui', input: state.file, result: state.analysed ? { format: 'cs16_le_iq', modulation: 'QPSK', confidence: 0.86 } : null }; const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }); const link = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'demod-analysis-report.json' }); link.click(); URL.revokeObjectURL(link.href); });
document.getElementById('reset-button').addEventListener('click', () => { state.file = null; state.analysed = false; state.demodulated = false; state.fec = false; document.getElementById('selected-file').textContent = 'No file selected'; document.getElementById('selected-detail').textContent = 'Upload a file or select a prepared demonstration capture.'; document.getElementById('run-analysis').disabled = true; document.getElementById('dsp-state').textContent = 'Awaiting file'; document.getElementById('demod-state').textContent = 'Awaiting DSP'; document.getElementById('fec-state').textContent = 'Awaiting demodulation'; document.querySelectorAll('.demo-file').forEach(button => button.classList.remove('selected')); openTab('file-management'); });
if (window.location.hash && document.getElementById(window.location.hash.slice(1))) openTab(window.location.hash.slice(1));
