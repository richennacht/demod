"""Rate evidence in sample units; absolute Hz needs an external time reference."""
from pathlib import Path
import io
import wave
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / 'data/models/symbol_rate.npz'
CLASSES = np.array([2, 4, 8, 16])


def features(samples):
    x = np.asarray(samples, dtype=complex)[:4096]
    if len(x) < 128 or not np.isfinite(x).all():
        raise ValueError('Rate estimation needs at least 128 finite complex samples.')
    x = x - x.mean()
    power = np.mean(abs(x)**2)
    if power < 1e-15:
        return np.zeros(96)
    x /= np.sqrt(power)
    lag = np.array([abs(np.mean(x[k:] * x[:-k].conj())) for k in range(1, 33)])
    # Square-law and transition-power spectra expose cyclic lines without CFO.
    spectra = []
    for v in (abs(x)**2, abs(np.diff(x))**2):
        v = v - v.mean()
        p = abs(np.fft.rfft(v * np.hanning(len(v))))**2
        bands = np.array([a.max(initial=0) for a in np.array_split(p[1:], 32)])
        spectra.extend(np.log1p(100 * bands / (p.sum() + 1e-12)))
    return np.r_[lag, spectra]


def estimate(samples, sample_rate_hz=None, known_symbol_rate_hz=None, recording_duration_seconds=None):
    if sample_rate_hz is not None and (not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0):
        raise ValueError('Sample rate must be finite and positive.')
    if recording_duration_seconds is not None:
        if not np.isfinite(recording_duration_seconds) or recording_duration_seconds <= 0:
            raise ValueError('Recording duration must be finite and positive.')
        sample_rate_hz = len(samples) / recording_duration_seconds
    all_samples = np.asarray(samples, dtype=complex)
    start = 0
    if len(all_samples) > 4096:
        starts = np.arange(0, len(all_samples) - 4096 + 1, 2048)
        energy = np.cumsum(np.r_[0., abs(all_samples)**2])
        start = int(starts[np.argmax(energy[starts + 4096] - energy[starts])])
    x = all_samples[start:start + 4096]
    f = features(x)
    transition = abs(np.diff(x))**2
    transition -= transition.mean()
    p = abs(np.fft.rfft(transition * np.hanning(len(transition))))**2
    frequencies = np.fft.rfftfreq(len(transition))
    valid = (frequencies >= 1/32) & (frequencies <= .5)
    indexes = np.flatnonzero(valid)
    order = indexes[np.argsort(p[indexes])[::-1]]
    candidates = []
    for k in order:
        sps = 1 / frequencies[k]
        if all(abs(sps - c['samples_per_symbol']) > .3 for c in candidates):
            candidates.append({'samples_per_symbol': float(sps), 'cycles_per_sample': float(frequencies[k]),
                               'symbol_rate_hz': float(frequencies[k] * sample_rate_hz) if sample_rate_hz else None})
        if len(candidates) == 5:
            break
    result = {'manual': {'method': 'transition-power periodogram; cyclic-line candidates, harmonic ambiguity', 'candidates': candidates},
              'region': {'sample_start': start, 'sample_count': len(x), 'selection': 'highest_energy_4096_sample_window'},
              'learned': None, 'absolute_sample_rate_hz': sample_rate_hz,
              'sample_rate_source': 'sample_count / analyst_recording_duration' if recording_duration_seconds else 'analyst_hypothesis' if sample_rate_hz else 'unavailable',
              'limitations': ['Absolute Fs is not identifiable without metadata or a physical time reference.',
                              'Candidates may be harmonics; no symbol clock recovery is performed.']}
    if MODEL_PATH.exists():
        with np.load(MODEL_PATH, allow_pickle=False) as m:
            logits = ((f - m['mean']) / m['scale']) @ m['weights'] + m['bias']
            probabilities = np.exp(logits - logits.max()); probabilities /= probabilities.sum()
            best = int(probabilities.argmax())
            abstained = float(probabilities[best]) < .8 or np.max(f[:32]) < .08
            result['learned'] = {'samples_per_symbol': None if abstained else int(CLASSES[best]),
                                 'abstained': bool(abstained), 'probability': float(probabilities[best]),
                                 'candidates': [{'samples_per_symbol': int(c), 'probability': float(v)} for c, v in zip(CLASSES, probabilities)],
                                 'model_file': MODEL_PATH.name, 'scope': 'Experimental softmax model: simulated linear PSK/QAM, integer SPS 2/4/8/16 only; probabilities uncalibrated.'}
            if known_symbol_rate_hz and not abstained:
                if not np.isfinite(known_symbol_rate_hz) or known_symbol_rate_hz <= 0:
                    raise ValueError('Known symbol rate must be finite and positive.')
                result['conditional_sample_rate_hz'] = float(CLASSES[best] * known_symbol_rate_hz)
                result['conditional_source'] = 'analyst_known_symbol_rate × model_samples_per_symbol; hypothesis'
    return result


def wav_metadata(raw):
    with wave.open(io.BytesIO(raw), 'rb') as w:
        if w.getsampwidth() not in (1, 2, 4):
            raise ValueError('Supported PCM WAV widths are 8, 16 and 32 bits.')
        frames = w.readframes(w.getnframes())
        width, channels, fs = w.getsampwidth(), w.getnchannels(), w.getframerate()
    dtype = {1: 'u1', 2: '<i2', 4: '<i4'}[width]
    v = np.frombuffer(frames, dtype=dtype).astype(float).reshape(-1, channels)
    v = (v - (128 if width == 1 else 0)) / (2**(width*8-1))
    return {'sample_rate_hz': fs, 'source': 'wav_header', 'channels': channels, 'sample_width_bytes': width}, v
