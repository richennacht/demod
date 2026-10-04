"""Explicit WAV waveform interpretation, independent of binary-container decoding."""
import numpy as np
from analyze_signal import decode_raw_iq
from rate_estimation import wav_metadata


def is_wav(raw):
    return raw[:4] == b'RIFF' and raw[8:12] == b'WAVE'


def analytic_signal(values):
    # Same FFT construction documented for scipy.signal.hilbert; NumPy fallback
    # keeps the existing runtime dependency small. DC/Nyquist are not doubled.
    n = len(values)
    if not n:
        raise ValueError('The WAV contains no frames.')
    h = np.zeros(n); h[0] = 1
    if n % 2 == 0:
        h[1:n//2] = 2; h[n//2] = 1
    else:
        h[1:(n+1)//2] = 2
    return np.fft.ifft(np.fft.fft(values) * h)


def decode_capture(raw, iq_format, sample_rate_hz, wav_role='unspecified', if_centre_hz=0.):
    if not is_wav(raw):
        return decode_raw_iq(raw, iq_format), sample_rate_hz, None
    meta, values = wav_metadata(raw)
    fs = meta['sample_rate_hz']
    if sample_rate_hz and (not np.isfinite(sample_rate_hz) or abs(sample_rate_hz-fs) > 1e-6):
        raise ValueError('Supplied Fs conflicts with the WAV header. Use the header rate; resampling is a separate operation.')
    if not np.isfinite(if_centre_hz) or abs(if_centre_hz) >= fs/2:
        raise ValueError('IF centre must be finite and inside the WAV Nyquist interval.')
    if wav_role in ('stereo_iq', 'stereo_qi'):
        if meta['channels'] != 2:
            raise ValueError('Stereo I/Q interpretation requires exactly two WAV channels.')
        i, q = (0,1) if wav_role == 'stereo_iq' else (1,0)
        samples = values[:,i] + 1j*values[:,q]
        method = f'channel {i+1} = I; channel {q+1} = Q; PCM normalization'
    elif wav_role == 'real_if':
        if meta['channels'] != 1:
            raise ValueError('Real IF interpretation requires a mono WAV; select the intended channel externally.')
        samples = analytic_signal(values[:,0])
        method = 'positive-frequency analytic signal by FFT Hilbert construction'
    elif wav_role == 'audio':
        samples = values.mean(axis=1).astype(complex)
        method = 'audio channel average for level/spectral overview; no RF reconstruction'
    else:
        raise ValueError('Declare WAV content: stereo_iq, stereo_qi, real_if, or audio. Channels alone do not establish I/Q.')
    if wav_role == 'audio' and if_centre_hz != 0:
        raise ValueError('IF translation is not applicable to demodulated audio.')
    if if_centre_hz and wav_role != 'audio':
        samples *= np.exp(-2j*np.pi*if_centre_hz*np.arange(len(samples))/fs)
        method += '; analyst IF frequency translated to baseband'
    meta.update({'waveform_role': wav_role, 'role_source': 'analyst_hypothesis', 'conversion': method,
                 'if_centre_hz': if_centre_hz, 'resampled': False, 'sample_count': len(samples)})
    return samples.tolist(), fs, meta


def capture_provenance(raw, iq_format, fs, centre, gain, source, wav_info):
    from provenance import input_provenance
    record = input_provenance(raw, iq_format, fs, centre, gain, source)
    if wav_info:
        record['container'] = 'wav'
        record['wav_header'] = wav_info
        record['representation'] = {'iq_format': f"wav_pcm_{wav_info['waveform_role']}", 'iq_format_source': 'analyst_wav_interpretation',
                                    'encoding_source': 'wav_header', 'encoding': wav_info['encoding']}
        record['capture']['sample_rate_source'] = 'wav_header'
        record['conversion'] = {'method': wav_info['conversion'], 'if_centre_hz': wav_info['if_centre_hz'],
                                'source': 'analyst_wav_interpretation', 'resampled': False,
                                'original_sha256_scope': 'complete uploaded WAV container, including header'}
    else:
        record['container'] = 'raw_iq'
    return record
