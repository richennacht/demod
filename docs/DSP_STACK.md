# DSP package choice

DEmod uses a layered Python stack.

| Package | Role now | Why |
| --- | --- | --- |
| Python standard library | WAV/container parsing, provenance, small deterministic baseline | Minimal ingestion dependency |
| [NumPy](https://numpy.org/doc/stable/reference/routines.fft.html) | Production MVP FFT, `fftfreq`, Hann windows, STFT matrix, vector operations | Fast, well-supported numerical array operations; required by `requirements-dsp.txt` |
| [SciPy signal](https://docs.scipy.org/doc/scipy/reference/signal.html) | Planned optional filters, resampling, peak detection and modern `ShortTimeFFT` | Useful once a supported receiver branch requires it; not yet a runtime dependency |
| Matplotlib/Plotly | Optional server-side rendered plots | The current web UI consumes JSON arrays; rendering package is not required by the backend |
| GNU Radio | Optional laboratory/streaming integration | Excellent for SDR-device flowgraphs and live QT sinks, but too large to make the hackathon file-analysis backend depend on it |

`src/spectral_analysis.py` currently uses NumPy directly so the calculation is a real complex FFT/STFT, not the prior browser byte preview. Its JSON has the transform/window settings, axes, DC-centering decision, plot arrays, segmentation settings and denoising state.

PySDR’s [frequency-domain chapter](https://pysdr.org/content/frequency_domain) gives the same practical model: windowed FFTs stacked over time form a spectrogram/waterfall. SciPy’s current documentation recommends `ShortTimeFFT` for new feature-rich STFT work, while treating `scipy.signal.spectrogram` as legacy. DEmod can adopt `ShortTimeFFT` when SciPy is added for filtering/resampling, without changing its provenance schema.
