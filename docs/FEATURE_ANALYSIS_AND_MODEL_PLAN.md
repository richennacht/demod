# Feature analysis and model plan

## Manual, deterministic analysis first

Feature analysis begins on raw IQ and on any explicitly selected derived branch. Each value is reported with its window, representation hypothesis and confidence/limitation.

| Question | Manual DSP evidence | Scope and literature |
| --- | --- | --- |
| Is energy present and where? | Windowed FFT/PSD, noise-floor estimate, peak persistence, occupied bandwidth, STFT/waterfall, burst segmentation | Basic spectrum sensing; retain time-frequency evidence rather than only a class label |
| Is it a tone, hopping signal, pulse or burst? | Spectral peak tracks, STFT connected components, duration/duty cycle, rise/fall timing, repetition interval | Useful before selecting a demodulator; do not call a transient "noise" solely because it is sparse |
| Is there a carrier/offset? | Complex phase-increment mean, spectral peak, frequency trajectory; report baseband offset separately from unknown RF centre frequency | Baseband carrier offset can be estimated; absolute RF centre frequency needs metadata/reference |
| What is the bandwidth? | 99%-energy and threshold bandwidth, spectral edges, multi-window stability | Always preserve definition/window; bandwidth is estimator-dependent |
| What modulation family is plausible? | Amplitude distribution/envelope variance, unwrapped phase and instantaneous frequency, constellation density, higher-order moments/cumulants | Classical feature-based AMC: [Dobre et al., 2007](https://doi.org/10.1049/iet-com:20050176); HOC hierarchy: [Abdelmutalab et al., 2016](https://doi.org/10.1016/j.phycom.2016.08.001) |
| What are the symbol-rate candidates? | Magnitude/phase autocorrelation periodicity, spectral/cyclic-spectrum peaks, eye/constellation only after candidate timing recovery | Cyclostationary CSD: [Zhang et al., 2012](https://doi.org/10.1016/j.proeng.2011.12.753); periodic variation: [Güner, 2014](https://doi.org/10.1002/dac.2606) |
| Is there modulation-specific periodic structure? | Spectral correlation function/cyclic cumulants over candidate cycle frequencies | More noise-tolerant but compute-heavy; [Dobre et al., 2010](https://doi.org/10.1007/s11277-009-9776-2) |

FEC, interleaving and payload are **not** waveform features. Estimate them only after a supported demodulator produces soft/hard bits and test hypotheses by synchronisation, parity/CRC, decoder metrics and repeatability.

## Recommended automatic system

Use a hybrid, staged system rather than an end-to-end black box:

1. **Deterministic front end:** format hypotheses, segmentation, raw measurements and quality gates.
2. **DSP baseline:** rules plus a calibrated, small feature classifier (regularised logistic regression or gradient-boosted trees) over the listed measurements. This is the mandatory explainable benchmark.
3. **Learning branch:** a compact 1-D residual CNN over normalised I/Q windows, fused late with the engineered feature vector and quality flags. It outputs coarse family (`noise/artifact`, FSK, PSK, QAM, OFDM/multicarrier, unsupported), not an invented protocol/FEC label.
4. **Evidence fusion:** report agreement/disagreement between branches; abstain or request analyst review under low SNR, multi-signal occupancy, unsupported family, or large disagreement.
5. **Later sequence head:** only if long context demonstrably matters, add a small temporal-convolution/GRU head for hopping/burst sequence classification. Do not start with a transformer: it needs substantially more varied real data and makes the 36-hour MVP harder to validate.

CNNs are a sensible first learned baseline for local raw-IQ structure and inexpensive inference; hybrid feature/raw-IQ approaches address known real-world domain-shift risks. See the current [AMC review](https://www.mdpi.com/2079-9292/15/10/2163) and the survey of data representations and architectures ([Tian et al., 2026](https://doi.org/10.1016/j.sigpro.2025.110444)).

## Evaluation rules

- Split by receiver/session/location/transmitter, never adjacent windows.
- Measure coarse-family accuracy, calibration/abstention, bandwidth and carrier-offset error separately from synthetic exact-truth demodulation/BER.
- Train only on truth that is source-provided or generated. Unknown fields remain unknown.
- Hold out all authorised NTRO representative captures from tuning until a final evaluation.
