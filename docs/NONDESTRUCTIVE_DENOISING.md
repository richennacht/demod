# Non-destructive denoising policy

## Principle

An unknown IQ recording must never be overwritten. DEmod stores an immutable raw stream and makes every correction a versioned derived stream with its parameters, masks and before/after measurements. A stage may run automatically only when its detection statistic passes a documented gate; otherwise it emits a finding for analyst review.

This is especially important for SIH26147: a narrowband interferer may be the signal of interest, a DC component may be a genuine baseband feature, and aggressive smoothing can remove FSK transitions, QAM amplitude levels, OFDM subcarriers or short radar pulses.

## Stages suitable before modulation identification

| Stage | What it addresses | Conservative action | Gate and audit | Literature |
| --- | --- | --- | --- | --- |
| DC/LO-leakage correction | Constant I/Q bias from a direct-conversion receiver | Subtract a robust complex location estimate from a **copy** | Require a persistent centre-bin excess and report removed I/Q value; disable by default when the desired signal is plausibly at DC | [Liu & Li, 2011](https://doi.org/10.1016/j.sigpro.2010.12.002); [Inamori et al., 2009](https://doi.org/10.1109/TWC.2009.080139) |
| Blind I/Q imbalance compensation | Gain/phase mismatch and image leakage in a quadrature front end | Estimate a widely-linear correction and create a candidate corrected stream | Compare image-rejection/circularity before and after; reject correction if it worsens held-out spectral symmetry | [Song et al., 2017](https://arxiv.org/abs/1712.05970); [Wang et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5751594/) |
| Impulse detection/blanking | Lightning, switching and isolated ADC/glitch spikes | Mark robust-MAD outliers; use blanking or bounded interpolation only in an optional branch | Preserve original samples and a per-sample mask; report fraction blanked, threshold and window | [Hwang et al., 2017](https://doi.org/10.1587/transfun.E100.A.3041) |
| Time-frequency RFI flagging | Intermittent narrowband or broadband non-Gaussian interference | Generate an STFT/spectral-kurtosis occupancy mask; do not automatically erase a band | Retain the mask and affected time/frequency cells; analyst or downstream supported decoder chooses notch/excision | [Nita & Gary, 2010](https://digitalcommons.njit.edu/fac_pubs/13405/); [Taylor et al., 2018](https://arxiv.org/abs/1808.10365) |
| Known-band filtering and decimation | Out-of-band noise after a bandwidth/centre hypothesis exists | Apply a documented complex FIR low-pass/band-pass to a derived stream, then decimate | Never infer a filter edge from a single noisy FFT; require declared or multi-window stable occupied-band hypothesis | Standard receiver DSP; evaluate on supported waveform families |
| Wavelet shrinkage | Non-stationary/background noise and burst noise | Optional analysis or audio-preview branch with reversible thresholding | Compare EVM/BER for labelled data and feature drift for unknown data; never promote merely because SNR rises | [Baxter & Upton, 2002](https://doi.org/10.1111/1467-9876.00276); [Hwang et al., 2017](https://doi.org/10.1587/transfun.E100.A.3041) |

## What is deliberately not "safe" as a universal first step

- A fixed low-pass, moving average, median or Savitzky-Golay filter: it changes symbol timing, phase/frequency trajectories and pulse/radar edges unless its passband is known.
- Automatic notch removal: the strongest tone may be the carrier, a beacon, FSK energy, or the target itself.
- Spectral subtraction/wavelet thresholding used as a replacement for raw IQ: improvements in visual SNR do not establish preserved bit decisions.
- Carrier-frequency-offset removal before identifying whether a rotating phase is impairment, modulation, hopping, Doppler or intentional signal structure.

## DEmod MVP order

1. Decode container metadata where available; for raw IQ create several explicitly labelled representation hypotheses.
2. Segment and measure raw IQ first: DC, clipping, robust scale, spectral occupancy, impulsive-rate, spectral-kurtosis map and image-symmetry indicators.
3. Offer DC correction and impulse masking as separate reversible branches. Do not combine corrections until each improves its own measured artifact criterion.
4. Run spectrum/waterfall/constellation and modulation hypotheses on **raw and derived** branches. Keep branch provenance in the report.
5. Only after a supported waveform/bandwidth hypothesis exists, apply a filter, timing/CFO recovery and demodulator. Judge the path by task metrics (sync success, EVM, BER/CRC where known), not by a prettier plot.

## Evaluation contract

For synthetic captures, compare raw and derived branches against exact truth using EVM, symbol error rate, BER/CRC, occupied-bandwidth error and pulse/burst-boundary error. For real captures without transmitter truth, report only changes in diagnostics and downstream decoder confidence; do not call a method successful solely from an increase in estimated SNR. Split evaluation by session/receiver/location so a receiver artifact cannot leak from train to test.
