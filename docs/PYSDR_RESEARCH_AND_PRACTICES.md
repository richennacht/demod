# PySDR research and coding-practice index

## Scope and provenance

This is a **curated DEmod index**, not a copy of PySDR. PySDR is an open-source textbook by Marc Lichtman, licensed CC BY-NC-SA 4.0; its complete, current source is at [777arc/PySDR](https://github.com/777arc/PySDR) and the reader is at [pysdr.org](https://pysdr.org/). The audit used upstream source snapshot `d9eb5534759c63a507faeeee13904373e11f6e18` (27 English content chapters), checked on 2026-09-27.

PySDR does not maintain one universal bibliography. It places “Further Reading” and “External Resources” within individual chapters, so this document records all references from the chapters DEmod relies on plus the coding rules adopted from them. The upstream source remains the authoritative complete reference set, including hardware, beamforming, radar, GUI and TDOA chapters that are out of this MVP’s scope.

## DEmod-relevant chapter map

| PySDR chapter | What DEmod takes from it | Resulting project rule |
| --- | --- | --- |
| [Frequency Domain](https://pysdr.org/content/frequency_domain) | FFT axes, `fftshift`, windowing, PSD and a waterfall as stacked FFTs | Record FFT size, window, hop, frequency axis and dB reference in every plot report |
| [IQ Sampling](https://pysdr.org/content/sampling) | Complex baseband meaning, Nyquist and PSD limits | Never infer absolute `Fs` or RF centre frequency from headerless bytes |
| [Digital Modulation](https://pysdr.org/content/digital_modulation) | ASK/PSK/QAM/FSK symbols and constellations | Separate family hypotheses from a decoder claim |
| [Noise and Random Variables](https://pysdr.org/content/noise) | Complex AWGN, SNR/SINR and statistical interpretation | Report assumptions/metrics, do not call visual cleanliness a decoding success |
| [Filters](https://pysdr.org/content/filters) | Complex filters, FIR/IIR, chunk state and overlap techniques | Apply filters only in a parameterised derived branch, preserving raw capture and state |
| [Channel Coding](https://pysdr.org/content/channel_coding) | Code-rate and soft versus hard decisions | Start FEC work only after a quality-gated soft-bit stream exists |
| [IQ Files and SigMF](https://pysdr.org/content/iq_files) | IQIQ interleaving, explicit dtype, saturation, SigMF sidecars | Treat layout as a hypothesis; store `Fs`, frequency, format and annotations as metadata |
| [Pulse Shaping](https://pysdr.org/content/pulse_shaping) | Matched filters, RRC and eye diagrams | Use timing/pulse hypotheses only after family/rate candidates exist |
| [Synchronization](https://pysdr.org/content/sync) | Coarse/fine CFO, timing and frame sync | Optimise continuous receiver parameters inside a modulation hypothesis, never globally |
| [Cyclostationary Processing](https://pysdr.org/content/cyclostationary) | CAF/SCF, FSM/TSM/FAM and OFDM cyclic structure | Use cyclic features as an expensive, evidence-gated stage; record matrix sizes/pooling |
| [Detection using Correlation](https://pysdr.org/content/detection) | Correlators, Neyman–Pearson framing and CFAR | Score known-preamble/frame hypotheses with measurable false-alarm controls |
| [End-to-End RDS Example](https://pysdr.org/content/rds) | A bounded protocol chain: frequency shift, filter, decimate, sync, BPSK and differential decode | Implement narrow, testable receiver branches rather than a universal decoder |

## Coding practices adopted

1. Use `np.ndarray`, not deprecated `np.matrix`; assert array shape and complex dtype at module boundaries.
2. Use complex IQ through DSP stages; do not discard Q or silently coerce to real.
3. Apply a declared window before FFT and use `fftfreq`/`fftshift` consistently.
4. For long recordings, process chunks with overlap/state; cap preview arrays but retain full-run provenance.
5. Explicitly track filter delay, edge transients, FIR taps, decimation ratio and resampler state.
6. Record the scale/reference of every dB figure and detect clipping before trusting spectral features.
7. Keep capture metadata in SigMF-compatible sidecars and annotations in sample-index coordinates.
8. Use worker processes/queues for costly STFT/cyclostationary tasks; keep the UI responsive and render only derived plot data.
9. Evaluate receiver branches with sync/EVM/BER/CRC or a calibrated detection metric—not a text-readability heuristic.
10. Version source, code, dependency versions, recipe seed, parameters and output report together.

## References surfaced by PySDR that DEmod will cite

These are the primary/external references directly useful to our next stages; they complement the AMC and denoising literature already listed in the README.

1. Roberts, Brown & Loomis Jr., “Computationally Efficient Algorithms for Cyclic Spectral Analysis,” *IEEE Signal Processing Magazine*, 1991. [Source linked by PySDR](https://www.researchgate.net/profile/Faxin-Zhang-2/publication/353071530_Computationally_efficient_algorithms_for_cyclic_spectral_analysis/links/60e69d2d30e8e50c01eb9484/Computationally-efficient-algorithms-for-cyclic-spectral-analysis.pdf).
2. Napolitano, *Cyclostationary Processes and Time Series: Theory, Applications, and Generalizations*. [Publisher record](https://www.sciencedirect.com/book/monograph/9780081027080/cyclostationary-processes-and-time-series).
3. Da Costa, *Detection and Identification of Cyclostationary Signals*, Naval Postgraduate School dissertation, 1996. [DTIC record](https://apps.dtic.mil/sti/pdfs/ADA311555.pdf).
4. Sutton, Nolan & Doyle, “Cyclostationary signatures in practical cognitive radio applications,” *IEEE JSAC*, 2008. [IEEE record](https://ieeexplore.ieee.org/document/4413137).
5. Papoulis & Pillai, *Probability, Random Variables, and Stochastic Processes*, 2002; Kay, *Intuitive Probability and Random Processes using MATLAB*, 2006. These support the statistical/noise assumptions, while the DEmod implementation remains testable without copying their material.
6. [SigMF specification/project](https://github.com/sigmf/SigMF), which PySDR uses for portable recording metadata.
7. [GNU Radio](https://www.gnuradio.org/) and [gr-rds](https://github.com/bastibl/gr-rds), which PySDR identifies as external RDS/demodulation resources; DEmod treats them as laboratory references, not an embedded dependency.

## What we will not copy or claim

- We will not mirror PySDR prose, figures, code listings or unrelated chapters into this repository.
- PySDR examples are educational guidance; they are not evidence that DEmod handles an unknown NTRO sensor stream.
- Hardware-specific advice for Pluto/USRP/RTL-SDR/HackRF applies only after actual sensor metadata/hardware is supplied.
- PySDR itself notes missing/ongoing end-to-end capabilities (for example equalization/OFDM work); DEmod must validate each added branch independently.
