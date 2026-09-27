# Bounded blind modulation-hypothesis search

## Why this is not ordinary gradient descent

The modulation family is a discrete choice: moving from QPSK to 4-FSK or OFDM is not a small numerical step. Gradient-like optimization is valid **inside** a candidate receiver for continuous settings such as carrier-frequency offset, timing phase, symbol rate, channel/filter bandwidth and equalizer coefficients. The outer loop is model selection: enumerate a bounded catalogue, optimise each supported branch, score it using physical/decoder evidence, then abstain when no branch passes.

"Less gibberish" is an unsafe primary objective: encrypted, compressed, binary telemetry and unknown character encodings look like gibberish even when decoded correctly; a false decoder can produce accidental printable text. Text/telemetry plausibility is only a late, low-weight score after framing/CRC or other objective evidence.

## Initial catalogue

| Family | Initial hypotheses | Status in DEmod |
| --- | --- | --- |
| Amplitude | OOK/ASK, M-ASK, AM, SSB | Detect/characterise later; audio path separate |
| Phase | BPSK, DBPSK, QPSK/DQPSK, π/4-DQPSK, 8PSK, 16PSK | BPSK/QPSK/8PSK synthetic recipes; demodulator pending |
| Quadrature amplitude | 16QAM, 32QAM, 64QAM, 256QAM | 16QAM recipe; demodulator pending |
| Frequency/continuous phase | 2FSK, 4FSK, MFSK, GFSK, GMSK, MSK | Synthetic/demodulator pending |
| Multicarrier | OFDM with BPSK/QPSK/QAM subcarriers, DMT | OFDM-shaped recipe; protocol decoder pending |
| Spread spectrum | DSSS/CDMA, CSS/LoRa | Detect/route as unsupported initially; LoRadar validates raw UHF ingestion |
| Analogue / non-telemetry | NFM/WFM, pulse/radar, chirp | Characterise and route; no generic demodulation claim |

The public problem wording specifically names FSK, QAM and PSK, and requests a broad modulation detector. OFDM is a necessary additional family because it changes spectral, cyclic and constellation behaviour. The catalogue is deliberately extensible rather than falsely presenting every listed mode as implemented.

## Candidate scoring

For each segment and representation hypothesis:

1. Use raw DSP features to prune impossible families: occupied bandwidth, amplitude variation, phase/frequency trajectory, spectral flatness/peaks, cyclostationary signatures, burst/pulse structure, and higher-order cumulants.
2. For each remaining family, explore a bounded grid/coarse-to-fine search over symbol-rate candidates, samples-per-symbol, CFO, timing phase and filter bandwidth. Optimise continuous values by likelihood/EVM/synchronisation objective where supported.
3. Run a family-specific receiver only after it has a timing/carrier quality gate. Do not apply destructive denoising globally; if a candidate has a declared occupied band or impulsive mask, test that derived branch and retain the branch audit.
4. Score, in order: synchronisation/preamble correlation; likelihood or residual/EVM; constellation/state separation; cyclic-feature agreement; framing repetition; parity/CRC; then low-weight payload plausibility.
5. Calibrate scores on held-out sessions. Return top-k candidates plus their complete parameter path, or `abstain/unsupported`.

## Literature basis

- The likelihood-based and feature-based AMC taxonomy is surveyed by [Dobre, Abdi, Bar-Ness & Su (2007)](https://doi.org/10.1049/iet-com:20050176).
- Higher-order cumulants and hierarchical classification for PSK/QAM are demonstrated by [Abdelmutalab, Assaleh & El-Tarhuni (2016)](https://doi.org/10.1016/j.phycom.2016.08.001).
- Cyclic cumulants for fading-channel modulation classification: [Dobre et al. (2010)](https://doi.org/10.1007/s11277-009-9776-2).
- Cyclostationary carrier/symbol-rate estimation: [Zhang et al. (2012)](https://doi.org/10.1016/j.proeng.2011.12.753) and [Güner (2014)](https://doi.org/10.1002/dac.2606).
- Current hybrid raw-IQ, engineered-feature and deep-learning trade-offs: [Thakur & Imtiaz (2026)](https://www.mdpi.com/2079-9292/15/10/2163).
