# SIH26147 data strategy

## What is actually known

The public SIH26147 statement describes `.IQ` and `.wav` recordings from designated sensors across HF, VHF and UHF, but it does **not** publish a capture download, sensor model, IQ datatype/byte order, sample-rate convention, centre-frequency plan, or label schema. DEmod must therefore not assert that any public corpus is the NTRO evaluation distribution.

Raw headerless IQ has an irreducible ambiguity: byte layout can be ranked from plausibility checks, but neither absolute `Fs` nor RF centre frequency is recoverable from bytes alone. The data contract records these as supplied metadata, truth (synthetic), or an explicit hypothesis.

## Versioned corpus design

| Layer | Role | Stored in Git? | Ground truth |
| --- | --- | --- | --- |
| `data/recipes/sih-hf-vhf-uhf.json` | deterministic HF/VHF/UHF proxy generation | Yes | Exact generator audit truth |
| Generated micro-fixtures | parser and DSP regression only | Yes, small only | Exact |
| External real captures | ingestion and domain validation | Manifest/download instructions only | Source-provided fields only |
| NTRO-provided captures | final blind evaluation and profile calibration | Controlled storage, never public Git | Only supplied/instrumented facts |

The recipe corpus initially covers BPSK/QPSK/16QAM and OFDM-shaped signals; it deliberately labels FEC and interleaving as `none_mvp`. It must not be used to claim FEC or interleaver recognition until coded waveform generation and evaluation are implemented.

## External data to collect, without vendoring large bytes

1. **HF mode coverage — Panoradio HF.** 172,800 synthetic 2,048-IQ windows at 6 kHz with 18 HF modes, AWGN, random offsets and CCIR-520/Watterson fading. Use as a training-reference and regression source, not as OTA realism. [Dataset description](https://panoradio-sdr.de/radio-signal-classification-dataset/).
2. **UHF satellite/sensor realism — LoRadar.** 181 real satellite-ground LoRa bursts, 399–403 MHz, 4 MS/s, headerless interleaved complex64 files, with SNR/Doppler and PHY configuration. It is valuable for raw-format ingestion, burst detection and Doppler features, but CSS/LoRa is outside the first BPSK/QAM/FSK/OFDM demodulator. [Zenodo record](https://zenodo.org/records/16302856).
3. **Recorded SigMF format realism — Open-Sourced Time-Frequency Domain RF Classification Framework.** SDR captures in paired SigMF data/metadata files; use only a pinned, licensed evaluation subset. It supports metadata parsing and operational-class checks, not universal FEC truth. [Zenodo record](https://doi.org/10.5281/zenodo.4603987).
4. **Modulation/channel validation — Real-world IQ AMR corpus.** Use only after downloading its dataset card and licence alongside the checksum; retain its source labels separately from DEmod predictions. [Mendeley dataset](https://data.mendeley.com/datasets/tjzsbph49x/1).

No source above has complete transmitter bits, FEC and interleaving truth across HF/VHF/UHF. That is why generated examples are necessary.

## Required acquisition manifest fields

For every external file, preserve: URL/DOI, version, licence, checksum, download timestamp, receiver/sample format, `Fs`, centre frequency, gain if supplied, session/device/location split key, source annotation and label confidence. Never copy a source label into synthetic truth or turn an unknown into a training target.

## Acceptance gate

When NTRO supplies representative authorised captures, lock them as a held-out session-level set. Calibrate receiver profiles and compare synthetic/real features per band, but call the corpus "matched" only if the held-out test and a documented confidence interval pass. Before then it remains a proxy corpus.
