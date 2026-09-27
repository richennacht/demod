# DEmod

**DEmod** is an analyst-facing, transparent signal-analysis workbench for SIH 2026 problem statement **SIH26147**: *Automated model for analysis of `.IQ` and `.wav` files along with signal parameter extraction* (NTRO).

## What the first MVP does

- Reads mono or multi-channel PCM WAV recordings with Python's standard library.
- Reads raw complex IQ samples when their format is explicitly supplied: `s16le`, `s16be`, `s8`, `cu8`, `f32le`, `f32be`.
- Produces reproducible, explainable DSP measurements: DC offset, RMS, peak, crest factor, instantaneous-frequency statistics, FFT peak, 99%-energy occupied bandwidth, and ranked modulation-family hypotheses.
- Emits a machine-readable JSON report that a dashboard and later ML classifier can consume.

It deliberately does **not** claim universal blind recovery of FEC, interleaving, encryption, or every unknown waveform. Those are research-level SIGINT problems and must be demonstrated only for supported signal families with recorded validation data.

## Quick start

Requires Python 3.10+; no third-party packages are needed for the baseline analyzer.

```powershell
python src/analyze_signal.py path\\to\\recording.wav
python src/analyze_signal.py path\\to\\recording.iq --sample-rate 2400000 --iq-format s16le
```

Each command writes a JSON report to standard output. Save a report when needed:

```powershell
python src/analyze_signal.py sample.wav --output report.json
```

For a headerless `.iq` recording, `--sample-rate` and `--iq-format` are hypotheses supplied by the analyst. Raw bytes alone cannot reliably establish absolute sample rate, RF centre frequency, byte order, signedness, or the original receiver gain. DEmod records that limitation in the report rather than treating a plausible interpretation as ground truth.

## Architecture direction

```text
WAV / IQ upload
      |
format validation + metadata declaration
      |
DSP feature extraction  --->  immutable JSON analysis report
      |                              |
supported-family classifier               analyst UI / export / review queue
```

## Data and labels

The project uses real SDR captures for evaluation and controlled synthetic waveforms for exact parameter labels. The [dataset plan and label contract](docs/DATASET_AND_LABEL_SCHEMA.md) distinguishes source-provided facts from predictions; the machine-readable [manifest schema](data/schema/demod-record.schema.json) and [synthetic example](data/manifests/example-synthetic.json) are ready for the generator and evaluator.

### Dataset design

Large recordings are not committed to Git. The repository contains reproducible recipes, manifests, checksums/provenance fields, and compact regression fixtures; authorised source files are fetched on demand into controlled storage. This avoids treating terabytes of immutable capture data as source code, while retaining a reproducible record of every training or evaluation input.

The [HF/VHF/UHF recipe corpus](data/recipes/sih-hf-vhf-uhf.json) is the current proxy for SIH26147: it produces BPSK/QPSK/16QAM and OFDM-shaped examples with deterministic seeds and receiver/channel impairments. It includes centre-frequency and sample-rate truth only as hidden generator audit metadata. FEC and interleaving are explicitly `none_mvp` until coded-waveform generation and evaluation are implemented; the project does not claim FEC recognition yet.

The [external dataset registry](data/manifests/external-datasets.json) separates each corpus by role:

- [Panoradio HF](https://panoradio-sdr.de/radio-signal-classification-dataset/) supplies labelled synthetic HF reference windows: 18 transmission modes, 2,048 complex samples at 6 kHz, with AWGN, frequency/phase offsets, and Watterson fading. It is useful for mode coverage, not proof of over-the-air realism.
- [LoRadar](https://zenodo.org/records/16302856) provides real UHF satellite-ground LoRa bursts: headerless interleaved complex64 IQ at 4 MS/s in the 399–403 MHz range, with real SNR/Doppler conditions. It validates raw-format ingestion, burst and Doppler features, but LoRa/CSS is outside the first BPSK/QAM/FSK/OFDM demodulation scope.
- The [Open-Sourced Time-Frequency Domain RF Classification Framework](https://zenodo.org/records/4603987) supplies recorded SDR data in SigMF data/metadata pairs for metadata ingestion and coarse operational-class validation.
- The [real-world IQ AMR dataset](https://data.mendeley.com/datasets/tjzsbph49x/1) is a candidate for modulation/channel validation after its licence, version, checksum, and supplied label fields are pinned locally.

No public SIH/NTRO capture package or sensor specification has been identified. These sources are therefore validation references and proxy distributions, **not** a claim that they match the hidden NTRO evaluation data. When authorised representative sensor captures are supplied, DEmod will lock them as session-level held-out data, calibrate receiver profiles, and report the measured synthetic-to-real gap with confidence intervals.

## Non-destructive denoising

DEmod will keep raw recordings immutable and create versioned, auditable derived branches for each correction. Its first safe candidates are gated DC/LO-leakage correction, blind I/Q-imbalance correction, robust impulse masks, and time-frequency RFI flags. Band filtering, CFO recovery and wavelet shrinkage are optional, supported-waveform steps—not universal preprocessing—because they can erase meaningful signal structure. See the [non-destructive denoising policy and literature](docs/NONDESTRUCTIVE_DENOISING.md).

The `src/denoise_iq.py` branch is present but not connected to the analyser or training pipeline. All stages are disabled by default; the initial implementation only records artifact candidates unless an explicit caller opts into a reversible DC or impulse branch.

## Feature analysis and ML direction

Feature analysis comes next: raw spectrum/STFT, occupancy and burst structure, carrier-offset and bandwidth estimates, amplitude/phase/frequency statistics, higher-order cumulants, and cyclostationary candidates. DEmod will use these deterministic measures as an explainable baseline, then compare them with a compact raw-IQ 1-D CNN fused with the feature vector. See the [manual feature-analysis and model plan](docs/FEATURE_ANALYSIS_AND_MODEL_PLAN.md).

### Real FFT, waterfall and segmentation backend

The Python backend now contains `src/spectral_analysis.py`: a true complex NumPy FFT/PSD, Hann-window STFT waterfall, sampled constellation points, and robust energy-based segment candidates. It runs over actual interpreted IQ samples, emits plot arrays and transform settings in the JSON provenance, and does **not** denoise or mutate raw samples. Install with `python -m pip install -r requirements-dsp.txt`; the [DSP stack decision](docs/DSP_STACK.md) explains why NumPy is the MVP runtime and SciPy/GNU Radio are optional next layers. The package choice follows the [NumPy FFT reference](https://numpy.org/doc/stable/reference/routines.fft.html), [SciPy signal documentation](https://docs.scipy.org/doc/scipy/reference/signal.html), and the practical [PySDR waterfall guide](https://pysdr.org/content/frequency_domain).

## Blind modulation detection direction

The supported search catalogue is designed around OOK/ASK, BPSK/DBPSK/QPSK/DQPSK/8PSK, 16/32/64/256QAM, 2/4/MFSK plus GFSK/GMSK/MSK, OFDM, and later spread-spectrum/chirp/pulse/analogue routing. This is not a claim that every family has a decoder today. DEmod will use coarse-to-fine **bounded hypothesis search**: continuous receiver settings can be optimised within a family, while discrete family selection uses raw DSP, cyclic/higher-order features, synchronisation, EVM/likelihood and framing/CRC evidence. Printable-text "gibberish" is only a low-weight late check. See the [modulation-hypothesis design and citations](docs/MODULATION_HYPOTHESIS_SEARCH.md).

## Research basis

The following papers directly motivate the currently documented technology choices. They are linked here so the implementation, evidence boundary and source material remain together.

- DC-offset/CFO/IQ-imbalance estimation and compensation: [Liu & Li, 2011](https://doi.org/10.1016/j.sigpro.2010.12.002), [Song et al., 2017](https://arxiv.org/abs/1712.05970), and [Wang et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5751594/).
- Selective impulse/RFI mitigation rather than blanket smoothing: [Hwang et al., 2017](https://doi.org/10.1587/transfun.E100.A.3041), [Nita & Gary, 2010](https://digitalcommons.njit.edu/fac_pubs/13405/), and [Taylor et al., 2018](https://arxiv.org/abs/1808.10365).
- Optional wavelet denoising, subject to decoder-level validation: [Baxter & Upton, 2002](https://doi.org/10.1111/1467-9876.00276).
- Classical modulation analysis: [Dobre et al., 2007](https://doi.org/10.1049/iet-com:20050176), [Abdelmutalab et al., 2016](https://doi.org/10.1016/j.phycom.2016.08.001), and [Dobre et al., 2010](https://doi.org/10.1007/s11277-009-9776-2).
- Blind carrier/symbol-rate candidates through cyclostationarity: [Zhang et al., 2012](https://doi.org/10.1016/j.proeng.2011.12.753) and [Güner, 2014](https://doi.org/10.1002/dac.2606).
- Hybrid raw-IQ/deep-learning model choice and domain-shift cautions: [Thakur & Imtiaz, 2026](https://www.mdpi.com/2079-9292/15/10/2163) and [Tian et al., 2026](https://doi.org/10.1016/j.sigpro.2025.110444).

## Delivery roadmap

1. **Baseline (now):** deterministic file ingestion and explainable DSP measures.
2. **Demo capability:** spectrogram/waterfall UI, supported modulation-family classifier, and labelled evaluation set.
3. **Advanced capability:** symbol-rate estimation and demodulation only for explicitly supported families; every result includes confidence and evidence.

## Safety and demo boundary

Use only recordings and datasets that your team is authorized to analyse. The MVP is a file-analysis tool; it does not acquire radio signals or perform operational interception.
