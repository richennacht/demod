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

## Delivery roadmap

1. **Baseline (now):** deterministic file ingestion and explainable DSP measures.
2. **Demo capability:** spectrogram/waterfall UI, supported modulation-family classifier, and labelled evaluation set.
3. **Advanced capability:** symbol-rate estimation and demodulation only for explicitly supported families; every result includes confidence and evidence.

## Safety and demo boundary

Use only recordings and datasets that your team is authorized to analyse. The MVP is a file-analysis tool; it does not acquire radio signals or perform operational interception.
