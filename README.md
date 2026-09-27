# DEmod

**DEmod** is an analyst-facing, transparent signal-analysis workbench for SIH 2026 problem statement **SIH26147**: *Automated model for analysis of `.IQ` and `.wav` files along with signal parameter extraction* (NTRO).

## What the first MVP does

- Reads mono or multi-channel PCM WAV recordings with Python's standard library.
- Reads raw complex IQ samples when their format is explicitly supplied (interleaved signed 16-bit I/Q by default).
- Produces reproducible, explainable baseline measurements: duration, sample rate, DC offset, RMS level, peak level, and a zero-crossing frequency estimate.
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

## Delivery roadmap

1. **Baseline (now):** deterministic file ingestion and explainable DSP measures.
2. **Demo capability:** spectrogram/waterfall UI, supported modulation-family classifier, and labelled evaluation set.
3. **Advanced capability:** symbol-rate estimation and demodulation only for explicitly supported families; every result includes confidence and evidence.

## Safety and demo boundary

Use only recordings and datasets that your team is authorized to analyse. The MVP is a file-analysis tool; it does not acquire radio signals or perform operational interception.
