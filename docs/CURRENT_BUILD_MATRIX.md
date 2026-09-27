# DEmod build matrix

This is the single source of truth for the current product state. “Built” means executable code exists and has automated tests. “Partial” means there is a usable fragment but it is not connected to the end-to-end workflow. “Not built” means no implementation currently exists.

| Area | Status | Present now | Still required before it is an end-to-end feature |
| --- | --- | --- | --- |
| Landing page and static workbench | Built | GitHub Pages landing page and five-tab web interface | Connect the controls and charts to the local analysis API. |
| WAV ingestion | Built | Standard PCM WAV parsing, channel handling, basic level/frequency summary | Audio playback, speech/telemetry interpretation, and session export. |
| Raw IQ ingestion | Built | Interleaved `s8`, `cu8`, `s16le`, `s16be`, `f32le`, `f32be` parsing | Format auto-detection ranking, explicit confirmation UI, and chunked large-file handling. |
| Input metadata provenance | Partial | Reports label raw-IQ sample rate as analyst hypothesis | SigMF parser, WAV metadata display, source hash, immutable session manifest. |
| Dataset registry and schema | Built | Recipe files, external-data registry, record schema and manifests | Controlled downloader/importer, licence/checksum pinning and train/validation/test split manager. |
| Synthetic IQ generation | Built | In-memory BPSK/QPSK/8PSK/16QAM and OFDM-shaped recipes with noise/impairments | Fractional timing, pulse shaping, FSK generation, coded frames and a validation corpus. |
| Non-destructive denoising | Partial | Reversible opt-in DC and impulse branches; disabled by default | Connect to pipeline; measure signal preservation; add guarded RFI/IQ-imbalance paths. |
| Deterministic basic features | Built | DC, RMS, crest, instantaneous-frequency statistics, DFT preview and modulation triage | Calibrated noise floor, clipping flags, cyclostationary features and symbol-rate candidates. |
| Spectrum / waterfall / constellation | Built | NumPy FFT/PSD, STFT arrays, sampled constellation and energy segments | Render real backend arrays in UI and support window/chunk navigation. |
| Burst segmentation | Partial | Energy-based segment candidates | Threshold calibration, multi-burst merging, real-data evaluation and UI interaction. |
| Manual parameter estimation | Partial | DC and coarse PSK-compatible CFO estimation | Symbol-rate/timing/carrier loops and documented validity ranges. |
| Automated parameter estimation | Partial | Recipe-trained 7→12→3 MLP for DC-I, DC-Q and coarse CFO | Saved/calibrated model, real held-out evaluation, timing/rate/bandwidth model targets and abstention policy. |
| Manual-versus-AI comparison | Built as local MVP | Local API reports estimates, absolute disagreement and provenance; source bytes remain in memory | UI integration, model/version registry, calibration plots and analyst override recording. |
| Modulation classification | Partial | Explainable heuristic family ranking | Train/evaluate an AMC model; confidence calibration; supported-family gating. |
| Paper-driven noise-robust AMC experiment | Not built | Design choice documented below | High-SNR selection, rotation/CTS augmentation, complex-valued autoencoder and held-out ablation. |
| Symbol synchronisation | Not built | Design and literature map only | Oerder–Meyr acquisition plus Gardner/Mueller–Müller tracking, fixtures and BER tests. |
| Carrier recovery / equalisation | Not built | Coarse CFO metric only | Family-gated carrier loop/equaliser and EVM evidence. |
| Demodulation | Not built | No bitstream or audio output is generated | Start with BPSK, QPSK and 2-FSK; validate BER/SER/EVM on known captures. |
| OFDM receiver | Not built | OFDM-shaped synthetic waveform only | Profile declaration, preamble/timing/CFO/channel estimation/equalisation and subcarrier decoder. |
| FEC, deinterleaving, descrambling | Not built | No implementation | Protocol-specific candidate executor with known vectors, framing and CRC acceptance. |
| Voice intelligence | Not built | WAV summary only | Audio demodulation for declared modes, playback and any authorised transcription workflow. |
| Result export | Partial | JSON analysis reports | Evidence bundle, plot assets, configuration/model versions and replay command. |
| Large-scale and secure deployment | Not built | Static public documentation; local loopback comparison API | Authorised-network deployment, object storage/chunk workers, access control, audit log and performance tests. |

## How the supplied paper changes the plan

Gao et al., *Enhancing Noise Robustness in Few-Shot Automatic Modulation Classification via Complex-Valued Autoencoders*, **Electronics** 2026, 15, 674, DOI [10.3390/electronics15030674](https://doi.org/10.3390/electronics15030674), is directly relevant to the **noise-robust AMC** stage—not direct protocol demodulation. Its proposed sequence is: isolate high-SNR examples from scarce captures, apply rotation and cyclic-time-shift augmentation, train a complex-valued noise-reduction network (CNRN), then classify modulation. The paper reports its evaluation on RML2016.10a plus a physical platform; those results are not evidence that it will work on NTRO data.

DEmod will use it as an ablation-tested branch after raw feature analysis:

```text
raw IQ -> deterministic feature/provenance branch -> high-SNR selection
       -> rotation + cyclic-time-shift augmentation -> complex autoencoder candidate
       -> AMC classifier -> bounded conventional synchronisation/demodulation
```

The raw, no-denoising DSP branch remains mandatory alongside it. The CNRN output is accepted only if held-out evidence improves AMC and does not worsen downstream BER/EVM for a supported receiver. It must never overwrite the original IQ recording or be presented as a universal demodulator.
