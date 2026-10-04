# DEmod build matrix

This is the single source of truth for the current product state. “Built” means executable code exists and has automated tests. “Partial” means there is a usable fragment but it is not connected to the end-to-end workflow. “Not built” means no implementation currently exists.

Latest rate/modulation update: automatic PCM WAV-header Fs extraction, existing SigMF metadata fill, duration-referenced raw Fs and DSP-versus-learned SPS hypotheses are now available through `/rates` and Capture. Unknown raw absolute Fs remains unavailable without a reference. The SPS feature-softmax model is simulation-tested for PSK/QAM at 2/4/8/16 SPS; full WAV waveform analysis is still CLI-only. DemodAMC now returns full ranking, mean-window probabilities, window agreement and noise abstention. See [review and measurements](RATE_AND_MODULATION_REVIEW.md); older entries below describe the broader workflow and are not a claim of universal rate inference.

| Area | Status | Present now | Still required before it is an end-to-end feature |
| --- | --- | --- | --- |
| Landing page and analyst UI | Built | GitHub Pages landing page; `web/` UI connected to the local API (capture settings with provenance, backend FFT/waterfall/scatter/segments, raw vs derived, manual vs model, classifier, BPSK/QPSK/2-FSK receiver, report export); also served by the API at `/ui/` | Large-file chunking beyond the 16 MiB API cap, real-capture example set, user testing with analysts. |
| WAV ingestion | Built | Standard PCM WAV parsing, channel handling, basic level/frequency summary | Audio playback, speech/telemetry interpretation, and session export. |
| Raw IQ ingestion | Built | Interleaved `s8`, `cu8`, `s16le`, `s16be`, `f32le`, `f32be` parsing | Format auto-detection ranking, explicit confirmation UI, and chunked large-file handling. |
| Input metadata provenance | Built for local IQ analysis | SHA-256, byte count, format/sample-rate/centre/gain source tags and SigMF sidecar-field parser | UI-side SigMF attachment and persistent session catalog. |
| Dataset registry and schema | Built | Recipe files, external-data registry, record schema and manifests | Controlled downloader/importer, licence/checksum pinning and train/validation/test split manager. |
| Synthetic IQ generation | Built | In-memory BPSK/QPSK/8PSK/16QAM and OFDM-shaped recipes with noise/impairments | Fractional timing, pulse shaping, FSK generation, coded frames and a validation corpus. |
| Non-destructive denoising | Built for the MVP analysis path | Raw and opt-in derived branches, reversible DC/impulse corrections, audit actions and both raw/derived spectral analysis | Held-out BER/EVM preservation study; guarded RFI/IQ-imbalance paths. |
| Deterministic basic features | Built | DC, RMS, crest, instantaneous-frequency statistics, DFT preview and modulation triage | Calibrated noise floor, clipping flags, cyclostationary features and symbol-rate candidates. |
| Spectrum / waterfall / constellation | Built | NumPy FFT/PSD, STFT arrays, sampled constellation and energy segments | Render real backend arrays in UI and support window/chunk navigation. |
| Burst segmentation | Built as an analysis proposal | Robust-smoothed-energy proposals flow into the raw/derived report | Threshold calibration, multi-burst merging, real-data evaluation and UI interaction. |
| Manual parameter estimation | Built baseline | Named DC, coarse PSK-CFO, power, FFT bandwidth/burst and transition-periodicity estimators | Symbol timing/carrier loops and validity calibration. |
| Automated parameter estimation | Built, simulation-validated | SpecCFO carrier-offset estimator (spectral CNN plus periodogram refinement, with confidence); recipe-trained TinyMLP kept for DC only | Real held-out captures, 8PSK and low-SNR weakness, retraining on the corrected recipe generator. See `research/README.md`. |
| Manual-versus-AI comparison | Built as local MVP | Local API returns raw/derived DSP, estimator disagreement, metadata provenance and source bytes remain in memory | UI integration, model/version registry, calibration plots and analyst override recording. |
| Modulation classification | Built, simulation-validated | DemodAMC: 12 classes, carrier offset removed first, calibrated probabilities and abstention. Legacy centroid kept as a fallback | Real held-out captures, 16QAM/64QAM confusion, classes beyond the 12. See `research/README.md`. |
| Paper-driven noise-robust AMC experiment | Not built | Design choice documented below | High-SNR selection, rotation/CTS augmentation, complex-valued autoencoder and held-out ablation. |
| Symbol synchronisation | Not built | Design and literature map only | Oerder–Meyr acquisition plus Gardner/Mueller–Müller tracking, fixtures and BER tests. |
| Carrier recovery / equalisation | Not built | Coarse CFO metric only | Family-gated carrier loop/equaliser and EVM evidence. |
| Demodulation | Built controlled MVP | Manual-override BPSK/QPSK/2-FSK integrate-and-dump hard decisions, bit preview, EVM and GNU Radio graph descriptor | Real timing/carrier recovery, BER/SER validation, framing/CRC/FEC and protocol support. |
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
