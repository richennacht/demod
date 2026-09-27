# DEmod end-to-end workflow specification (draft 0.1)

## Purpose

DEmod is an analyst-facing workflow for authorised `.iq` and `.wav` recordings. It must preserve the original input, state every assumption, make deterministic DSP evidence inspectable, compare supported learned estimates against manual methods, and abstain where evidence is insufficient. It is not an interception system and does not claim universal FEC recovery or decryption.

## Product outcome

For one authorised recording, an analyst can create an immutable analysis session; declare or import capture metadata; obtain visual and numeric RF features; compare deterministic and learned estimates; run supported modulation/demodulation hypotheses; inspect any recovered payload or audio; and export an evidence bundle that reproduces the result.

## Workflow contract

```text
Authorised file / SigMF package
  -> session + immutable source manifest
  -> metadata read or analyst hypotheses
  -> raw-format decoder and structural validation
  -> non-destructive feature branch
  -> DSP and learned-estimator comparison
  -> bounded modulation hypotheses
  -> supported-family synchronisation / demodulation
  -> optional framing, FEC and de-whitening candidates
  -> evidence-ranked result or abstention
  -> provenance-rich export and analyst review
```

Every stage emits a versioned JSON artifact with: input artifact ID/hash, configuration, code/model version, execution time, evidence/metrics, confidence or abstention reason, and links to its parents. Derived samples never replace raw samples.

## Current implementation inventory

For the compact, all-items status table, use the [current build matrix](CURRENT_BUILD_MATRIX.md). The expanded workflow inventory remains below.

| Workflow component | Current state | What actually works now | Important boundary |
| --- | --- | --- | --- |
| Static web workbench | Implemented | Five-stage analyst UI and provenance-oriented browser preview | Browser preview is not a Python DSP backend or a decoder. |
| WAV ingest | Implemented baseline | PCM WAV reading, mono reduction, basic level/frequency summary | No speech intelligence/transcription. |
| Raw IQ ingest | Implemented baseline | `s8`, `cu8`, `s16le`, `s16be`, `f32le`, `f32be` interleaved I/Q decoding | Format and sample rate are analyst hypotheses unless metadata supplies them. |
| SigMF / dataset registry | Planned with schemas | External dataset registry, record schema, recipe manifests | No complete SigMF sidecar parser or acquisition workflow. |
| Synthetic data | Implemented MVP | Deterministic, in-memory recipe generation with PSK/QAM/OFDM-shaped signals and RF impairments | Not proof of match to hidden NTRO captures; no coded waveform curriculum. |
| Non-destructive denoising | Implemented but disconnected | Reversible opt-in DC/impulse branch; disabled by default | No automatic denoising decision and no destructive branch in the workflow. |
| Feature analysis | Implemented baseline | DC/RMS/crest, instantaneous frequency, DFT/FFT, occupied bandwidth, STFT waterfall, sampled constellation, energy segments | Results are estimates, not a full receiver parameter solution. |
| Manual versus learned comparison | Implemented MVP | Local API compares manual DC/CFO estimates with a small recipe-trained MLP; returns disagreement | Model only targets DC-I, DC-Q, coarse CFO and is not calibrated for field use. |
| Modulation detection | Design only | Explainable family triage scores | No trained/validated AMC classifier and no selection acceptance threshold. |
| Synchronisation/demodulation | Not built | — | No symbol-timing recovery, carrier loop, or supported-family decoder. |
| Framing/FEC/de-whitening | Not built | — | Must not be labelled decryption until a supported, authorised protocol passes validation. |
| Results/export | Partial | JSON reports and UI provenance cues | No session store, downloadable evidence bundle, or case-review workflow. |
| Deployment | Partial | Public static GitHub Pages; local comparison API | Sensitive/large RF data must stay in a controlled local or ministry-hosted service. |

## Required build specification

### 1. Session, ingest and metadata

Build a local-first analysis service with a session ID, SHA-256 source hash, size, source authorisation flag, and immutable original-file reference. Detect WAV container metadata. Parse SigMF `.sigmf-meta` when present. For headerless IQ, offer ranked format hypotheses but require analyst confirmation before downstream claims. Store sample rate, centre frequency, gain, format, endian, channels and their provenance separately: `metadata`, `analyst_hypothesis`, `deterministic_estimate`, or `learned_estimate`.

**Acceptance:** the same input and configuration produce the same artifact graph; no source bytes are silently altered; every non-metadata physical parameter is labelled as an assumption or estimate.

### 2. Feature-analysis engine

Make the NumPy feature pipeline the canonical analysis path and render its spectrum, waterfall, constellation and energy segments in the UI. Add DC/LO leakage flags, clipping/quantisation flags, noise-floor estimate, peak/occupied bandwidth, spectral peaks, burst start/stop candidates, and bounded symbol-rate candidates. Preserve both uncorrected and optional-derived views.

**Acceptance:** fixture tones/bursts meet numeric tolerances; plots cite transform/window/FFT settings; feature output is exportable JSON.

### 3. Parameter-estimation ladder

For each target, retain one named manual baseline and one model only where matched labels exist.

| Target | Manual baseline | Learned component to build | Acceptance evidence |
| --- | --- | --- | --- |
| DC offset | complex mean / robust location | Existing MLP, then calibrated regressor | held-out MAE and disagreement rate by impairment slice |
| Coarse CFO | fourth-power / phase-increment for compatible PSK | Existing MLP, then family-gated model | error in Hz and abstention outside family/SNR range |
| Symbol rate | cyclostationary / spectral candidates | feature-fusion ranker | top-k correct rate on held-out labelled captures |
| Timing phase | Oerder-Meyr acquisition, Gardner/Mueller-Müller tracking | raw-IQ + feature regressor | residual timing error and BER/EVM after recovery |
| Bandwidth / bursts | FFT/STFT/energy segmentation | optional segment proposal model | IoU and frequency-error measurements |
| Modulation family | cumulants/cyclic/amplitude-phase statistics | raw-IQ 1-D CNN + feature fusion | macro-F1, confusion matrix and calibrated abstention |

Do not train timing models until generator labels include pulse shape, fractional timing offset and channel conditions.

### 4. Modulation and demodulation workflow

Implement a bounded, evidence-ranked hypothesis executor rather than trying every algorithm blindly. Start with an explicit MVP family set: BPSK, QPSK, 2-FSK/4-FSK, and narrow, declared OFDM profiles. For each candidate, record preprocessing, synchronisation state, constellation/EVM evidence, frame/CRC evidence when a known profile applies, and failure reason. Add QAM only after timing/carrier recovery tests are robust.

The noise-robust AMC experiment before this receiver stage follows [Gao et al. (2026)](https://doi.org/10.3390/electronics15030674): select high-SNR examples from scarce data, augment with rotation and cyclic time shifts, train a complex-valued noise-reduction autoencoder, then classify modulation. This is a research branch for *classification robustness*, not itself a demodulator. It will be retained only if a held-out ablation improves AMC without reducing BER/EVM after conventional supported-family recovery.

**Acceptance:** generated and authorised held-out captures show measured BER/SER/EVM against known labels; unsupported formats end in an abstention, not fabricated text.

### 5. Optional payload reconstruction

Build this only for a declared protocol/profile after demodulation. The pipeline is: bit ordering -> differential decoding if selected -> descrambler candidates -> deinterleaver candidates -> FEC decoder candidates -> framing/CRC validator. Every operation must be reversible in the evidence bundle and only accept a result using structural validation (CRC, valid frame grammar, known telemetry range), not printable-text heuristics.

**Acceptance:** known test vectors and held-out captures reconstruct with documented success rate; encrypted/unknown content remains classified as undecodable rather than “decrypted.”

### 6. UI, export and controlled deployment

Connect the five existing tabs to the local analysis API. Show raw/derived selection, each metadata source, manual-versus-model side-by-side estimates, disagreement flags, plots, candidate rankings, and explicit abstentions. Export a compact evidence bundle containing report JSON, configuration, software/model versions and optional rendered plots—but not copies of restricted raw RF data by default.

Run production inside the authorised network with chunked storage and workers near the data. Public GitHub Pages is limited to demonstrations and documentation.

**Acceptance:** uploading a small authorised fixture completes all implemented stages; the UI never claims an unavailable stage ran; a report can be replayed from its source/configuration IDs.

## Data and evaluation requirements

Maintain three separated splits: synthetic training, synthetic held-out test, and authorised real held-out evaluation. A real/synthetic domain discriminator is a diagnostic, not the delivery objective. Evaluate by signal family, SNR, CFO, timing offset, multipath, ADC/format and band. Report confidence intervals, calibration and abstention coverage. Never mix a test capture into recipes or model fitting.

## Milestones

1. **Feature-analysis demo:** API/UI connection, SigMF metadata parser, canonical plots, JSON export.
2. **Receiver MVP:** labelled timing curriculum; BPSK/QPSK/2-FSK synchronisation and demodulation; BER/EVM validation.
3. **Automated comparison:** trained family classifier and per-target parameter models with disagreement/abstention policy.
4. **Protocol MVP:** one authorised telemetry profile with framing/CRC and declared FEC, evaluated end-to-end.
5. **Operational hardening:** controlled deployment, chunking, session/audit store, access controls and performance tests.

## Non-goals until evidence exists

- Inferring absolute sample rate, centre frequency, gain, or receiver format from arbitrary headerless bytes as fact.
- Universal blind decoding of radar, hopping, satellite, encrypted or unknown signals.
- Uploading sensitive or terabyte-scale recordings to a public hosting provider.
- Calling an output “decrypted” without an authorised protocol and cryptographic/key context.
