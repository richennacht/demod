# DEMOD: SIH26147 solution brief

## Problem interpretation

SIH26147 asks for automated analysis of `.IQ` and `.wav` signal captures, with fine-grained parameter extraction including modulation type, sampling rate, FEC, and interleaving. The system must treat missing capture metadata honestly: a headerless byte stream does not uniquely reveal its sample representation, sample rate, or RF centre frequency.

## Submission thesis

**DEMOD is an evidence-first signal-analysis copilot, not a black-box decoder.** It turns raw recordings into an auditable chain of hypotheses: file representation -> signal segmentation -> DSP measurements -> ranked modulation family -> supported demodulation -> conditional FEC/interleaver candidates. Every conclusion carries its assumptions, plots, confidence, and reproducible feature values.

This is more defensible than a dashboard that simply labels a spectrogram, and more feasible than claiming universal blind decoding.

## Evidence library

| Module | Paper / source | Design use |
| --- | --- | --- |
| Modulation taxonomy | Dobre, Abdi, Bar-Ness & Su, *Survey of Automatic Modulation Classification Techniques* (2007), DOI: [10.1049/iet-com:20050176](https://doi.org/10.1049/iet-com:20050176) | Establish classical likelihood and feature-based modulation recognition baselines. |
| Carrier and symbol rate | Jia et al., *Blind Estimation of Carrier Frequency and Symbol Rate Based on Cyclic Spectrum Density* (2012), DOI: [10.1016/j.proeng.2011.12.753](https://doi.org/10.1016/j.proeng.2011.12.753) | Cyclic-spectrum candidate for blind carrier/symbol-rate estimation. |
| Practical baud/occupancy detection | Mathys, *Efficient Band Occupancy and Modulation Parameter Detection* (2017), [GNU Radio Conference](https://pubs.gnuradio.org/index.php/grcon/article/view/36) | Frequency-domain detection of occupancy and symbol rates before per-signal analysis. |
| OFDM synchronization | Liu & Zhou, *Joint Blind Estimation of Symbol Timing Offset and Carrier Frequency Offset for OFDM Systems with I/Q Imbalance* (2009), DOI: [10.1587/elex.6.443](https://doi.org/10.1587/elex.6.443) | Bounded OFDM branch; do not apply its assumptions to all signals. |
| RF impairment handling | Anttila et al., *RF Impairments in Wireless Transceivers: Phase Noise, CFO, and IQ Imbalance - A Survey* (2021), DOI: [10.1109/ACCESS.2021.3101845](https://doi.org/10.1109/ACCESS.2021.3101845) | DC-offset, CFO, phase-noise, and IQ-imbalance checks before classification. |
| FEC identification | Moosavi & Larsson, *A Fast Scheme for Blind Identification of Channel Codes* (2011), DOI: [10.1109/GLOCOM.2011.6133507](https://doi.org/10.1109/GLOCOM.2011.6133507) | Candidate-code identification only after reliable bit recovery. |
| Interleaver identification | Li et al., *Blind Identification of Interleaver in Channel Coding* (2015), DOI: [10.11884/HPLPB201527.103250](https://doi.org/10.11884/HPLPB201527.103250) | Candidate families: block, convolutional, helical; scope depends on clean decoded bits. |
| Modern AMC model | Kumar et al., *Deep-Learning-Based Classifier With Custom Feature-Extraction Layers for Digitally Modulated Signals* (2024), DOI: [10.1109/TBC.2024.3391056](https://doi.org/10.1109/TBC.2024.3391056) | Later compact classifier informed by cyclic-cumulant operations, not an opaque model-only claim. |
| Data format | [SigMF specification](https://sigmf.org/sigmf-spec.pdf) | Canonical metadata for sample datatype, sample rate, capture properties and annotations. |

## Proposed architecture

```text
                 analyst upload / local capture
                              |
               [1] Format Probe and Metadata Gate
                   header parser + candidate scorer
                              |
               [2] Signal Workbench / DSP Core
       PSD + waterfall + bursts + DC/IQ checks + rate candidates
                              |
               [3] Evidence Fusion and Model Router
         deterministic DSP rules + compact modulation classifier
                              |
                    [4] Analyst Review Console
      ranked result, confidence, supporting plots, override, export
                              |
               [5] Optional Deep Analysis Queue
      supported demodulator -> bits -> FEC/interleaver candidate checks
```

### Layer responsibilities

1. **Format Probe:** parse WAV/SigMF/vendor headers; otherwise rank `cu8`, `cs8`, `cs16_le/be`, and `cf32_le/be` hypotheses from byte structure and signal-likeness. It never fabricates sample-rate or centre-frequency metadata.
2. **DSP Core:** chunk the file; calculate PSD/waterfall, occupied bandwidth, power/noise floor, DC offset, I/Q balance, burst boundaries, instantaneous amplitude/phase/frequency, and symbol-rate candidates.
3. **Evidence Fusion:** begin with deterministic rules. A compact 1D CNN on normalized IQ plus a small feature vector is a later candidate classifier, gated by signal quality and always compared with DSP evidence.
4. **Review Console:** make the evidence visible. The analyst can choose a storage-format candidate and record an override, avoiding an irreversible black-box decision.
5. **Deep Analysis:** supports an explicitly declared list of waveform and code families. Unknown / low-quality input returns `insufficient_evidence`, not invented FEC or interleaver labels.

## Models: what to use and when

| Stage | MVP model | Later model | Reason |
| --- | --- | --- | --- |
| Binary format inference | Candidate enumeration + interpretable scoring | Calibrated gradient-boosted ranker | No labelled data is needed to ship the baseline; scores remain inspectable. |
| Signal detection / segmentation | Energy detector with adaptive noise estimate | CFAR-style detector | Fast, understandable, and suitable for a demo. |
| Modulation family | DSP features: constellation moments, amplitude/phase/frequency statistics, cyclic peaks | Small 1D CNN with feature-fusion head | DSP works on small datasets; the hybrid model is current without requiring a huge foundation model. |
| FEC/interleaver | `not attempted` unless a supported, synchronized bitstream exists | Candidate-family classifier + algebraic validation | Prevents unsupported claims and provides a research route. |

## 36-hour build contract

**Build and demo:** WAV + SigMF + raw `cs16_le`/`cu8` ingestion; format hypothesis ranking; waterfall/PSD; segmentation; measurements; supported BPSK/QPSK/2FSK family ranking; JSON/PDF-style analyst report; synthetic labelled test set.

**Do not claim in 36 hours:** arbitrary proprietary file recovery, universal blind demodulation, or FEC/interleaver reconstruction from unknown noisy IQ. Show these as a gated roadmap, with one carefully controlled supported example only if time remains.

## Four-criterion case

### 1. Novelty

The novelty is **evidence-linked uncertainty management**: the system jointly reasons about file encoding and waveform properties, exposes alternative interpretations, and gives analysts a reproducible evidence ledger. A CRUD upload dashboard or a single-label CNN does not do this.

### 2. Feasibility

The MVP uses mature FFT/STFT, deterministic DSP, a narrow set of formats/modulations, and synthetic test signals. It needs no classified operational dataset, bespoke RF hardware, or a large model-training run.

### 3. Practical scalability

Process recordings as immutable chunks; store only compact features, reports, and object-store pointers. Stateless workers can parallelize files/chunks; an asynchronous queue separates expensive spectrogram/classifier work from the UI. Deploy the analysis runtime inside a controlled ministry network; no capture data needs a public cloud.

### 4. Modern appropriate stack

| Concern | Choice | Why |
| --- | --- | --- |
| DSP service | Python 3.12, NumPy/SciPy, `sigmf` | Mature numerical/DSP ecosystem and open metadata standard. |
| API | FastAPI + Pydantic | Typed contracts; easy local demo and internal-service evolution. |
| UI | React + TypeScript + Plotly | Responsive waterfall/constellation/trace viewing. |
| Models | PyTorch, ONNX Runtime | Train flexibly; deploy a compact, portable inference model. |
| Work queue | Redis + Celery/RQ | Isolates long file analysis from interactive requests. |
| Data | MinIO/S3-compatible object store + PostgreSQL | Large binaries stay in object storage; queryable provenance stays relational. |
| Security | offline-capable deployment, RBAC, audit events, encrypted storage | Appropriate for sensitive authorised recordings. |

## Demo acceptance tests

1. Correctly parse a WAV and a SigMF capture with visible metadata.
2. Rank the correct raw format first on held-out `cu8` and `cs16_le` fixtures.
3. Display bandwidth and a symbol-rate candidate with the plot that supports it.
4. Distinguish supported BPSK/QPSK/2FSK examples at defined SNR levels, reporting uncertainty below threshold.
5. Export an immutable JSON report containing versions, assumptions, inputs, measurements, model outputs, and analyst override history.
