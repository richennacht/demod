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

This follows PySDR’s [IQ Files and SigMF chapter](https://pysdr.org/content/iq_files): raw complex recordings are conventionally interleaved `I,Q,I,Q…`; 16-bit integer IQ consumes four bytes per complex sample, while `complex64` consumes eight. That permits a sample **count** after a format is known, but not an absolute sample rate from file size alone. The same reference motivates DEmod’s SigMF-sidecar strategy and future saturation/clipping flag.

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

## DSP-to-model training map

DEmod will train narrow models for named physical parameters, each paired with a manual estimator and a calibrated disagreement/abstention rule—not one opaque “DSP AI.” The first task is symbol timing: Oerder–Meyr acquisition, Gardner tracking and Mueller–Müller decision-directed tracking provide the classical baselines; a compact raw-IQ/feature-fusion regressor predicts the same rate/phase targets only where their labels exist. The full one-to-one map, data targets and acceptance metrics are in [parameter-estimation model map](docs/PARAMETER_ESTIMATION_MODEL_MAP.md).

### Directly testable DSP-versus-model comparison

`src/local_comparison_api.py` now makes the first comparison testable against an authorised `.iq` capture: it holds a raw byte upload only in memory, returns deterministic DSP features and FFT/STFT evidence, and puts the existing synthetic-trained MLP's DC-I, DC-Q and coarse-CFO outputs beside their manual estimates. It binds to localhost by default, imposes a 16 MiB experiment cap, and does not pretend that Vercel/GitHub Pages should receive sensitive or terabyte-scale RF recordings. The full request format, scope table, provenance fields and papers are in the [comparison-harness guide](docs/COMPARISON_HARNESS.md).

The local comparison API now also produces an immutable input-provenance record (SHA-256 and each metadata field's source), raw and optional-derived denoising branches, FFT/STFT constellation and burst evidence for both branches, named manual parameter estimates, and a synthetic-trained modulation-classification baseline with ranked candidates and abstention. Set `X-DEmod-Denoise-Profile` to `raw` (default), `dc_only`, or `dc_and_impulse`; this never overwrites raw IQ. These models are baseline evidence tools, not field-calibrated blind decoders. The [build matrix](docs/CURRENT_BUILD_MATRIX.md) records this precise completion boundary.

### Receiver MVP and GNU Radio graphs

The local API also exposes `POST /demodulate` for controlled BPSK, QPSK and 2-FSK recordings. It requires an analyst-supplied samples-per-symbol value and returns reproducible hard-bit candidates, decision/EVM evidence and a GNU Radio graph descriptor. GNU Radio is preferred when installed for interactive frequency, timing and constellation nodes; the supplied Python receiver is only the tested fallback for declared parameters. It does not claim FEC, framing, payload recovery or blind synchronisation. See the [receiver MVP contract](docs/MVP_RECEIVER.md) and [GNU Radio graph notes](gnuradio/README.md).

### Analyst UI (connected to the local API)

`web/` is now a working front end for the local API, not a byte preview. Start the API and open the UI it serves:

```powershell
python -m pip install -r requirements-dsp.txt
python src/local_comparison_api.py
# then open http://127.0.0.1:8787/ui/
```

The hosted page at <https://richennacht.github.io/demod/web/> can also talk to a loopback API (the API answers Chromium's private-network preflight), but Safari blocks a secure page calling `http://127.0.0.1`, so the locally served `/ui/` is the dependable path. Captures are only ever sent to the API address shown in the sidebar.

What it does:

- **Capture:** load an `.iq` file (or a SigMF `.sigmf-meta` to fill format, rate and centre frequency), set IQ format, sample rate, centre frequency, gain, the source of those values, the cleaning profile and receiver settings. Every value is sent and recorded as a hypothesis with its source. A built-in generator makes a QPSK test capture with known truth for checking estimators end to end.
- **Evidence:** backend FFT, STFT waterfall with energy segments overlaid, IQ scatter and segment timeline, switchable between the raw and derived branch; manual estimates (with one-click hand-off of CFO and symbol-rate candidates to the receiver); manual-versus-model comparison that flags disagreement and model values outside ±fs/2; classifier ranking with confidence or abstention; the cleaning audit; and the API's own stated limitations.
- **Receiver:** runs `POST /demodulate` for BPSK, QPSK and 2-FSK only, then shows decisions against ideal points, EVM, the GNU Radio block graph and the API's limits. Output is labelled *candidate hard decisions*, never decoded, validated or decrypted data.
- **Report:** run ID, browser-side SHA-256 checked against the API's hash, every parameter with its source, and a JSON export that keeps both API responses unchanged.

`web/examples/` holds a synthetic QPSK burst capture and the API responses recorded from it, so the hosted page can show real output with no local API. It is labelled as synthetic everywhere it appears.

This change also removed two accidental O(n²) loops (a median and two means recomputed per element in `denoise_iq.py` and `feature_analysis.py`). A 30,000-sample `/analyse` call dropped from about 4 minutes to about 5 seconds with byte-identical output, and a regression test guards it.

## Research basis

The following papers directly motivate the currently documented technology choices. They are linked here so the implementation, evidence boundary and source material remain together.

- DC-offset/CFO/IQ-imbalance estimation and compensation: [Liu & Li, 2011](https://doi.org/10.1016/j.sigpro.2010.12.002), [Song et al., 2017](https://arxiv.org/abs/1712.05970), and [Wang et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5751594/).
- Selective impulse/RFI mitigation rather than blanket smoothing: [Hwang et al., 2017](https://doi.org/10.1587/transfun.E100.A.3041), [Nita & Gary, 2010](https://digitalcommons.njit.edu/fac_pubs/13405/), and [Taylor et al., 2018](https://arxiv.org/abs/1808.10365).
- Optional wavelet denoising, subject to decoder-level validation: [Baxter & Upton, 2002](https://doi.org/10.1111/1467-9876.00276).
- Classical modulation analysis: [Dobre et al., 2007](https://doi.org/10.1049/iet-com:20050176), [Abdelmutalab et al., 2016](https://doi.org/10.1016/j.phycom.2016.08.001), and [Dobre et al., 2010](https://doi.org/10.1007/s11277-009-9776-2).
- Blind carrier/symbol-rate candidates through cyclostationarity: [Zhang et al., 2012](https://doi.org/10.1016/j.proeng.2011.12.753) and [Güner, 2014](https://doi.org/10.1002/dac.2606).
- Hybrid raw-IQ/deep-learning model choice and domain-shift cautions: [Thakur & Imtiaz, 2026](https://www.mdpi.com/2079-9292/15/10/2163) and [Tian et al., 2026](https://doi.org/10.1016/j.sigpro.2025.110444).

## Submission differentiator

The concise, submission-ready explanation of DEmod’s evidence graph, implemented technologies, research references, and defensible novelty is in [DEmod technical brief](docs/DEMOD_TECHNICAL_DIFFERENTIATORS.md). The key distinction is not “an AI dashboard”: it is a provenance-preserving chain from raw bytes through representation hypotheses, raw/derived DSP evidence and bounded modulation candidates to either a supported result or an explicit abstention.

## PySDR source audit

DEmod uses PySDR as an educational implementation reference, not as code to copy. The [PySDR research and coding-practice index](docs/PYSDR_RESEARCH_AND_PRACTICES.md) maps relevant chapters to DEmod rules and records the primary external references surfaced by its cyclostationary, noise, metadata, synchronization and detection chapters. The full upstream textbook and source remain linked there under their own CC BY-NC-SA 4.0 licence.

## Delivery roadmap

1. **Baseline (now):** deterministic file ingestion and explainable DSP measures.
2. **Demo capability:** spectrogram/waterfall UI, supported modulation-family classifier, and labelled evaluation set.
3. **Advanced capability:** symbol-rate estimation and demodulation only for explicitly supported families; every result includes confidence and evidence.

The complete, implementation-state-aware build contract is in the [end-to-end workflow specification](docs/END_TO_END_WORKFLOW_SPEC.md). It separates code that works today from required receiver, ML, UI and controlled-deployment work, and defines acceptance evidence for each stage.

For the short, unambiguous inventory of every built, partial and unbuilt module, see the [current build matrix](docs/CURRENT_BUILD_MATRIX.md). It also records how [Gao et al. (2026)](https://doi.org/10.3390/electronics15030674) informs a future complex-autoencoder **noise-robust AMC** branch before conventional demodulation; it is not inaccurately represented as a ready-made demodulator.

## Safety and demo boundary

Use only recordings and datasets that your team is authorized to analyse. The MVP is a file-analysis tool; it does not acquire radio signals or perform operational interception.
