# DEmod

**DEmod** is an analyst-facing, transparent signal-analysis workbench for SIH 2026 problem statement **SIH26147**: *Automated model for analysis of `.IQ` and `.wav` files along with signal parameter extraction* (NTRO).

## What the first MVP does

The [FEC/interleaver literature and implementation review](docs/FEC_AND_INTERLEAVER_REVIEW.md) documents the conventional analyst process and 2014–2026 research. The receiver now compares hard parity evidence against a trained **FECFeatureMLP** on full candidate bits (not the 512-bit preview), with candidate ranking, abstention, model/input hashes and an analyst frame-offset setting. Its deliberately small universe covers repetition-3, Hamming(7,4)/(15,11), convolutional K=3 (7,5 octal), and no interleaving or three known 420-bit matrix permutations. Training uses ephemeral recipe-generated bits; weights and aggregate metrics are shipped. **This is simulation-only candidate identification, not verified frame/permutation recovery, FEC decoding, arbitrary code reconstruction or decryption.** Published CNN datasets/networks have not been benchmark-reproduced. Use **Load coded BPSK example** then **Run demodulation** for a controlled integration demo. See `research/results/fec_results.json` for baseline comparisons, confidence intervals and failure/stress results.

On 1700 independent-seed synthetic tests (17 classes; 3360 bits; BSC BER 0–8%), the hybrid achieved **99.82% joint class accuracy** versus **95.12%** for the minimum hard-syndrome baseline. Acceptance coverage was **89.24%**; observed accepted-class accuracy was 100% with a Wilson 95% interval of 99.75–100%. These are same-generator candidate-set numbers, not operational guarantees or comparisons to published seven-family CNN accuracy. Shifted-frame/bit-slip stress cases can still match the correct class without establishing valid framing. [Measured tables](research/results/fec_tables.md) include all baselines and stress results.

The [WAV input workflow](docs/WAV_INPUT.md) is now connected to the web API and guided receiver. PCM WAV header fields populate Fs automatically; declared stereo I/Q maps channels to complex samples, and declared mono RF/IF uses an analytic-signal conversion with optional IF translation. Declared audio receives local playback, level/spectral overview and RF abstention. The original WAV hash, header fields, conversion and analyst channel-role hypothesis stay in reports. Use **Load WAV example** for the bundled synthetic fixture. PCM 8/16/24/32-bit is supported; malformed, truncated, compressed/float WAV and over-limit files receive explicit errors. Protocol/audio intelligence is still pending.

The [sampling-rate and modulation review](docs/RATE_AND_MODULATION_REVIEW.md) now maps classical methods to executable estimates. `POST /rates` and the Capture rate-evidence control work before a raw IQ sample rate is supplied; PCM WAV header Fs is read automatically, and normalized symbol-rate candidates are compared with a newly trained feature-based SPS model (2/4/8/16). A capture-log duration enables `Fs = sample_count / duration`, with per-field provenance and conflict rejection on analysis. The reproducible recipe and synthetic evaluation live in `research/train_symbol_rate.py` and `research/results/symbol_rate_results.json`. Absolute raw Fs still needs metadata or a known time reference. Modulation inference now returns all 12 candidates, arithmetic-mean window probabilities, chunk agreement and noise abstention; `research/eval_capture_amc.py` records a new 120-capture simulation smoke evaluation. The old benchmark numbers do not validate this new capture aggregation rule.

- Reads mono or multi-channel PCM WAV recordings with Python's standard library.
- Reads raw complex IQ samples when their format is explicitly supplied: `s16le`, `s16be`, `s8`, `cu8`, `f32le`, `f32be`.
- Produces reproducible, explainable DSP measurements: DC offset, RMS, peak, crest factor, instantaneous-frequency statistics, FFT peak, 99%-energy occupied bandwidth, and ranked modulation-family hypotheses.
- Emits a machine-readable JSON report that a dashboard and later ML classifier can consume.

It deliberately does **not** claim universal blind recovery of FEC, interleaving, encryption, or every unknown waveform. Those are research-level SIGINT problems and must be demonstrated only for supported signal families with recorded validation data.

## Frozen-model validation (2026-10-05)

The [three-stage validation report](research/results/stage_validation.md) records fresh independent-seed synthetic comparisons with unchanged product weights: SPS 400/400 versus spectral-peak baseline 260/400; current capture-level AMC 114/120 versus separately fitted centroid 35/120; joint known-candidate FEC/interleaver 339/340 versus hard-syndrome baseline 324/340. Improvements are respectively +35.00, +65.83 and +4.41 percentage points **on those specific synthetic cohorts**, not published SOTA or real-intercept claims. Separate CFO sweeps, confidence intervals, abstention coverage, per-component FEC/interleaver scores and limitations are included. AMC accepted one incorrect prediction; its selective accuracy is not perfect. Absolute raw sampling frequency, FEC decoding and decryption remain unvalidated/unimplemented as previously documented.

`research/validate_stages.py` regenerates ephemeral examples and retains only recipes, hashes, predictions, confusion matrices and paired bootstrap intervals. `research/report_stage_validation.py` renders the report. Model SHA-256 checks prevent accidental recalibration during evaluation; the old evaluator that fits temperature is not used. The report also audits the O'Shea/Chen/Rajendran recreations and the 2026 DBFCNN and subspace-code literature without equating narrow candidate recognition with general blind recovery. Four scoring regression tests distinguish closed accuracy from accepted-only accuracy and account for unknown predictions in confusion matrices.

The fresh [research-model replay](research/results/stage_paper_recreations.json) also runs the saved IQ-ResNet/O'Shea CFO and VT-CNN2/LSTM AMC recreations on the same test captures. For example, wide-range CFO within ±0.005 cycles/sample is 96.50% for SpecCFO versus 68.00% for the O'Shea recreation; AMC is 95.00% versus 24.17%/25.83% for the saved VT-CNN2/LSTM. These large margins are **not published research wins**: the baseline checkpoints were undertrained, their budgets differ, and no original external benchmark was acquired. Replay uses isolated CPU JAX/optax dependencies and leaves all weights untouched.

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

The [HF/VHF/UHF recipe corpus](data/recipes/sih-hf-vhf-uhf.json) is the current proxy for SIH26147: it produces BPSK/QPSK/16QAM and OFDM-shaped examples with deterministic seeds and receiver/channel impairments. It includes centre-frequency and sample-rate truth only as hidden generator audit metadata. That corpus still labels FEC/interleaving `none_mvp`; the separate [coded-bit recipes](data/recipes/fec-recipes.json) and coded BPSK fixture now support limited experimental identification. They do not establish performance on real HF/VHF/UHF intercepted coding schemes.

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

### Learned carrier-offset estimator and modulation classifier

The first-generation learned models were replaced after testing showed they were not usable outside their training recipe: the TinyMLP returned about -595 kHz for a true +1.5 kHz offset on a 250 kS/s capture, and the centroid classifier called an FSK capture QPSK. The replacements, and the research behind them, are in [`research/README.md`](research/README.md).

- **SpecCFO** estimates carrier offset in cycles per sample from the spectra of x, x^2, x^4 and x^8 with a circular convolutional network, then refines the result with a periodogram. It reports a confidence and falls back to a posterior mean when unsure.
- **DemodAMC** classifies 12 classes (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2-FSK, 4-FSK, GMSK, OFDM, AM, FM, noise only) after removing the carrier offset, with calibrated probabilities and abstention.

Evaluated on simulated signals against recreated baselines from Chen et al. (2023), O'Shea et al. (2017 and 2016), Rajendran et al. (2018), Kay, Luise-Reggiannini and Swami-Sadler:

- SpecCFO has the lowest overall RMSE on the +-0.2 cycles/sample sweep (0.0165 against 0.0216 for the O'Shea CNN and 0.0357 for IQ-ResNet), the lowest median error (15 times below the CNN), and 64% of estimates within the README's 250 Hz tolerance on the repo's own recipe signals against 18% for the shipped TinyMLP.
- DemodAMC reaches 76% over 12 classes against 27 to 35% for the recreated VT-CNN2 and LSTM, with 95 to 96% accuracy on the 62 to 63% of captures it does not abstain on.
- It does not win everywhere: it is no better than Kay's estimator on the narrow-range set, loses to the O'Shea CNN at 0 dB on that set, and is weak on 8PSK carrier offset at one sample per symbol. The paper baselines were trained for minutes on a CPU, so the margins are inflated. Everything is simulation only.

The API loads the model files from `data/models/` and falls back to the original models when they are absent. Training and evaluation are resumable (`research/run_all_training.sh`, `research/run_v2_training.sh`), and `research/train_on_free_gpu.ipynb` runs them on a free Colab or Kaggle GPU.

A defect in `src/generate_synthetic.py` was found along the way: it draws a new random symbol on every sample, so `samples_per_symbol` is ignored. It is documented by an expected-failure test and not yet fixed, because fixing it requires retraining the first-generation models.

### Receiver MVP and GNU Radio graphs

The **Run AMC-guided receiver** action connects DemodAMC → confident SPS estimate → SpecCFO correction → the supported receiver on the selected analysis region. `/demodulate` accepts omitted SPS/CFO headers; supplied values remain manual overrides, including an explicit zero CFO. Unsupported modulation, abstained rate estimates and low-confidence CFO abstain. Automatic SPS supports linear PSK/QAM; FSK requires manual SPS. Fixed or static-search integer timing/phase can be selected. Reports retain applied settings, model/analyst sources, input hash and selected sample range. Output is candidate hard decisions, not decoded payload.

`POST /demodulate` now uses **Komm 0.36.0 + SciPy 1.18.1** for BPSK/QPSK/8-PSK, 16/64-QAM and unshaped 2/4-FSK, retaining the old three-mode `legacy_fixed` fallback. Install `requirements-receiver.txt`. Receivers apply rectangular/RRC filtering, supplied/estimated CFO correction, optional static integer-timing and phase search, explicit bit mapping and library hard decisions. `src/receive_capture.py` is the simple IQ/WAV → full candidate-bit JSON CLI; the UI shows correct constellation/tone thresholds and configuration provenance. See the [tooling and literature review](docs/DEMODULATION_TOOLING.md), [controlled BER/SER](research/results/receiver_results.md), and [optional GNU Radio runner](gnuradio/README.md). GNU Radio continuous tracking remains unexecuted here; Python does not implement fractional/clock-drift recovery, equalisation, framing, FEC decoding or decryption.

Receiver validation generates 192 ephemeral captures (48 conditions × four captures): seven modes, rectangular/RRC as applicable, 15/25 dB, supplied +750 Hz CFO, known tone levels and manual/static-auto acquisition. BER is measured against transmitted bits with a fixed edge exclusion, never truth-guided realignment. 64-QAM at 15 dB has nonzero errors; the report records them. The intentional 90-degree QPSK counterexample demonstrates that low-EVM automatic phase search does not resolve absolute bit mapping. All modes remain simulation-validated only.

### Analyst UI (connected to the local API)

For the live demo, start `python src/local_comparison_api.py` and open `http://127.0.0.1:8787/ui/`. The server loads the shipped NumPy SpecCFO/DemodAMC model files directly. Its retained legacy TinyMLP/centroid fallback now uses a compact in-memory batch by default so that local startup is suitable for a demonstration; use `--examples` to raise that fallback-only batch size when investigating legacy comparisons.

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
- **Receiver:** runs the seven declared PSK/QAM/FSK modes, exposes pulse/rolloff/phase/static timing/tone hypotheses and shows decisions against actual ideal points or tone thresholds. Output is labelled *candidate hard decisions*, never verified payload or decrypted data.
- **Report:** run ID, browser-side SHA-256 checked against the API's hash, every parameter with its source, and a JSON export that keeps both API responses unchanged.

Plots are scientific instruments modelled on GNU Radio's Qt GUI sinks (`web/plots.js`, no dependencies): a framed plot area with a dotted major grid, 1-2-5 ticks, labelled axes in engineering units, toggleable legends, a crosshair readout rounded to the actual bin resolution, drag-to-zoom and double-click reset.

- **Frequency sink:** Welch average over the whole capture (Hann, 50% overlap), max hold, and the old single FFT of the first block for comparison. Shows RBW (Hann ENBW, 1.5 bins) and switches between baseband offset and an RF axis when a centre frequency is supplied.
- **Waterfall sink:** STFT referenced to the strongest bin across all shown frames, so quiet frames stay dark. GNU Radio multi-colour, white-hot and black-hot maps, a dB colour bar, and segment marks on the time axis.
- **Constellation sink:** equal-aspect I/Q with a polar readout. In the receiver it overlays ideal points on the decisions.
- **Time sinks:** the max-pooled power envelope with the segmentation threshold and baseline, I and Q around the first burst, and the 2-FSK discriminator with its slicer threshold.

The API adds these fields without changing existing ones: `spectrum.welch_power_db`, `spectrum.max_hold_db`, `waterfall.power_db_global`, `segmentation.envelope` and `time_preview`. The single-FFT `power_db` only covers the first `fft_size` samples, which misses bursts that start later; the Welch trace fixes that.

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
