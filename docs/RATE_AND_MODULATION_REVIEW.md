# Sampling frequency, symbol rate and modulation: review and implementation

The SIH sampling-frequency target must be separated from symbol timing. A sampled sequence x[n] has no seconds attached: rescaling both physical symbol rate Rs and ADC rate Fs leaves Rs/Fs unchanged. No waveform-only predictor can resolve that ambiguity in general. A receiver-profile classifier can guess among declared Fs values, but that is a prior, not a physical measurement.

| Target | Classical analyst method | Learned method / implementation |
| --- | --- | --- |
| Absolute Fs | Read PCM WAV header or SigMF core:sample_rate; use capture duration N/T, or a known physical symbol/pilot reference | Metadata resolver, not an unconditional regressor. WAV header reader and existing SigMF parser. Unknown raw Fs remains unavailable. |
| Rs/Fs and samples/symbol | Inspect transition/envelope periodicity, autocorrelation, cyclic spectral lines; check harmonics with eye diagrams | New 96-feature softmax baseline predicts SPS 2/4/8/16. Inputs: 32 complex autocorrelation magnitudes and 32-bin square-law/transition-power spectral peaks each. No Fs or generator labels enter inference. |
| Symbol timing phase | Feedforward square-law acquisition; Gardner or Mueller–Müller tracking | Not implemented by this rate classifier. |
| Modulation | CFO correction, constellation/phase/amplitude statistics, normalized cumulants, cyclic features, likelihood tests | Existing DemodAMC time/spectral CNN. Arithmetic mean across windows, full 12-class ranking, chunk agreement, probability threshold and noise abstention. |

## Literature and attribution

- Oerder & Meyr (1988), [Digital filter and square timing recovery](https://doi.org/10.1109/26.1476): square-law cyclostationary timing acquisition. The implemented transition-power peak search is a related heuristic, not a reproduction of their timing-recovery algorithm. Timing overview and original references: [Bertolucci, Cassettari & Fanucci receiver study](https://www.mdpi.com/1424-8220/21/9/2915). Gardner and Mueller–Müller are tracking algorithms, not standalone absolute Fs estimators.
- Mosquera, Scalise & López-Valcarce (2008), *Non-Data-Aided Symbol Rate Estimation of Linearly Modulated Signals*, [DOI 10.1109/TSP.2007.907888](https://doi.org/10.1109/TSP.2007.907888): cyclostationarity/pulse-based maximum-likelihood coarse/fine search. Useful future deterministic baseline; not implemented here. [Author institution record](https://portalcientifico.uvigo.gal/documentos/5fa0838b299952440aaebd65?lang=fr).
- Zhang et al. (2024), *Deep Learning-Based Blind Estimation of Symbol Rate*, [DOI 10.1109/ICSP62122.2024.10743623](https://doi.org/10.1109/ICSP62122.2024.10743623): candidate raw-IQ ResNet direction. Publisher full text was inaccessible in this review; no architecture reproduction or reported numerical results are claimed. Our feature softmax is our own baseline, not that paper's model.
- Swami & Sadler (2000), *Hierarchical Digital Modulation Classification Using Cumulants*, [DOI 10.1109/26.837045](https://doi.org/10.1109/26.837045): classical cumulant baseline; recreation already in research harness. Publisher full text inaccessible in this review.
- O'Shea, Corgan & Clancy (2016), [Convolutional Radio Modulation Recognition Networks](https://arxiv.org/abs/1602.04105): temporal IQ CNN versus expert-feature classification.
- O'Shea, Roy & Clancy (2018), [Over the Air Deep Learning Based Radio Signal Classification](https://arxiv.org/abs/1712.04578): learned versus higher-order-feature baselines and simulation/over-the-air training differences. Supports requiring recorded-signal evaluation.
- [SigMF specification](https://sigmf.org/): explicit sample rate metadata. [Python wave](https://docs.python.org/3/library/wave.html): PCM WAV frame rate and channels.

## Reproducible scope and validation

Run `python research/train_symbol_rate.py`. 1,000 in-memory training examples (seed 26147), 400 evaluation examples (seed 826147), existing vectorized signal_sim generator, PSK/QAM, RRC and rectangular pulses, CFO ±0.2 cycles/sample, hardware augmentation probability 0.3, 5–30 dB SNR. Only recipe, weights and metrics are saved. Training and test share a generator; this is not distribution-held-out evaluation. See `research/results/symbol_rate_results.json` for measurements. The 0.8 softmax threshold is uncalibrated; low autocorrelation also triggers abstention. Unsupported fractional SPS, different pulses, FSK/OFDM/analogue and low SNR require separate evaluation and retraining. An unsupported signal can still receive a confident wrong answer.

`POST /rates` runs without Fs and returns normalized evidence, or reads a PCM WAV header. `POST /analyse` adds rate evidence alongside existing DSP/modulation results when Fs is provided. Web Capture offers rate evidence before Hz analysis. WAV rate metadata is available automatically; full WAV waveform analysis is still outside this web path.

DemodAMC weights are unchanged. `python research/eval_capture_amc.py` evaluates the new aggregation with seed 526147: 120 independent simulated captures, 110 signals plus 10 noise captures; 92/110 signals accepted and 92/92 correct, 10/10 noise captures abstained. This small same-generator test is not calibration or real-data evidence. Earlier research AMC accuracy refers to the original evaluation protocol. Full ranking and chunk agreement are returned in JSON.

SPS results: 400/400 correct, probability-threshold coverage 396/400. A bundled burst capture is estimated at 8 SPS after selecting an energetic window, rather than quiet capture-leading samples. The selection's start/count are recorded. Known recording duration uses the entire capture sample count before region selection. Rate candidates are not automatically applied as receiver timing; the analyst retains manual override.
