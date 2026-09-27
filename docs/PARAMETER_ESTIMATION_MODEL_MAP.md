# One-to-one map: manual RF estimation to learned model

## Rule

Each learned target must correspond to a defined physical quantity, a classical estimator, an exact/generated or independently measured label, and a task-appropriate error metric. The ML output is compared with—not substituted for—the deterministic estimate.

## Parameter map

| Parameter/task | Manual/DSP estimator | Validity boundary | Learned counterpart | Target and evaluation |
| --- | --- | --- | --- | --- |
| Raw sample representation | Header/SigMF parse; byte-length/pair completeness; amplitude/clipping plausibility across layouts | Cannot prove an unlabelled layout from bytes alone | Small format-ranker using representation-plausibility features | Categorical layout only on generated/known-format files; top-k accuracy and abstention |
| Sample rate `Fs` | Read container/SigMF/receiver metadata; infer only with a known external feature | Not identifiable in Hz from headerless baseband alone | **No raw-IQ regression target** | Return metadata/hypothesis/unknown, never fabricated truth |
| RF centre frequency | Read capture metadata; use known beacon/reference if present | Baseband offset is not absolute RF centre frequency | **No raw-IQ regression target** | Return metadata/hypothesis/unknown |
| Signal presence/burst timing | Smoothed power, robust threshold, STFT connected components, matched/correlation detection if known | Energy alone cannot classify a burst | 1-D temporal segmentation CNN only after baseline exists | Per-sample/segment mask; IoU, onset/offset error, false alarms |
| Bandwidth | Windowed FFT/PSD edges and 99%-energy occupied bandwidth | Definition/window affects result | Feature regressor as a cross-check | Hz error against generated/instrumented bandwidth; report definition |
| Baseband carrier offset | Spectral peak, phase-increment; M-th-power method within PSK/QAM branch | Carrier/modulation/hopping can be confounded | Raw-IQ + feature regressor, family-conditioned | Hz error; no RF centre-frequency claim |
| Symbol rate / samples per symbol | Cyclostationary spectral lines; Oerder–Meyr squared-signal acquisition | Needs adequate observation, oversampling and compatible linear modulation | Multi-task CNN: log symbol rate + `sin(2πφ)`,`cos(2πφ)` | Relative symbol-rate error; circular phase error |
| Timing phase tracking | Gardner TED (non-data-aided, typically near 2 sps); Mueller–Müller (decision-directed, after decisions) | Not a blind universal acquisition method | Same timing model supplies initial phase; classical loop tracks it | Timing error over held-out capture/channel sessions |
| I/Q imbalance | Widely-linear/covariance or image-rejection estimate | Can be confounded with signal asymmetry | Feature regressor as secondary audit | Gain/phase error, image rejection improvement, no BER regression |
| SNR/quality | Noise-floor or moment/EVM estimate; EVM only after a constellation/timing branch | EVM needs correct branch | Quality classifier/regressor | Calibrated interval, ECE; never use as ground truth without label |
| Modulation family | Likelihood/HOC/cyclic features, amplitude/phase/frequency statistics | Multi-signal and low-SNR windows require abstention | 1-D residual CNN fused with DSP features | Macro-F1, calibration, abstention and session-held-out generalisation |
| FEC/interleaver | Sync, soft bits, parity/CRC, candidate code tests | Impossible before credible bitstream | Later bounded hypothesis ranker, not end-to-end labeler | CRC/parity/decoder evidence; no raw-IQ FEC target |

## First training task: symbol timing

### Manual baseline

1. **Acquisition:** Oerder–Meyr’s digital filter-and-square method uses timing information in the squared signal. It applies to synchronous linear modulation such as PAM/QAM/PSK and is designed for digital, efficient high-rate recovery. [Oerder & Meyr, 1988](https://doi.org/10.1109/26.1476)
2. **Non-data-aided tracking:** Gardner’s BPSK/QPSK timing-error detector updates the sampling phase from an error signal when the receiver has an appropriate oversampled stream. [Gardner, 1986](https://doi.org/10.1109/TCOM.1986.1096561)
3. **Decision-directed tracking:** Mueller–Müller uses baud-rate decisions and is therefore downstream of a provisional constellation/decision stage, not the first blind estimator. [Mueller & Müller, 1976](https://doi.org/10.1109/TCOM.1976.1093326)

### Learned counterpart

Train one small shared encoder, not a general “DSP model”:

```text
normalised I/Q window ─> 1-D residual CNN ─┬─> log(symbol_rate) regression
                                            ├─> sin(2π timing_phase)
manual timing/cyclic features ─> MLP ──────┴─> cos(2π timing_phase)
```

- **Targets:** generator-hidden symbol rate and fractional timing phase. Store phase as sine/cosine so `0` and one-symbol period are adjacent.
- **Loss:** Huber loss on log-rate plus squared error on the unit-circle phase representation; use a unit-norm penalty.
- **Training data:** generated BPSK/QPSK/8PSK/16QAM, pulse shapes, `Fs`/baud ranges, CFO, AWGN, multipath and I/Q impairment. The model receives only IQ and permitted task labels—not recipe fields.
- **Acceptance:** beat neither method by assertion. Compare the model, Oerder–Meyr acquisition and a Gardner tracking branch on capture-session-held-out synthetic data, then authorised real data with known timing. Report median/95th-percentile relative-rate error, circular phase error, lock rate and confidence calibration.
- **Abstention:** reject OFDM, FSK, unknown `Fs`, low-quality/multi-signal windows, and uncertainty/disagreement beyond calibrated thresholds. Route them to family-specific branches.

## Why this is the correct ML role

The neural model learns nuisance invariance and can provide a fast initial estimate; DSP provides the physical definition, reproducibility and lock criterion. A dual estimate exposes a bad input-format hypothesis or out-of-distribution signal rather than amplifying it into a confident false decode.
