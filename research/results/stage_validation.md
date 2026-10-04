# Frozen-stage synthetic validation — 2026-10-05

This is a fresh local comparison, not a published-benchmark or real-intercept result. All product weights, calibration and abstention thresholds were frozen; SHA-256 hashes were checked before and after evaluation. Only the simple nearest-centroid baseline was fitted, on a separate 600-capture training cohort. New test seeds differ from the recorded train/validation/test seeds. The waveform and code generators remain the same as training, so generator-domain generalization is **not** established.

## 1. Parameter estimation

| Method | Correct SPS class, 95% Wilson CI | Paired improvement |
| --- | --- | --- |
| Learned SPS | 100.00% (400/400); CI 99.05–100.00% | +35.00 pp; CI +30.25–+39.75 |
| Transition-power top peak, quantized | 65.00% (260/400); CI 60.20–69.51% | baseline |

SPS cohort: 400 linear PSK/QAM captures, 2048 samples, 5–30 dB, RRC/rectangular pulses, integer SPS 2/4/8/16 and 30% hardware-impairment probability. Baseline quantization uses the same four-class universe; it does not resolve harmonics with ground truth. This is a simple spectral-peak baseline, not every classical cyclostationary/timing-recovery method. **Absolute sampling frequency Fs was not predicted or validated.** Metadata or an independent time reference is still necessary.

| CFO range, cycles/sample | Method | RMSE, cycles/sample | Within ±0.005, 95% CI |
| --- | --- | ---: | --- |
| wide [-0.2, 0.2] | SpecCFO_v2 | 0.002106 | 96.50% (193/200); CI 92.95–98.29% |
| wide [-0.2, 0.2] | Kay_raw | 0.016877 | 49.50% (99/200); CI 42.65–56.37% |
| wide [-0.2, 0.2] | blind_multi_M | 0.101965 | 61.50% (123/200); CI 54.60–67.97% |
| wide [-0.2, 0.2] | known_M_periodogram_oracle | 0.110966 | 64.50% (129/200); CI 57.65–70.80% |
| narrow [-0.025, 0.025] | SpecCFO_v2 | 0.002782 | 97.50% (195/200); CI 94.28–98.93% |
| narrow [-0.025, 0.025] | Kay_raw | 0.005881 | 68.00% (136/200); CI 61.25–74.07% |
| narrow [-0.025, 0.025] | blind_multi_M | 0.016214 | 81.50% (163/200); CI 75.54–86.27% |
| narrow [-0.025, 0.025] | known_M_periodogram_oracle | 0.010931 | 90.50% (181/200); CI 85.64–93.83% |

CFO cohorts: independent 200-capture cohorts, linear PSK/QAM, 1024 samples, RRC pulses and 5–30 dB. ±0.005 is a declared normalized-frequency tolerance, not a modulation accuracy or universal engineering requirement. Multiply by supplied Fs for Hz. The known-M periodogram is explicitly an oracle baseline; its wide-range aliasing is not proof that ML universally beats classical estimation. These results do not estimate RF centre frequency.

## 2. Current API modulation classifier

| Method | Closed top-1 accuracy, 95% Wilson CI | Paired improvement |
| --- | --- | --- |
| DemodAMC v2, current API aggregation | 95.00% (114/120); CI 89.52–97.69% | +65.83 pp; CI +56.67–+74.17 |
| Separate-training nearest centroid | 29.17% (35/120); CI 21.78–37.84% | baseline |

Cohort: 120 captures, ten per each of 12 classes, 4096 samples, 5–25 dB, RRC/rect, CFO ±0.2, SPS 2–16. CFO is estimated, never supplied from test truth. The current API averages chunk probabilities and applies its frozen abstention rules.

| Linear-only subset (50 captures) | Accuracy |
| --- | --- |
| DemodAMC, still choosing among 12 classes | 90.00% (45/50); CI 78.64–95.65% |
| Five-class cumulant, raw | 24.00% (12/50); CI 14.30–37.41% |
| Five-class cumulant, estimated CFO removed | 76.00% (38/50); CI 62.59–85.70% |

Cumulant comparison is restricted to its five supported linear classes; its theoretical constellation reference does not include a full matched-filter/timing-recovery receiver. It is not comparable to the full-cohort total. Capture-level errors and their labels are retained in the JSON rather than hidden by abstention.

## 3. FEC and interleaver candidate identification

| Method | Joint top-1 accuracy, 95% Wilson CI | Paired improvement |
| --- | --- | --- |
| Frozen parity-feature MLP | 99.71% (339/340); CI 98.35–99.95% | +4.41 pp; CI +2.35–+6.76 |
| Hard minimum-syndrome baseline | 95.29% (324/340); CI 92.49–97.08% | baseline |

340 streams, 3360 bits each: four captures per joint class per BER 0/1/3/5/8%. Sixteen supported code/interleaver combinations plus iid uncoded bits; known 420-bit matrix boundaries. Errors are iid bit flips, not bits from an impaired RF receiver. This validates the post-receiver identifier in isolation, not end-to-end performance.

| Supported-only component (320 streams) | Learned | Hard syndrome baseline |
| --- | --- | --- |
| code | 99.69% (319/320); CI 98.25–99.94% | 95.00% (304/320); CI 92.03–96.90% |
| interleaver | 99.69% (319/320); CI 98.25–99.94% | 95.00% (304/320); CI 92.03–96.90% |

A preselected 85-stream paired subset gives MLP 98.82% (84/85); CI 93.63–99.79%, GF(2) rank-deficiency baseline 42.35% (36/85); CI 32.40–52.96%; +56.47 pp; CI +45.88–+67.06. This rank baseline is **not** a reproduction of the full published noise-tolerant interleaver algorithm.

## 4. Abstention, kept separate from prediction accuracy

| Stage | Accepted / total | Coverage | Accepted accuracy | Accepted accuracy 95% CI |
| --- | ---: | ---: | ---: | --- |
| SPS | 394/400 | 98.50% | 100.00% | 99.03–100.00% |
| AMC, including noise | 93/120 | 77.50% | 98.92% | 94.16–99.81% |
| Joint FEC/interleaver | 304/340 | 89.41% | 100.00% | 98.75–100.00% |

Noise-class predictions are correct in closed-set AMC scoring but are deliberately rejected by the API. FEC unknown predictions are likewise not accepted. 100% observed selective accuracy is not a guarantee. Confidence intervals quantify sampling uncertainty within this generator; they do not account for simulator bias. Paired improvement intervals use 5000 capture bootstrap resamples, not independent intervals subtracted from each other.

## 5. What the research comparisons do and do not establish

| Reference / benchmark | Evidence status | Honest conclusion |
| --- | --- | --- |
| [O'Shea et al., CFO estimator, 2017](https://arxiv.org/abs/1707.06260), [Chen et al., 2023](https://arxiv.org/abs/2311.16155) | Local CNN recreations and archived shared-cohort results in `tables.md`; fresh replay status below. | Not original published training budgets/datasets. |
| [O'Shea et al., AMC, 2016](https://arxiv.org/abs/1602.04105), Rajendran et al., 2018 | VT-CNN2/LSTM recreations are explicitly undertrained; fresh replay status below. | A local checkpoint win is not a published-benchmark win. |
| [Ma et al., DBFCNN, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12881466/) | Seven-family study; external dataset not acquired; our four-code universe differs. | No numerical SOTA comparison is defensible. |
| [Swaminathan et al., interleavers, 2017](https://doi.org/10.1109/ACCESS.2017.2684189), [Singh et al., subspace codes, 2026](https://arxiv.org/abs/2601.15903) | Full algorithms not reproduced. | Our simple rank test is not their benchmark. |

**Counterexamples to a universal improvement claim:** the archived broad-SNR CFO report gives v2 better RMSE (0.0165 vs v1 0.0209) but worse ±0.005 success (83.5% vs 86.5%). Archived narrow-range Kay RMSE 0.00798 is slightly better than v2 0.00800. The old no-CFO AMC control also beats v2 in-distribution (77.7% vs 76.2%). These are historical conditions, not the fresh easier 5–30 dB cohorts above.

Remaining validation: independent real captures; full published benchmark datasets and matched training budgets; continuous/noninteger SPS; unseen pulses and protocols; synchronization/bit-mapping errors; arbitrary interleavers; FEC decoding, BER of recovered payloads and end-to-end CRC checks. No encryption/decryption validation exists.

## Reproduce and inspect

```powershell
python research/validate_stages.py
python research/report_stage_validation.py
python -m unittest discover -s tests
```

NumPy-only execution. `stage_validation.json` retains recipes, seeds, source and model hashes, confusion matrices, capture-level predictions and paired intervals; no raw synthetic signal files are saved. Logs show original product training reached 4000 SpecCFO v2 steps and 2500 DemodAMC v2 steps; this validation does not resume training or tune test-set thresholds.

## Fresh replay of local paper recreations

The CPU JAX evaluator regenerated the exact recipes above, verified test labels and froze checkpoint hashes. CFO networks receive the identical 1024-sample cohorts; window AMC models see all 32 disjoint 128-sample windows of each 4096-sample capture, aggregated by mean log probability as in the repository evaluator. No model gets oracle test CFO.

| CFO cohort | Recreation | RMSE | Within ±0.005 | SpecCFO v2 paired gain |
| --- | --- | ---: | --- | --- |
| wide | iq_resnet | 0.014664 | 33.00% (66/200); CI 26.86–39.78% | +63.50 pp; CI +56.00–+70.50 |
| wide | oshea_cfo | 0.005618 | 68.00% (136/200); CI 61.25–74.07% | +28.50 pp; CI +22.00–+35.00 |
| narrow | iq_resnet | 0.014201 | 29.00% (58/200); CI 23.15–35.64% | +68.50 pp; CI +62.00–+75.00 |
| narrow | oshea_cfo | 0.005465 | 68.00% (136/200); CI 61.25–74.07% | +29.50 pp; CI +23.50–+36.00 |

| AMC recreation | Accuracy | Current DemodAMC paired gain |
| --- | --- | --- |
| vtcnn2 | 24.17% (29/120); CI 17.39–32.55% | +70.83 pp; CI +62.50–+79.17 |
| lstm_ap | 25.83% (31/120); CI 18.84–34.33% | +69.17 pp; CI +60.83–+77.50 |

The very large AMC margins are against the saved undertrained checkpoints (2000 steps, loss about 2.00 and 1.85), not evidence of beating adequately trained VT-CNN2/LSTM or the papers. Fresh recipe replay does not remedy unequal original training budgets or acquire a published benchmark. Full predictions, checkpoint hashes and training metadata are in `stage_paper_recreations.json`.

Optional replay: install `jax[cpu]` and `optax`, then run `python research/validate_paper_recreations.py` and regenerate this report.
