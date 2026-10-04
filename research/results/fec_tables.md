# FEC/interleaver candidate-set evaluation

Simulation only; independent seeds, same encoder generator. Not published-paper benchmark reproduction or real-intercept validation.

| BSC BER | Hybrid joint accuracy | Hard syndrome | Statistics-only MLP | Hybrid acceptance coverage |
| --- | --- | --- | --- | --- |
| 0% | 100.00% | 100.00% | 29.71% | 94.12% |
| 1% | 100.00% | 100.00% | 30.29% | 94.12% |
| 3% | 100.00% | 100.00% | 27.65% | 94.12% |
| 5% | 100.00% | 99.41% | 25.88% | 93.53% |
| 8% | 99.12% | 76.18% | 26.47% | 70.29% |

Overall hybrid: 1700 cases, 99.82% joint accuracy (Wilson 95% interval 99.48%–99.94%), macro F1 0.9982.
Accepted 1517 (89.24%); observed accepted-class accuracy 100.00% (Wilson interval 99.75%–100.00%). Not a guarantee for unseen distributions.

## Identical preselected 85-case subset

| Method | Joint accuracy |
| --- | --- |
| hybrid | 100.00% |
| manual | 95.29% |
| statistics | 28.24% |
| rank | 40.00% |

The rank baseline is a small GF(2) deficiency method; it is not the full noise-tolerant 2017 interleaver algorithm or the 2026 minimum-denoised-subspace method.

## Stress tests

| Stress condition | Accepted / tested | Correct accepted class, if known |
| --- | --- | --- |
| burst_errors | 80 / 80 | 80 |
| misaligned_frame | 72 / 80 | 72 |
| random_permutation | 0 / 80 | not applicable / unknown |
| biased_uncoded | 0 / 80 | not applicable / unknown |
| hamming31 | 0 / 80 | not applicable / unknown |
| whitened | 0 / 80 | not applicable / unknown |
| bit_slips | 80 / 80 | 80 |

**Shifted boundaries and bit slips can retain a correct class hypothesis while invalidating exact deinterleaving/frame recovery.** No frame synchronization was validated, including the accepted stress cases. No decoder or CRC ran.

Source: `fec_results.json`. Reproduce with `python research/train_fec_identifier.py` then `python research/report_fec_results.py`.
