"""Render the frozen-stage results without mixing cohorts or rounding before scoring."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def score(m):
    lo,hi=m['wilson95']
    return f"{100*m['accuracy']:.2f}% ({m['correct']}/{m['n']}); CI {100*lo:.2f}–{100*hi:.2f}%"


def delta(m):
    lo,hi=m['delta_bootstrap95_pp']
    return f"{m['delta_percentage_points']:+.2f} pp; CI {lo:+.2f}–{hi:+.2f}"


def main():
    r=json.loads((ROOT/'research/results/stage_validation.json').read_text(encoding='utf-8'))
    p=r['parameter_estimation']; a=r['modulation']; f=r['fec_interleaver']
    lines=['# Frozen-stage synthetic validation — 2026-10-05','',
      'This is a fresh local comparison, not a published-benchmark or real-intercept result. '
      'All product weights, calibration and abstention thresholds were frozen; SHA-256 hashes were checked before and after evaluation. '
      'Only the simple nearest-centroid baseline was fitted, on a separate 600-capture training cohort. '
      'New test seeds differ from the recorded train/validation/test seeds. The waveform and code generators remain the same as training, so generator-domain generalization is **not** established.','',
      '## 1. Parameter estimation','',
      '| Method | Correct SPS class, 95% Wilson CI | Paired improvement |',
      '| --- | --- | --- |',
      f"| Learned SPS | {score(p['sps']['learned'])} | {delta(p['sps']['paired'])} |",
      f"| Transition-power top peak, quantized | {score(p['sps']['manual_quantized'])} | baseline |",'',
      'SPS cohort: 400 linear PSK/QAM captures, 2048 samples, 5–30 dB, RRC/rectangular pulses, '
      'integer SPS 2/4/8/16 and 30% hardware-impairment probability. '
      'Baseline quantization uses the same four-class universe; it does not resolve harmonics with ground truth. '
      'This is a simple spectral-peak baseline, not every classical cyclostationary/timing-recovery method. '
      '**Absolute sampling frequency Fs was not predicted or validated.** Metadata or an independent time reference is still necessary.','',
      '| CFO range, cycles/sample | Method | RMSE, cycles/sample | Within ±0.005, 95% CI |',
      '| --- | --- | ---: | --- |']
    for sweep in ('wide','narrow'):
        s=p['cfo'][sweep]
        for name,m in s['metrics'].items():
            lines.append(f"| {sweep} {s['recipe']['cfo_range']} | {name} | {m['rmse_cycles_per_sample']:.6f} | {score(m['within_005'])} |")
    lines+=['','CFO cohorts: independent 200-capture cohorts, linear PSK/QAM, 1024 samples, '
      'RRC pulses and 5–30 dB. ±0.005 is a declared normalized-frequency tolerance, not a modulation accuracy or universal engineering requirement. '
      'Multiply by supplied Fs for Hz. The known-M periodogram is explicitly an oracle baseline; its wide-range aliasing is not proof that ML universally beats classical estimation. '
      'These results do not estimate RF centre frequency.','',
      '## 2. Current API modulation classifier','',
      '| Method | Closed top-1 accuracy, 95% Wilson CI | Paired improvement |',
      '| --- | --- | --- |',
      f"| DemodAMC v2, current API aggregation | {score(a['learned'])} | {delta(a['paired_centroid'])} |",
      f"| Separate-training nearest centroid | {score(a['centroid'])} | baseline |",'',
      'Cohort: 120 captures, ten per each of 12 classes, 4096 samples, 5–25 dB, RRC/rect, '
      'CFO ±0.2, SPS 2–16. CFO is estimated, never supplied from test truth. '
      'The current API averages chunk probabilities and applies its frozen abstention rules.','',
      '| Linear-only subset (50 captures) | Accuracy |',
      '| --- | --- |',
      f"| DemodAMC, still choosing among 12 classes | {score(a['linear_subset']['learned'])} |",
      f"| Five-class cumulant, raw | {score(a['linear_subset']['cumulant_raw'])} |",
      f"| Five-class cumulant, estimated CFO removed | {score(a['linear_subset']['cumulant_derotated'])} |",'',
      'Cumulant comparison is restricted to its five supported linear classes; its theoretical constellation reference does not include a full matched-filter/timing-recovery receiver. '
      'It is not comparable to the full-cohort total. Capture-level errors and their labels are retained in the JSON rather than hidden by abstention.','',
      '## 3. FEC and interleaver candidate identification','',
      '| Method | Joint top-1 accuracy, 95% Wilson CI | Paired improvement |',
      '| --- | --- | --- |',
      f"| Frozen parity-feature MLP | {score(f['learned'])} | {delta(f['paired_manual'])} |",
      f"| Hard minimum-syndrome baseline | {score(f['manual'])} | baseline |",'',
      '340 streams, 3360 bits each: four captures per joint class per BER 0/1/3/5/8%. '
      'Sixteen supported code/interleaver combinations plus iid uncoded bits; known 420-bit matrix boundaries. '
      'Errors are iid bit flips, not bits from an impaired RF receiver. This validates the post-receiver identifier in isolation, not end-to-end performance.','',
      '| Supported-only component (320 streams) | Learned | Hard syndrome baseline |',
      '| --- | --- | --- |']
    for name,methods in f['supported_components'].items():
        lines.append(f"| {name} | {score(methods['learned'])} | {score(methods['manual'])} |")
    subset=f['paired_rank_subset']
    lines+=['',f"A preselected 85-stream paired subset gives MLP {score(subset['learned'])}, "
      f"GF(2) rank-deficiency baseline {score(subset['rank'])}; {delta(subset['paired'])}. "
      'This rank baseline is **not** a reproduction of the full published noise-tolerant interleaver algorithm.','',
      '## 4. Abstention, kept separate from prediction accuracy','',
      '| Stage | Accepted / total | Coverage | Accepted accuracy | Accepted accuracy 95% CI |',
      '| --- | ---: | ---: | ---: | --- |']
    for name,m in [('SPS',p['sps']['learned']),('AMC, including noise',a['learned']),('Joint FEC/interleaver',f['learned'])]:
        lo,hi=m['accepted_wilson95']
        lines.append(f"| {name} | {m['accepted_count']}/{m['n']} | {100*m['coverage']:.2f}% | {100*m['accepted_accuracy']:.2f}% | {100*lo:.2f}–{100*hi:.2f}% |")
    lines+=['','Noise-class predictions are correct in closed-set AMC scoring but are deliberately rejected by the API. '
      'FEC unknown predictions are likewise not accepted. 100% observed selective accuracy is not a guarantee. '
      'Confidence intervals quantify sampling uncertainty within this generator; they do not account for simulator bias. '
      'Paired improvement intervals use 5000 capture bootstrap resamples, not independent intervals subtracted from each other.','',
      '## 5. What the research comparisons do and do not establish','',
      '| Reference / benchmark | Evidence status | Honest conclusion |',
      '| --- | --- | --- |',
      '| [O\'Shea et al., CFO estimator, 2017](https://arxiv.org/abs/1707.06260), [Chen et al., 2023](https://arxiv.org/abs/2311.16155) | Local CNN recreations and archived shared-cohort results in `tables.md`; fresh replay status below. | Not original published training budgets/datasets. |',
      '| [O\'Shea et al., AMC, 2016](https://arxiv.org/abs/1602.04105), Rajendran et al., 2018 | VT-CNN2/LSTM recreations are explicitly undertrained; fresh replay status below. | A local checkpoint win is not a published-benchmark win. |',
      '| [Ma et al., DBFCNN, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12881466/) | Seven-family study; external dataset not acquired; our four-code universe differs. | No numerical SOTA comparison is defensible. |',
      '| [Swaminathan et al., interleavers, 2017](https://doi.org/10.1109/ACCESS.2017.2684189), [Singh et al., subspace codes, 2026](https://arxiv.org/abs/2601.15903) | Full algorithms not reproduced. | Our simple rank test is not their benchmark. |','',
      '**Counterexamples to a universal improvement claim:** the archived broad-SNR CFO report gives v2 better RMSE '
      '(0.0165 vs v1 0.0209) but worse ±0.005 success (83.5% vs 86.5%). '
      'Archived narrow-range Kay RMSE 0.00798 is slightly better than v2 0.00800. '
      'The old no-CFO AMC control also beats v2 in-distribution (77.7% vs 76.2%). '
      'These are historical conditions, not the fresh easier 5–30 dB cohorts above.','',
      'Remaining validation: independent real captures; full published benchmark datasets and matched training budgets; '
      'continuous/noninteger SPS; unseen pulses and protocols; synchronization/bit-mapping errors; arbitrary interleavers; '
      'FEC decoding, BER of recovered payloads and end-to-end CRC checks. No encryption/decryption validation exists.','',
      '## Reproduce and inspect','',
      '```powershell','python research/validate_stages.py','python research/report_stage_validation.py',
      'python -m unittest discover -s tests','```','',
      'NumPy-only execution. `stage_validation.json` retains recipes, seeds, source and model hashes, '
      'confusion matrices, capture-level predictions and paired intervals; no raw synthetic signal files are saved. '
      'Logs show original product training reached 4000 SpecCFO v2 steps and 2500 DemodAMC v2 steps; '
      'this validation does not resume training or tune test-set thresholds.','']
    replay=ROOT/'research/results/stage_paper_recreations.json'
    if replay.exists():
        q=json.loads(replay.read_text(encoding='utf-8'))
        lines+=['## Fresh replay of local paper recreations','',
          'The CPU JAX evaluator regenerated the exact recipes above, verified test labels and froze checkpoint hashes. '
          'CFO networks receive the identical 1024-sample cohorts; window AMC models see all 32 disjoint 128-sample windows of each 4096-sample capture, '
          'aggregated by mean log probability as in the repository evaluator. No model gets oracle test CFO.','',
          '| CFO cohort | Recreation | RMSE | Within ±0.005 | SpecCFO v2 paired gain |',
          '| --- | --- | ---: | --- | --- |']
        for sweep,methods in q['cfo'].items():
            for name,m in methods.items():
                lines.append(f"| {sweep} | {name} | {m['rmse_cycles_per_sample']:.6f} | {score(m['within_005'])} | {delta(m['paired'])} |")
        lines+=['','| AMC recreation | Accuracy | Current DemodAMC paired gain |',
                '| --- | --- | --- |']
        for name,m in q['amc'].items():
            lines.append(f"| {name} | {score(m['metrics'])} | {delta(m['paired'])} |")
        lines+=['','The very large AMC margins are against the saved undertrained checkpoints (2000 steps, loss about 2.00 and 1.85), '
          'not evidence of beating adequately trained VT-CNN2/LSTM or the papers. '
          'Fresh recipe replay does not remedy unequal original training budgets or acquire a published benchmark. '
          'Full predictions, checkpoint hashes and training metadata are in `stage_paper_recreations.json`.','',
          'Optional replay: install `jax[cpu]` and `optax`, then run `python research/validate_paper_recreations.py` and regenerate this report.','']
    else:
        lines+=['Fresh research-network replay has not completed in this report; archived figures are not silently substituted.','']
    (ROOT/'research/results/stage_validation.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
