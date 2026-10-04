"""Render aggregate measurements without manually transcribing benchmark numbers."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    r=json.loads((ROOT/'research/results/fec_results.json').read_text())
    lines=['# FEC/interleaver candidate-set evaluation','',
           'Simulation only; independent seeds, same encoder generator. Not published-paper benchmark reproduction or real-intercept validation.',
           '', '| BSC BER | Hybrid joint accuracy | Hard syndrome | Statistics-only MLP | Hybrid acceptance coverage |',
           '| --- | --- | --- | --- | --- |']
    for ber,v in r['ber_strata'].items():
        lines.append(f"| {float(ber):.0%} | {v['hybrid']['joint_accuracy']:.2%} | {v['hard_syndrome']['joint_accuracy']:.2%} | {v['statistics_mlp']['joint_accuracy']:.2%} | {v['hybrid']['coverage']:.2%} |")
    h=r['overall_hybrid']; ci=h['accuracy_wilson95']; aci=h['accepted_accuracy_wilson95']
    lines.extend(['',f"Overall hybrid: {h['n']} cases, {h['joint_accuracy']:.2%} joint accuracy (Wilson 95% interval {ci[0]:.2%}–{ci[1]:.2%}), macro F1 {h['macro_f1']:.4f}.",
        f"Accepted {h['accepted_count']} ({h['coverage']:.2%}); observed accepted-class accuracy {h['accepted_joint_accuracy']:.2%} (Wilson interval {aci[0]:.2%}–{aci[1]:.2%}). Not a guarantee for unseen distributions.",
        '', '## Identical preselected 85-case subset', '', '| Method | Joint accuracy |', '| --- | --- |'])
    for name,m in r['paired_baselines'].items(): lines.append(f"| {name} | {m['joint_accuracy']:.2%} |")
    lines.extend(['', 'The rank baseline is a small GF(2) deficiency method; it is not the full noise-tolerant 2017 interleaver algorithm or the 2026 minimum-denoised-subspace method.',
                  '', '## Stress tests', '', '| Stress condition | Accepted / tested | Correct accepted class, if known |', '| --- | --- | --- |'])
    for kind,v in r['ood'].items():
        correct=v['accepted_correct_class']; lines.append(f"| {kind} | {v['accepted_count']} / {v['n']} | {correct if correct is not None else 'not applicable / unknown'} |")
    lines.extend(['', '**Shifted boundaries and bit slips can retain a correct class hypothesis while invalidating exact deinterleaving/frame recovery.** No frame synchronization was validated, including the accepted stress cases. No decoder or CRC ran.',
                  '', 'Source: `fec_results.json`. Reproduce with `python research/train_fec_identifier.py` then `python research/report_fec_results.py`.'])
    path=ROOT/'research/results/fec_tables.md'; path.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines[:16]))

if __name__=='__main__':main()
