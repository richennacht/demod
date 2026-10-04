"""Independent-seed simulation smoke evaluation of the API's capture aggregation."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from signal_sim import simulate, SimConfig, MODULATIONS
from learned_analysis import classify, carrier_offset

def main():
    rng = np.random.default_rng(526147)
    records = []
    for name in MODULATIONS:
        batch = simulate(rng, 10, SimConfig(length=4096, snr_db=(5,25), pulses=('rrc','rect'), fixed={'modulation': name}))
        for x in batch['x']:
            c = carrier_offset(x, 250000)
            r = classify(x, c['normalised_cycles_per_sample'] if c else None)
            records.append({'truth': name, 'prediction': r['predicted_modulation'], 'abstained': r['abstained'], 'probability': r['confidence'], 'chunk_agreement': r['chunk_agreement']})
    signals = [r for r in records if r['truth'] != 'noise']
    accepted = [r for r in signals if not r['abstained']]
    report = {'seed': 526147, 'examples': len(records), 'signal_examples': len(signals),
              'coverage': len(accepted)/len(signals),
              'accepted_accuracy': sum(r['prediction']==r['truth'] for r in accepted)/len(accepted) if accepted else None,
              'noise_abstention_rate': sum(r['abstained'] for r in records if r['truth']=='noise')/10,
              'scope': 'Small same-generator synthetic smoke evaluation; 4096 samples, 5–25 dB, RRC/rect, SPS 2–16; no real-data or calibration claim.',
              'records': records}
    (ROOT/'research/results/capture_amc_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__ == '__main__':
    main()
