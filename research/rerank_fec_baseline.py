"""Replay the preselected test subset to refresh only the normalized rank baseline."""
import json
import numpy as np
from train_fec_identifier import ROOT, example, metrics, rank_baseline, LABELS

def main():
    path=ROOT/'research/results/fec_results.json'; report=json.loads(path.read_text())
    rng=np.random.default_rng(826147); truth=[]; predictions=[]
    for ber in (0,.01,.03,.05,.08):
        for label in range(len(LABELS)):
            for k in range(20):
                bits=example(rng,label,ber=ber)
                if k==0:
                    truth.append(label); predictions.append(rank_baseline(bits))
    report['paired_baselines']['rank']=metrics(truth,predictions)
    report['rank_baseline_definition']='GF(2) deficiency divided by candidate n-k (conv window dimension 10 for 16 coded bits); not a full published algorithm.'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report['paired_baselines']['rank']['joint_accuracy'])

if __name__=='__main__': main()
