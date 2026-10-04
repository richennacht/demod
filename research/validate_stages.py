"""Frozen-model, paired synthetic validation. Never recalibrate or train product weights.

Run from any directory. Only recipes, hashes, predictions and metrics are persisted.
The simple baselines are local implementations, NOT full published benchmark replicas.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from signal_sim import simulate, SimConfig, MODULATIONS, LINEAR
import cfo_estimators as cfo
import amc_models as amc
import rate_estimation as rate
import fec_identification as fec
from learned_analysis import classify, carrier_offset
from train_fec_identifier import example, wilson, parity_gate

SEED = 20261005


def accuracy(truth, prediction, accepted=None):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    correct = truth == prediction
    n = len(truth)
    labels = np.unique(np.concatenate([truth, prediction]))
    confusion = [[int(((truth == a) & (prediction == b)).sum()) for b in labels] for a in labels]
    # Includes predictions outside the truth subset in false-positive/negative counts.
    f1 = [2 * int(((truth == a) & (prediction == a)).sum()) /
          max(1, int((truth == a).sum()) + int((prediction == a).sum())) for a in np.unique(truth)]
    out = dict(n=n, correct=int(correct.sum()), accuracy=float(correct.mean()),
               wilson95=wilson(int(correct.sum()), n), macro_f1=float(np.mean(f1)),
               labels=labels.tolist(), confusion=confusion)
    if accepted is not None:
        accepted = np.asarray(accepted, dtype=bool)
        k = int(accepted.sum()); successes = int((correct & accepted).sum())
        out.update(accepted_count=k, coverage=k/n, accepted_correct=successes,
                   accepted_accuracy=successes/k if k else None,
                   accepted_wilson95=wilson(successes, k))
    return out


def paired(truth, proposed, baseline, seed=SEED):
    """Paired percentile bootstrap CI for accuracy difference; unit is one capture."""
    truth = np.asarray(truth)
    a = np.asarray(proposed) == truth; b = np.asarray(baseline) == truth
    difference = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    samples = difference[rng.integers(0, len(a), (5000, len(a)))].mean(1)
    return dict(delta_percentage_points=float(100*difference.mean()),
                delta_bootstrap95_pp=(100*np.quantile(samples, [.025,.975])).tolist(),
                proposed_only_correct=int((a & ~b).sum()),
                baseline_only_correct=int((b & ~a).sum()),
                scope='paired capture bootstrap, 5000 resamples; conditional on this synthetic mixture')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_suite():
    rng = np.random.default_rng(SEED)
    rows = []
    for sps in rate.CLASSES:
        batch = simulate(rng, 100, SimConfig(length=2048, modulations=LINEAR,
                         snr_db=(5,30), pulses=('rrc','rect'), hardware_probability=.3,
                         fixed={'sps': int(sps)}))
        for x in batch['x']:
            result = rate.estimate(x)
            learned = result['learned']
            pred = max(learned['candidates'], key=lambda v:v['probability'])['samples_per_symbol']
            manual = result['manual']['candidates'][0]['samples_per_symbol']
            rows.append(dict(truth=int(sps), learned=pred,
                        manual_continuous=manual,
                        manual=int(rate.CLASSES[np.argmin(abs(rate.CLASSES-manual))]),
                        accepted=not learned['abstained']))
    y = [r['truth'] for r in rows]; p = [r['learned'] for r in rows]; b = [r['manual'] for r in rows]
    sps = dict(recipe=dict(seed=SEED, n=400, samples=2048, sps=[2,4,8,16],
               snr_db=[5,30], modulations=list(LINEAR), pulses=['rrc','rect'], hardware_probability=.3),
               learned=accuracy(y,p,[r['accepted'] for r in rows]),
               manual_quantized=accuracy(y,b), paired=paired(y,p,b), records=rows,
               manual_definition='Top transition-power spectral peak rounded to nearest declared SPS class; no truth-guided harmonic selection.',
               absolute_Fs='Not measured: unidentifiable without metadata/time reference; SPS accuracy is not absolute Fs accuracy.')
    sweeps = {}
    for key, bound in [('wide',.2), ('narrow',.025)]:
        rng = np.random.default_rng(SEED+1 if key=='wide' else SEED+2)
        batch = simulate(rng, 200, SimConfig(length=1024, modulations=LINEAR,
                         snr_db=(5,30), cfo=(-bound,bound)))
        x = batch['x']; y = batch['cfo']
        estimates = {'SpecCFO_v2': cfo.default_model().estimate(x)['cfo'],
                     'Kay_raw': cfo.kay_wpa(x),
                     'blind_multi_M': cfo.blind_multi_m(x)[0]}
        oracle = np.empty(len(y))
        for index, name in enumerate(MODULATIONS):
            mask = batch['modulation'] == index
            if mask.any():
                oracle[mask] = cfo.power_periodogram(x[mask], {'bpsk':2,'qpsk':4,'8psk':8,'16qam':4,'64qam':4}[name])[0]
        estimates['known_M_periodogram_oracle'] = oracle
        metrics = {}
        predictions = {}
        for name, pred in estimates.items():
            error = cfo.wrap(pred-y); good = abs(error) <= .005
            predictions[name] = good.astype(int)
            metrics[name] = dict(rmse_cycles_per_sample=float(np.sqrt(np.mean(error**2))),
                                 within_005=accuracy(np.ones(len(y),dtype=int),good.astype(int)))
        sweeps[key] = dict(recipe=dict(seed=SEED+(1 if key=='wide' else 2), n=200, samples=1024,
                           cfo_range=[-bound,bound], modulations=list(LINEAR), snr_db=[5,30], pulses=['rrc']),
                           metrics=metrics,
                           paired={k:paired(np.ones(len(y),dtype=int),predictions['SpecCFO_v2'],v)
                                   for k,v in predictions.items() if k!='SpecCFO_v2'},
                           records=[dict(truth=float(y[i]), modulation=MODULATIONS[int(batch['modulation'][i])],
                                         estimates={k:float(v[i]) for k,v in estimates.items()}) for i in range(len(y))],
                           caveat='Known-M baseline has oracle modulation information; wide sweep exceeds its unambiguous range. Normalized CFO, not RF centre frequency.')
    return dict(sps=sps,cfo=sweeps)


def amc_suite():
    # Fit only a new simple baseline, using a separate training seed. Product weights remain frozen.
    rng = np.random.default_rng(SEED+3); feats=[]; labels=[]
    for name in MODULATIONS:
        batch = simulate(rng, 50, SimConfig(length=4096,snr_db=(5,25),pulses=('rrc','rect'),fixed={'modulation':name}))
        feats.extend(amc.centroid_features(batch['x'])); labels.extend([MODULATIONS.index(name)]*50)
    centroid = amc.NearestCentroid().fit(np.asarray(feats),np.asarray(labels))
    rng = np.random.default_rng(SEED+4); records=[]
    for name in MODULATIONS:
        batch = simulate(rng, 10, SimConfig(length=4096,snr_db=(5,25),pulses=('rrc','rect'),fixed={'modulation':name}))
        for x in batch['x']:
            offset = carrier_offset(x,250000)['normalised_cycles_per_sample']
            result = classify(x,offset)
            adjusted = amc.derotate(x[None,:],np.asarray([offset]))
            records.append(dict(truth=MODULATIONS.index(name),
                           learned=MODULATIONS.index(result['ranked_candidates'][0]['modulation']),
                           accepted=not result['abstained'],
                           centroid=int(centroid.predict(amc.centroid_features(x))[0]),
                           cumulant_raw=int(amc.cumulant_classify(x)[0]),
                           cumulant_derotated=int(amc.cumulant_classify(adjusted)[0])))
        print('AMC evaluated',name,flush=True)
    y=[r['truth'] for r in records]; p=[r['learned'] for r in records]; b=[r['centroid'] for r in records]
    linear=[r for r in records if MODULATIONS[r['truth']] in LINEAR]
    ly=[r['truth'] for r in linear]; lp=[r['learned'] for r in linear]
    return dict(recipe=dict(test_seed=SEED+4, baseline_training_seed=SEED+3, baseline_training_n=600,
                  n=120,samples=4096,snr_db=[5,25],pulses=['rrc','rect'],cfo_range=[-.2,.2],sps_range=[2,16],classes=list(MODULATIONS)),
                learned=accuracy(y,p,[r['accepted'] for r in records]), centroid=accuracy(y,b),paired_centroid=paired(y,p,b),
                linear_subset=dict(n=len(linear),learned=accuracy(ly,lp),
                  cumulant_raw=accuracy(ly,[r['cumulant_raw'] for r in linear]),
                  cumulant_derotated=accuracy(ly,[r['cumulant_derotated'] for r in linear]),
                  paired_derotated=paired(ly,lp,[r['cumulant_derotated'] for r in linear])),records=records,
                caveat='Five-class cumulant evaluated only on linear subset; DemodAMC still chooses among all 12 classes. Noise is intentionally abstained by API.')


def fec_suite():
    with np.load(fec.MODEL_PATH,allow_pickle=False) as z:
        model={k:z[k] for k in z.files}
    rng=np.random.default_rng(SEED+5); rows=[]; subset=[]
    for ber in (0,.01,.03,.05,.08):
        for label in range(len(fec.LABELS)):
            for k in range(4):
                bits=example(rng,label,ber=ber); evidence=fec.candidate_evidence(bits); f=fec.features(bits,evidence)
                prob=fec.ml_probabilities(f,model); pred=int(prob.argmax())
                rates=np.asarray([c['syndrome_violation_rate'] for c in evidence])
                manual=int(rates.argmin()) if rates.min()<=.32 else len(fec.CANDIDATES)
                accepted=bool(parity_gate(f[None,:],np.asarray([pred]))[0] and prob.max()>=float(model['threshold']))
                r=dict(truth=label,learned=pred,manual=manual,accepted=accepted,ber=ber,input_sha256=hashlib.sha256(bits.tobytes()).hexdigest())
                rows.append(r)
                if k==0:
                    subset.append(dict(**r,rank=fec.rank_baseline(bits)))
        print('FEC evaluated BER',ber,flush=True)
    y=[r['truth'] for r in rows]; p=[r['learned'] for r in rows]; b=[r['manual'] for r in rows]
    supported=[r for r in rows if r['truth']<len(fec.CANDIDATES)]
    def component(label,index):
        return fec.CANDIDATES[label][index] if label<len(fec.CANDIDATES) else 'unknown'
    return dict(recipe=dict(seed=SEED+5,n=len(rows),length_bits=3360,ber=[0,.01,.03,.05,.08],
                  examples_per_class_per_ber=4,labels=list(fec.LABELS),frame_alignment='known 420-bit boundary',channel='iid binary symmetric errors'),
                learned=accuracy(y,p,[r['accepted'] for r in rows]), manual=accuracy(y,b),paired_manual=paired(y,p,b),
                supported_components={key:{method:accuracy([component(r['truth'],i) for r in supported],
                                                  [component(r[method],i) for r in supported])
                                                  for method in ('learned','manual')}
                                      for i,key in enumerate(('code','interleaver'))},
                strata={str(ber):accuracy([r['truth'] for r in rows if r['ber']==ber],
                                        [r['learned'] for r in rows if r['ber']==ber],
                                        [r['accepted'] for r in rows if r['ber']==ber]) for ber in (0,.01,.03,.05,.08)},
                paired_rank_subset=dict(rule='first capture per class/BER chosen before scoring',n=len(subset),
                    learned=accuracy([r['truth'] for r in subset],[r['learned'] for r in subset]),
                    rank=accuracy([r['truth'] for r in subset],[r['rank'] for r in subset]),
                    paired=paired([r['truth'] for r in subset],[r['learned'] for r in subset],[r['rank'] for r in subset])),
                records=rows,rank_records=subset,
                caveat='Known-candidate identification, not FEC decoding or arbitrary interleaver recovery. Unknown means iid uncoded test bits, not all unseen protocols.')


def main():
    started=time.time()
    paths=[cfo.MODEL_PATH,amc.MODEL_PATH,rate.MODEL_PATH,fec.MODEL_PATH]
    before={str(p.relative_to(ROOT)):digest(p) for p in paths}
    code_paths=[Path(__file__),ROOT/'research/train_fec_identifier.py']+list((ROOT/'src').glob('*.py'))
    report=dict(protocol_version=1,seed=SEED,model_hashes=before,numpy_version=np.__version__,
                source_hashes={str(p.relative_to(ROOT)):digest(p) for p in code_paths},
                scope='New independent seeds; same synthetic generators as training. No real-data, published-dataset or SOTA claim.',
                parameter_estimation=parameter_suite())
    print('Parameter evaluation complete',flush=True)
    report['modulation']=amc_suite(); report['fec_interleaver']=fec_suite()
    assert before=={str(p.relative_to(ROOT)):digest(p) for p in paths}, 'Product model changed during validation'
    report['elapsed_seconds']=time.time()-started
    output=ROOT/'research/results/stage_validation.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved',output,'seconds',report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    main()
