"""Generate ephemeral coded-bit captures; train/calibrate/evaluate a scoped identifier.

Persist recipes, weights and aggregate metrics only. Validation selects temperature
and threshold. Test seeds/labels never affect either. This is not a paper replication.
"""
import sys
import json
import time
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fec_identification import (encode, permute, candidate_evidence, features, LABELS,
    CANDIDATES, PERIOD, MODEL_PATH, rank_baseline, analyse_bits, ml_probabilities)


def example(rng, label, ber=None, length=3360, burst=False):
    if label == len(CANDIDATES):
        bits = rng.integers(0, 2, length, dtype=np.uint8)
    else:
        code, inter = CANDIDATES[label]
        payload = rng.integers(0, 2, length, dtype=np.uint8)
        bits = encode(payload, code)[:length]
        bits = permute(bits, inter)
    p = float(ber) if ber is not None else float(rng.choice((0, .005, .01, .03, .05, .08)))
    errors = (rng.random(len(bits)) < p).astype(np.uint8)
    if burst:
        for _ in range(3):
            start = int(rng.integers(0, len(bits)-32))
            errors[start:start+32] = rng.integers(0, 2, 32, dtype=np.uint8)
    return bits ^ errors


def corpus(seed, per_class):
    rng = np.random.default_rng(seed); x, y = [], []
    for label in range(len(LABELS)):
        for _ in range(per_class):
            b = example(rng, label, length=int(rng.choice((1680, 3360, 5040))))
            x.append(features(b)); y.append(label)
        print(f'features seed={seed} class={LABELS[label]}', flush=True)
    return np.asarray(x), np.asarray(y)


def train(x, y, seed=26147):
    rng = np.random.default_rng(seed)
    mean, scale = x.mean(0), x.std(0)+1e-3; x = (x-mean)/scale
    m = {'mean': mean, 'scale': scale, 'w1': rng.normal(0, np.sqrt(2/x.shape[1]), (x.shape[1], 64)),
         'b1': np.zeros(64), 'w2': rng.normal(0, .1, (64, len(LABELS))), 'b2': np.zeros(len(LABELS)),
         'temperature': np.asarray(1.), 'threshold': np.asarray(.9), 'labels': np.asarray(LABELS)}
    keys = ('w1','b1','w2','b2'); v = {k: np.zeros_like(m[k]) for k in keys}; s = {k: np.zeros_like(m[k]) for k in keys}; step = 0
    target = np.eye(len(LABELS))[y]
    for epoch in range(120):
        for batch in np.array_split(rng.permutation(len(y)), max(1, len(y)//128)):
            a = x[batch]; h = np.maximum(a@m['w1']+m['b1'], 0)
            z = h@m['w2']+m['b2']; p = np.exp(z-z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            dz = (p-target[batch])/len(batch); dh = (dz@m['w2'].T)*(h>0)
            g = {'w2': h.T@dz+.0005*m['w2'], 'b2': dz.sum(0),
                 'w1': a.T@dh+.0005*m['w1'], 'b1': dh.sum(0)}
            step += 1
            for k in keys:
                v[k] = .9*v[k]+.1*g[k]; s[k] = .999*s[k]+.001*g[k]**2
                m[k] -= .003*(v[k]/(1-.9**step))/(np.sqrt(s[k]/(1-.999**step))+1e-8)
    return m


def wilson(successes, count):
    if not count:
        return None
    p = successes/count; z = 1.96; d = 1+z*z/count
    mid = (p+z*z/(2*count))/d
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count**2))/d
    return [float(max(0.,mid-half)), float(min(1.,mid+half))]


def metrics(y, pred, accepted=None):
    y, pred = np.asarray(y), np.asarray(pred)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int); np.add.at(confusion, (y,pred), 1)
    f1 = []
    for j in range(len(LABELS)):
        tp = confusion[j,j]; den = confusion[j,:].sum()+confusion[:,j].sum()
        f1.append(float(2*tp/den) if den else 0.)
    r = {'n': len(y), 'joint_accuracy': float((y==pred).mean()), 'accuracy_wilson95': wilson(int((y==pred).sum()),len(y)),
         'macro_f1': float(np.mean(f1)), 'confusion': confusion.tolist()}
    if accepted is not None:
        accepted = np.asarray(accepted, dtype=bool); count = int(accepted.sum()); correct = int(((y==pred)&accepted).sum())
        r.update(coverage=float(accepted.mean()), accepted_count=count,
                 accepted_joint_accuracy=correct/count if count else None, accepted_accuracy_wilson95=wilson(correct,count))
        unknown = y == len(CANDIDATES)
        r['unknown_false_acceptance_rate'] = float(accepted[unknown].mean()) if unknown.any() else None
    return r


def parity_gate(x, pred):
    # x comprises 16 candidate summaries followed by eight sequence statistics.
    rates = x[:, :64:4]; order = np.argsort(rates, axis=1)
    return (pred == order[:,0]) & (rates.min(1) <= .32) & ((np.sort(rates,axis=1)[:,1]-rates.min(1)) >= .015) & (pred != len(CANDIDATES)) & (
        np.maximum(x[np.arange(len(x)), np.minimum(pred,15)*4+1], x[np.arange(len(x)), np.minimum(pred,15)*4+2]) <= .36)


def main():
    started = time.time(); x,y = corpus(26147,100); vx,vy = corpus(526147,30)
    model = train(x,y)
    # Validation-only temperature selection using cross-entropy, then selective threshold.
    options = []
    for t in np.linspace(.5,3.,26):
        model['temperature'] = np.asarray(t); p = ml_probabilities(vx,model)
        options.append((float(-np.log(p[np.arange(len(vy)),vy]+1e-12).mean()),t))
    model['temperature'] = np.asarray(min(options)[1]); p = ml_probabilities(vx,model); pred = p.argmax(1)
    threshold = 1.01
    for t in np.linspace(.8,.99,20):
        accept = parity_gate(vx,pred) & (p.max(1)>=t)
        if accept.sum() >= 30 and float((pred[accept]==vy[accept]).mean()) >= .99:
            threshold = float(t); break
    model['threshold'] = np.asarray(threshold)
    statistical = train(x[:,64:],y,seed=326147)
    recipe = {'version':1, 'train_seed':26147, 'validation_seed':526147, 'test_seed':826147, 'ood_seed':926147,
              'train_per_joint_class':100, 'validation_per_joint_class':30, 'test_per_joint_class_per_ber':20,
              'labels':list(LABELS), 'payload':'iid Bernoulli(0.5)', 'train_lengths_bits':[1680,3360,5040],
              'train_bsc_ber':[0,.005,.01,.03,.05,.08], 'interleaver_period_bits':PERIOD,
              'frame_alignment':'known interleaver block boundary; codeword offset scanned',
              'model':'standardized parity/statistics features -> ReLU64 -> 17-class softmax; Adam120epochs',
              'raw_capture_storage':'none; regenerate per instance',
              'selection':'validation NLL temperature; >=99% observed selective validation precision and >=30 accepted examples'}
    recipe_raw = json.dumps(recipe,indent=2)
    model['recipe_sha256'] = np.asarray(hashlib.sha256(recipe_raw.encode()).hexdigest())
    MODEL_PATH.parent.mkdir(exist_ok=True); np.savez(MODEL_PATH,**model)
    np.savez(ROOT/'research/checkpoints/fec_statistical_baseline.npz',**statistical)
    (ROOT/'data/recipes/fec-recipes.json').write_text(recipe_raw,encoding='utf-8')
    rng = np.random.default_rng(826147); strata = {}; all_y=[]; all_pred=[]; all_accept=[]; comparison=[]
    for ber in (0,.01,.03,.05,.08):
        truth=[]; learned=[]; accepts=[]; manual=[]; stats=[]
        for label in range(len(LABELS)):
            for k in range(20):
                bits = example(rng,label,ber=ber); e=candidate_evidence(bits); f=features(bits,e)
                p=ml_probabilities(f,model); pred=int(p.argmax()); accept=bool(parity_gate(f[None,:],np.asarray([pred]))[0] and p.max()>=threshold)
                rates=np.asarray([c['syndrome_violation_rate'] for c in e]); man=int(rates.argmin()) if rates.min()<=.32 else len(CANDIDATES)
                stat=int(ml_probabilities(f[64:],statistical).argmax())
                truth.append(label); learned.append(pred); accepts.append(accept); manual.append(man); stats.append(stat)
                if k==0: # identical preselected one-per-class/BER subset for expensive GF(2) comparison
                    comparison.append({'truth':label,'hybrid':pred,'manual':man,'statistics':stat,'rank':rank_baseline(bits)})
        strata[str(ber)]={'hybrid':metrics(truth,learned,accepts),'hard_syndrome':metrics(truth,manual),'statistics_mlp':metrics(truth,stats)}
        all_y.extend(truth); all_pred.extend(learned); all_accept.extend(accepts)
        print(f'evaluation BER={ber} accuracy={strata[str(ber)]["hybrid"]["joint_accuracy"]:.3f}',flush=True)
    # Distribution stress, NOT test-tuned. Unknown tests are unsupported distributions.
    oodrng=np.random.default_rng(926147); ood={}
    for kind in ('burst_errors','misaligned_frame','random_permutation','biased_uncoded','hamming31','whitened','bit_slips'):
        accepted=0; correct=0
        for k in range(80):
            label=int(oodrng.integers(0,len(CANDIDATES))); bits=example(oodrng,label,ber=.01,burst=kind=='burst_errors')
            if kind=='misaligned_frame': bits=np.roll(bits,int(oodrng.integers(1,PERIOD)))
            if kind=='bit_slips': bits=np.r_[bits[:1000],1,bits[1000:2000],bits[2001:]]
            if kind=='random_permutation': bits=bits[oodrng.permutation(len(bits))]
            if kind=='biased_uncoded': bits=(oodrng.random(len(bits))<.2).astype(np.uint8)
            if kind=='hamming31':
                # Independently construct systematic Hamming(31,26), not a trained candidate.
                w=np.zeros((len(bits)//31,31),dtype=np.uint8); data=[j for j in range(31) if (j+1)&j]
                w[:,data]=oodrng.integers(0,2,(len(w),26),dtype=np.uint8)
                for j in range(5): w[:,(1<<j)-1]=np.bitwise_xor.reduce(w[:,((np.arange(1,32)>>j)&1).astype(bool)],axis=1)
                bits=w.ravel()
            if kind=='whitened': bits ^= oodrng.integers(0,2,len(bits),dtype=np.uint8)
            r=analyse_bits(bits); accept=r['status']=='candidate_identified'; accepted+=accept
            if accept and kind in ('burst_errors','misaligned_frame','bit_slips'):
                correct += (r['selected_candidate']['code'],r['selected_candidate']['interleaver'])==CANDIDATES[label]
        ood[kind]={'n':80,'accepted_count':accepted,'acceptance_rate':accepted/80,
                   'accepted_correct_class':correct if kind in ('burst_errors','misaligned_frame','bit_slips') else None,
                   'frame_alignment_recovery_verified':False,
                   'interpretation':'unsupported stress set, acceptance not proof of correct recovery' if kind!='burst_errors' else 'supported family with unseen burst errors'}
    paired={name:metrics([r['truth'] for r in comparison],[r[name] for r in comparison]) for name in ('hybrid','manual','statistics','rank')}
    report={'recipe':recipe,'elapsed_seconds':time.time()-started,'validation_temperature':float(model['temperature']),
            'validation_threshold':threshold,'overall_hybrid':metrics(all_y,all_pred,all_accept),
            'ber_strata':strata,'paired_baselines':paired,'paired_subset_rule':'k=0 per class and BER, fixed before evaluation; 85 examples',
            'rank_baseline_definition':'GF(2) deficiency divided by candidate n-k (conv window dimension 10 for 16 coded bits); not a full published algorithm.',
            'ood':ood,'model_sha256':hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            'scope':'Synthetic independent-seed candidate-set benchmark, same encoder generator; not real intercept validation or published benchmark reproduction.',
            'external_benchmark_status':'DBFCNN2026 dataset available on author request, not acquired; no claimed SOTA comparison.'}
    (ROOT/'research/results/fec_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('elapsed_seconds','validation_temperature','validation_threshold','overall_hybrid','ood')},indent=2),flush=True)


if __name__=='__main__': main()
