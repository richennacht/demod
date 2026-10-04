"""Replay stage-validation recipes against frozen local paper recreations.

Never invokes evaluator mains (which can calibrate weights). Optional JAX/optax
are isolated in .venv/stage-eval-deps; bundled NumPy stays first on the import path.
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'.venv/stage-eval-deps'))
sys.path.insert(0,str(ROOT/'src'))
from validate_stages import accuracy, paired, digest, SEED
from signal_sim import simulate, SimConfig, MODULATIONS, LINEAR
from cfo_estimators import wrap
from amc_models import iq_windows, ap_windows
from eval_cfo import load_net
from eval_amc import load, log_softmax


def main():
    import jax
    report=json.loads((ROOT/'research/results/stage_validation.json').read_text(encoding='utf-8'))
    names=('iq_resnet','oshea_cfo','vtcnn2','lstm_ap')
    paths=[ROOT/f'research/checkpoints/{n}.npz' for n in names]
    before={str(p.relative_to(ROOT)):digest(p) for p in paths}
    results=dict(scope='Frozen local paper-architecture recreations, not original paper models/budgets or published benchmark datasets.',
                 jax_version=jax.__version__,numpy_version=np.__version__,checkpoint_hashes=before,
                 source_sha256=digest(__file__),cfo={},amc={})
    nets={n:load_net(n) for n in ('iq_resnet','oshea_cfo')}
    for sweep in ('wide','narrow'):
        seed,bound=(SEED+1,.2) if sweep=='wide' else (SEED+2,.025)
        b=simulate(np.random.default_rng(seed),200,SimConfig(length=1024,modulations=LINEAR,snr_db=(5,30),cfo=(-bound,bound)))
        old=report['parameter_estimation']['cfo'][sweep]['records']
        np.testing.assert_array_equal(b['cfo'],[v['truth'] for v in old])
        proposed=(abs(wrap(np.array([v['estimates']['SpecCFO_v2'] for v in old])-b['cfo']))<=.005).astype(int)
        results['cfo'][sweep]={}
        for name,fn in nets.items():
            pred=fn(b['x']); error=wrap(pred-b['cfo']); good=(abs(error)<=.005).astype(int)
            results['cfo'][sweep][name]=dict(rmse_cycles_per_sample=float(np.sqrt(np.mean(error**2))),
                within_005=accuracy(np.ones(200,dtype=int),good),
                paired=paired(np.ones(200,dtype=int),proposed,good),estimates=pred.tolist())
            print('Evaluated',sweep,name,flush=True)
    rng=np.random.default_rng(SEED+4); captures=[]; truth=[]
    for name in MODULATIONS:
        b=simulate(rng,10,SimConfig(length=4096,snr_db=(5,25),pulses=('rrc','rect'),fixed={'modulation':name}))
        captures.extend(b['x']);truth.extend(b['modulation'])
    captures=np.asarray(captures);truth=np.asarray(truth)
    old=report['modulation']['records']
    np.testing.assert_array_equal(truth,[v['truth'] for v in old])
    for name in ('vtcnn2','lstm_ap'):
        fn,kind,meta=load(name); pred=[]
        for start in range(0,120,10):
            xb=captures[start:start+10]
            w=iq_windows(xb) if kind=='iq' else ap_windows(xb)
            z=np.asarray(fn(w.reshape(-1,*w.shape[2:]))).reshape(len(xb),w.shape[1],-1)
            pred.extend(log_softmax(z).mean(axis=1).argmax(axis=1).tolist())
        results['amc'][name]=dict(metrics=accuracy(truth,pred),
                 paired=paired(truth,[v['learned'] for v in old],pred),predictions=pred,
                 training_metadata=meta,aggregation='mean window log probability across all 32 disjoint 128-sample windows; no true CFO supplied')
        print('Evaluated',name,flush=True)
    assert before=={str(p.relative_to(ROOT)):digest(p) for p in paths}
    (ROOT/'research/results/stage_paper_recreations.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__': main()
