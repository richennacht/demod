"""Reference-bit BER/SER on reproducible ephemeral controlled transmissions."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from library_receiver import receive, constellation, SUPPORTED, libraries
from signal_sim import rrc_taps


def fixture(rng,modulation,n=1024,sps=8,pulse='rect',snr=30,cfo=.003,phase=.17,offset=3):
    """Known symbol labels -> rectangular/RRC IQ. Receiver never receives truth."""
    m={'bpsk':2,'qpsk':4,'8psk':8,'16qam':16,'64qam':64,'2fsk':2,'4fsk':4}[modulation]
    indexes=rng.integers(0,m,n)
    if modulation.endswith('fsk'):
        levels=np.arange(-(m-1),m,2)*.025
        # phase[n+1]-phase[n] uses frequency[n], so within-symbol discriminator matches exactly.
        frequency=np.repeat(levels[indexes],sps)
        x=np.exp(2j*np.pi*np.r_[0.,np.cumsum(frequency[:-1])])
        label_values=np.arange(m)^(np.arange(m)>>1)
        labels=((label_values[:,None]>>np.arange(int(np.log2(m))-1,-1,-1))&1)
    else:
        points,labels=constellation(modulation)
        if pulse=='rect': x=np.repeat(points[indexes],sps)
        else:
            _,signal=libraries(); impulses=np.zeros(n*sps,dtype=complex);impulses[::sps]=points[indexes]
            x=signal.convolve(impulses,rrc_taps(sps,.35),mode='same')*np.sqrt(sps)
    x=np.r_[np.zeros(offset),x]
    x=x*np.exp(1j*(phase+2*np.pi*cfo*np.arange(len(x))))
    noise_power=np.mean(abs(x[offset:])**2)*10**(-snr/10)
    x+=np.sqrt(noise_power/2)*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))
    return x,indexes,labels, (levels*250000 if modulation.endswith('fsk') else None)


def main():
    rng=np.random.default_rng(20261006); rows=[]
    for mod in SUPPORTED:
        for pulse in (('rect',) if mod.endswith('fsk') else ('rect','rrc')):
            for snr in (15,25):
                for acquisition in ('manual','static_auto'):
                    errors=0; total=0; symbol_errors=0; symbols=0; records=[]
                    for k in range(4):
                        x,truth,labels,tones=fixture(rng,mod,pulse=pulse,snr=snr)
                        r=receive(x,250000,mod,8,3 if acquisition=='manual' else None,750.,pulse,.35,
                                  .17 if acquisition=='manual' else None,tones)
                        predicted=np.asarray(r['bits']).reshape(-1,labels.shape[1])
                        # Fixed, declared edge exclusion only; no truth-based realignment/rotation.
                        limit=min(len(predicted),len(truth)); sl=slice(12,limit-12)
                        target=labels[truth[:limit]][sl]; pred=predicted[:limit][sl]
                        e=int((target!=pred).sum()); count=target.size
                        se=int(np.any(target!=pred,axis=1).sum())
                        errors+=e;total+=count;symbol_errors+=se;symbols+=len(target)
                        records.append(dict(input_sha256=hashlib.sha256(x.astype('<c8').tobytes()).hexdigest(),
                            bit_errors=e,bits_compared=count,symbol_errors=se,symbols_compared=len(target),
                            actual_timing=r['configuration']['timing_offset_samples'],
                            actual_phase=r['configuration']['phase_radians']))
                    rows.append(dict(modulation=mod,pulse=pulse,snr_db=snr,acquisition=acquisition,
                                bit_errors=errors,bits_compared=total,ber=errors/total,
                                symbol_errors=symbol_errors,symbols_compared=symbols,ser=symbol_errors/symbols,records=records))
                    print(mod,pulse,snr,acquisition,'BER',errors/total,flush=True)
    out=dict(seed=20261006,scope='Controlled same-generator burst BER/SER, not real intercept or blind phase/mapping recovery.',
        recipe=dict(symbols=1024,sps=8,fs=250000,cfo_hz=750,phase_radians=.17,timing_offset=3,
                    rrc_rolloff=.35,edge_symbols_excluded=12,captures_per_row=4,fsk_tones='known equally spaced 0.025 cycles/sample odd levels'),
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'src/library_receiver.py')},
        komm_version=libraries()[0].__version__,numpy_version=np.__version__,rows=rows)
    (ROOT/'research/results/receiver_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# Controlled receiver BER/SER','',out['scope'],'',
        'Four captures per row, 1024 transmitted symbols, SPS 8, Fs 250 kHz, CFO +750 Hz, phase +0.17 radians, leading offset 3 samples. '
        'CFO and FSK tone levels are supplied. Manual mode supplies phase/timing; auto searches static phase/timing. '
        'The first/last 12 symbols are excluded by a fixed rule; no oracle alignment or rotation is performed. '
        'RRC uses rolloff 0.35. Independent random data per row, so manual/auto rows are not paired comparisons.','',
        '| Mode | Pulse | SNR dB | Acquisition | Bit errors / compared | BER | SER |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['modulation']} | {r['pulse']} | {r['snr_db']} | {r['acquisition']} | {r['bit_errors']}/{r['bits_compared']} | {r['ber']:.6f} | {r['ser']:.6f} |")
    lines+=['','Zero measured errors is not a zero-error guarantee. Static phase search is M-fold ambiguous; '
        'these phases lie inside the selected fundamental sector. Tests do not establish arbitrary-phase recovery, '
        'clock-drift/fractional timing recovery, multipath equalisation, unknown mapping or protocol/FEC decoding. '
        'GNU Radio execution is not validated here. Recipes and per-capture counts/hashes are in receiver_results.json.','']
    (ROOT/'research/results/receiver_results.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
