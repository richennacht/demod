"""Library-backed bounded burst receivers, not blind protocol decoders.

Komm makes constellation decisions; SciPy applies receive filters. Static
timing/phase acquisition is a bounded hypothesis search, not a tracking clock.
"""
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]/'.venv/receiver-deps'))
SUPPORTED=('bpsk','qpsk','8psk','16qam','64qam','2fsk','4fsk')


def libraries():
    try:
        import komm
        from scipy import signal
        if not hasattr(komm,'Constellation'): raise ImportError('Komm constellation API unavailable.')
    except ImportError as error:
        raise ValueError('Library receiver needs requirements-receiver.txt (Komm and SciPy).') from error
    return komm,signal


def _slice(slicer,values):
    # Bound Komm's pairwise-distance working set for long captures / 64-QAM.
    return np.concatenate([np.asarray(slicer.closest_indices(values[j:j+4096])).reshape(-1)
                           for j in range(0,len(values),4096)])


def constellation(modulation):
    """Points and MSB-first bit labels. Preserve legacy BPSK/QPSK mapping."""
    if modulation=='bpsk':
        return np.array([-1.,1.],dtype=complex), np.array([[0],[1]])
    if modulation=='qpsk':
        return np.array([-1-1j,-1+1j,1-1j,1+1j])/np.sqrt(2),np.array([[0,0],[0,1],[1,0],[1,1]])
    if modulation=='8psk':
        points=np.exp(2j*np.pi*np.arange(8)/8); indexes=np.arange(8)^ (np.arange(8)>>1); width=3
    elif modulation in ('16qam','64qam'):
        side=4 if modulation=='16qam' else 8
        levels=np.arange(-(side-1),side,2)
        points=(levels[:,None]+1j*levels[None,:]).ravel()
        points=points/np.sqrt(np.mean(abs(points)**2))
        axis=np.arange(side)^(np.arange(side)>>1)
        indexes=(axis[:,None]*side+axis[None,:]).ravel(); width=2*int(np.log2(side))
    else:
        raise ValueError('No linear constellation for '+modulation)
    return points,((indexes[:,None]>>np.arange(width-1,-1,-1))&1)


def _static_acquire(values, points, slicer, sps, timing, phase, pulse):
    times=[timing] if timing is not None else range(sps)
    symmetry=2 if len(points)==2 else 8 if len(points)==8 else 4
    phases=[phase] if phase is not None else np.unique(np.r_[0.,np.linspace(-np.pi/symmetry,np.pi/symmetry,65,endpoint=False)])
    best=None
    for offset in times:
        symbols=values[offset:len(values)-sps+1:sps] if pulse=='rrc' else values[offset:offset+(len(values)-offset)//sps*sps].reshape(-1,sps).mean(1)
        if len(symbols)<2: continue
        gain=np.sqrt(np.mean(abs(symbols)**2))
        if gain<1e-12: continue
        z=symbols[:1024]/gain
        hypotheses=z[None,:]*np.exp(-1j*np.asarray(phases)[:,None])
        indexes=_slice(slicer,hypotheses.ravel()).reshape(hypotheses.shape)
        scores=np.mean(abs(hypotheses-points[indexes])**2,axis=1)
        j=int(scores.argmin()); candidate=(float(scores[j]),offset,float(phases[j]),gain,symbols)
        if best is None or candidate[0]<best[0]: best=candidate
    if best is None: raise ValueError('No nonzero complete symbol region available.')
    _,offset,angle,gain,symbols=best
    return symbols/gain*np.exp(-1j*angle),offset,angle,gain


def receive(samples, fs, modulation, sps, timing=0, cfo_hz=0., pulse='rect',
            rolloff=.35, phase_radians=0., fsk_tones_hz=None):
    komm,signal=libraries()
    if modulation not in SUPPORTED: raise ValueError('Unsupported modulation.')
    if type(sps) is not int or not 1<=sps<=64: raise ValueError('SPS must be an integer in [1,64].')
    if timing is not None and (type(timing) is not int or not 0<=timing<sps): raise ValueError('Timing offset must be in [0,SPS), or auto.')
    if not np.isfinite([fs,cfo_hz,rolloff]).all() or fs<=0 or abs(cfo_hz)>=fs/2: raise ValueError('Invalid Fs or CFO (must be inside Nyquist).')
    if phase_radians is not None and not np.isfinite(phase_radians): raise ValueError('Phase must be finite.')
    if pulse not in ('rect','rrc') or not 0<rolloff<=1: raise ValueError('Pulse must be rect/rrc; RRC rolloff in (0,1].')
    x=np.asarray(samples,dtype=complex)
    if x.ndim!=1 or len(x)<2*sps or not np.isfinite(x).all(): raise ValueError('Need finite one-dimensional samples for at least two symbols.')
    x=x*np.exp(-2j*np.pi*cfo_hz*np.arange(len(x))/fs)
    x=x-x.mean()
    if np.mean(abs(x)**2)<1e-15: raise ValueError('Capture has no usable AC signal.')
    filter_delay=0
    if modulation.endswith('fsk'):
        if pulse!='rect': raise ValueError('This discriminator receiver supports unshaped CPFSK; GFSK/GMSK require a separate receiver.')
        m=2 if modulation=='2fsk' else 4
        d=np.angle(x[1:]*x[:-1].conj())/(2*np.pi)*fs
        centers=None
        if fsk_tones_hz is not None:
            centers=np.asarray(fsk_tones_hz,dtype=float)
            if centers.shape!=(m,) or not np.isfinite(centers).all() or np.any(np.diff(centers)<=0) or np.any(abs(centers)>=fs/2):
                raise ValueError('FSK tones must be ascending finite Hz values inside Nyquist, one per symbol.')
        choices=[]
        for off in ([timing] if timing is not None else range(sps)):
            # Differences from the first to last sample WITHIN each symbol: exclude transition-crossing pairs.
            starts=np.arange(off,len(x)-sps+1,sps)
            if sps<2: raise ValueError('FSK discriminator needs SPS >= 2.')
            v=np.array([d[start:start+sps-1].mean() for start in starts])
            if len(v)<m: continue
            c=centers.copy() if centers is not None else np.quantile(v,(np.arange(m)+.5)/m)
            if centers is None:
                for _ in range(20):
                    lab=abs(v[:,None]-c).argmin(1)
                    c=np.array([v[lab==j].mean() if np.any(lab==j) else c[j] for j in range(m)])
                c.sort()
            if np.min(np.diff(c))<1e-8: continue
            slicer=komm.Constellation(c[:,None])
            indexes=_slice(slicer,v)
            error=float(np.mean((v-c[indexes])**2)/np.min(np.diff(c))**2)
            choices.append((error,off,v,c,indexes))
        if not choices: raise ValueError('FSK tone levels could not be separated; supply tone frequencies.')
        error,offset,v,centers,indexes=min(choices,key=lambda a:a[0])
        labels=np.arange(m)^(np.arange(m)>>1)
        width=int(np.log2(m)); bits=((labels[indexes,None]>>np.arange(width-1,-1,-1))&1).ravel()
        decisions=[dict(frequency_hz=float(a),symbol=int(b)) for a,b in zip(v[:256],indexes[:256])]
        quality=dict(evm_rms=None,frequency_cluster_residual=error,meaning='FSK within-tone residual; not BER.')
        detail=dict(tone_centres_hz=centers.tolist(),tone_source='analyst' if fsk_tones_hz is not None else 'bounded 1D clustering; all tones must be represented',phase_radians=None)
    else:
        points,labels=constellation(modulation); slicer=komm.Constellation(points[:,None])
        if pulse=='rrc':
            from signal_sim import rrc_taps
            taps=rrc_taps(sps,rolloff); filter_delay=(len(taps)-1)//2
            # Zero-phase alignment of a finite burst; boundary symbols remain transient.
            x=signal.convolve(x,taps,mode='same',method='auto')
        z,offset,angle,gain=_static_acquire(x,points,slicer,sps,timing,phase_radians,pulse)
        indexes=_slice(slicer,z); bits=labels[indexes].ravel()
        evm=float(np.sqrt(np.mean(abs(z-points[indexes])**2)))
        decisions=[dict(i=float(a.real),q=float(a.imag),symbol=int(b)) for a,b in zip(z[:256],indexes[:256])]
        quality=dict(evm_rms=evm,meaning='RMS-normalized decision-directed EVM, not reference EVM or BER.')
        detail=dict(phase_radians=angle,gain_normalization=gain,constellation_points=[dict(i=float(a.real),q=float(a.imag)) for a in points],phase_source='analyst' if phase_radians is not None else 'bounded EVM search; rotational ambiguity unresolved')
    return dict(bits=bits.astype(int).tolist(),symbol_count=len(indexes),decisions_preview=decisions,quality=quality,
                configuration=dict(backend='komm_scipy',komm_version=komm.__version__,samples_per_symbol=sps,
                    timing_offset_samples=int(offset),timing_source='analyst_fixed_offset' if timing is not None else 'static integer-offset search; no clock-drift tracking',
                    carrier_offset_hz=cfo_hz,pulse=pulse,rrc_rolloff=rolloff if pulse=='rrc' else None,
                    filter_group_delay_samples=filter_delay,filter_alignment='same-mode, nominal delay compensated; burst edges transient',
                    bit_mapping='legacy sign/quadrant for BPSK/QPSK; reflected Gray angular/axis/tone order otherwise',**detail))
