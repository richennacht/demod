"""Small, clearly synthetic BPSK + Hamming + matrix-interleaver demo fixture."""
import sys
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fec_identification import encode,permute

def main():
    rng=np.random.default_rng(726147)
    bits=permute(encode(rng.integers(0,2,2200),'hamming7')[:3360],'matrix420_r4')
    samples=np.repeat(2*bits.astype(float)-1,8)
    # No silence, CFO or phase ambiguity: controlled receiver and coding integration test.
    iq=np.zeros((len(samples),2),dtype='<i2'); iq[:,0]=np.round(samples*12000)
    path=ROOT/'web/examples/synthetic-fec-bpsk.s16le.iq'; iq.tofile(path)
    recipe={'seed':726147,'kind':'synthetic known-parameter integration fixture','iq_format':'s16le',
            'sample_rate_hz':250000,'modulation':'bpsk','samples_per_symbol':8,'carrier_offset_hz':0,
            'fec':'hamming7','interleaver':'matrix420_r4','frame_offset_bits':0,
            'purpose':'Receiver -> full bits -> FEC/interleaver candidate identification; not an over-the-air validation.'}
    (path.with_suffix('.recipe.json')).write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    print(path)

if __name__=='__main__':main()
