"""Optional executable GNU Radio 3.10 linear-modulation receiver.

Requires complex64 little-endian input and GNU Radio installed separately.
Not executed/validated in the Windows development environment.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from library_receiver import constellation


def main():
    p=argparse.ArgumentParser(description='GNU Radio RRC PSK/QAM receiver, outputs candidate bits one byte per bit.')
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--modulation',choices=('bpsk','qpsk','8psk','16qam','64qam'),required=True)
    p.add_argument('--sps',type=float,required=True);p.add_argument('--rolloff',type=float,default=.35)
    a=p.parse_args()
    if not a.input.is_file() or a.output.exists(): p.error('Input must exist and output must not already exist.')
    if not np.isfinite(a.sps) or a.sps<2 or not 0<a.rolloff<=1: p.error('SPS >= 2 and rolloff in (0,1] required.')
    try: from gnuradio import gr,blocks,digital
    except ImportError: p.error('A real GNU Radio 3.10 installation is required; the repository folder is not the runtime.')
    points,labels=constellation(a.modulation)
    numbers=labels @ (2**np.arange(labels.shape[1]-1,-1,-1))
    points=points[np.argsort(numbers)]
    symmetry=2 if a.modulation=='bpsk' else 8 if a.modulation=='8psk' else 4
    const=digital.constellation_calcdist(points.tolist(),[],symmetry,1)
    rx=digital.generic_demod(const,differential=False,pre_diff_code=False,samples_per_symbol=a.sps,excess_bw=a.rolloff)
    tb=gr.top_block('DEmod library receiver')
    tb.connect(blocks.file_source(gr.sizeof_gr_complex,str(a.input),False),rx,blocks.file_sink(gr.sizeof_char,str(a.output)))
    tb.run()
    print('Candidate bits saved. Carrier phase/mapping/frame ambiguity remains; no CRC/FEC/decryption.')


if __name__=='__main__': main()
