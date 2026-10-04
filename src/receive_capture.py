"""Simple file -> candidate bits tool. Output files are created exclusively."""
import argparse
import hashlib
import json
from pathlib import Path
from capture_input import decode_capture
from library_receiver import receive, SUPPORTED


def main():
    p=argparse.ArgumentParser(description='Library-backed IQ/WAV receiver; no protocol decoding/decryption.')
    p.add_argument('input',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modulation',choices=SUPPORTED,required=True); p.add_argument('--sps',type=int,required=True)
    p.add_argument('--sample-rate',type=float,default=0.); p.add_argument('--iq-format',default='s16le')
    p.add_argument('--wav-role',default='unspecified');p.add_argument('--if-centre',type=float,default=0.)
    p.add_argument('--timing',default='0');p.add_argument('--phase',default='0');p.add_argument('--cfo',type=float,default=0.)
    p.add_argument('--pulse',choices=('rect','rrc'),default='rect');p.add_argument('--rolloff',type=float,default=.35)
    p.add_argument('--fsk-tones',help='Ascending tone frequencies in Hz, comma separated after CFO correction.')
    a=p.parse_args()
    if a.input.stat().st_size>16*1024*1024: p.error('Bounded MVP accepts files up to 16 MiB; split larger captures explicitly.')
    raw=a.input.read_bytes()
    samples,fs,meta=decode_capture(raw,a.iq_format,a.sample_rate,a.wav_role,a.if_centre)
    if meta and a.wav_role=='audio': p.error('Audio is already demodulated; cannot recover original RF bits.')
    try:
        r=receive(samples,fs,a.modulation,a.sps,None if a.timing=='auto' else int(a.timing),a.cfo,a.pulse,a.rolloff,
                  None if a.phase=='auto' else float(a.phase),[float(v) for v in a.fsk_tones.split(',')] if a.fsk_tones else None)
        r.update(input_sha256=hashlib.sha256(raw).hexdigest(),wav_metadata=meta,
                 scope='Candidate MSB-first hard bits; no validated framing, FEC decoding or payload.',modulation=a.modulation)
        with a.output.open('x',encoding='utf-8') as f: json.dump(r,f,indent=2)
    except (ValueError,FileExistsError) as error: p.error(str(error))
    print(f"Saved {len(r['bits'])} candidate bits to {a.output}")


if __name__=='__main__': main()
