"""Wrap the checked-in IQ demo in a PCM stereo WAV without modifying its samples."""
from pathlib import Path
import wave
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'web/examples/synthetic-qpsk-burst.s16le.iq'
target=ROOT/'web/examples/synthetic-qpsk-burst.wav'
with wave.open(str(target),'wb') as w:
    w.setnchannels(2);w.setsampwidth(2);w.setframerate(250000)
    w.writeframes(source.read_bytes())
print(target)
