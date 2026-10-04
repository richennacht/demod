# GNU Radio interactive receiver graphs

DEmod’s local API can emit an inspectable graph descriptor through `src/gnu_radio_adapter.py`. The intended GNU Radio Companion paths are:

```text
BPSK/QPSK: File Source -> DC Blocker -> Frequency Xlating FIR -> FLL Band-Edge
          -> PFB Clock Sync -> Constellation Receiver -> Unpack K Bits

2-FSK: File Source -> DC Blocker -> Frequency Xlating FIR -> Quadrature Demod
       -> Symbol Sync -> Binary Slicer
```

The mapping follows GNU Radio’s documented generic demodulator structure and its frequency-discriminator / Constellation Receiver blocks. The current developer runtime does not contain GNU Radio, so DEmod returns this graph for analyst review rather than claiming it was executed. Use the Python fallback only for controlled BPSK, QPSK and 2-FSK fixtures with analyst-supplied samples-per-symbol.
