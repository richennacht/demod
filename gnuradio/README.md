# GNU Radio interactive receiver graphs

`run_receiver.py` now supplies an actual optional GNU Radio 3.10 RRC PSK/QAM execution path, using `digital.generic_demod` rather than reimplementing its AGC/FLL/PFB/constellation loops. Input is little-endian complex64, output is one byte per candidate bit. This runner is **not executed or validated** in this environment; the repository's `gnuradio/` folder must not be mistaken for an installed runtime. Runtime availability now requires successful imports of `gr` and `digital`, not just discovery of that folder.

```powershell
python gnuradio/run_receiver.py capture.cf32 candidate-bits.u8 --modulation 16qam --sps 8 --rolloff 0.35
```

FSK graph descriptions remain review-only; the executable local FSK path is the tested Komm/SciPy discriminator receiver. See [tooling, literature and limits](../docs/DEMODULATION_TOOLING.md) and [controlled BER/SER](../research/results/receiver_results.md). No GNU Radio execution or FEC/payload decoding is implied by the graph below.

DEmod’s local API can emit an inspectable graph descriptor through `src/gnu_radio_adapter.py`. The intended GNU Radio Companion paths are:

```text
BPSK/QPSK: File Source -> DC Blocker -> Frequency Xlating FIR -> FLL Band-Edge
          -> PFB Clock Sync -> Constellation Receiver -> Unpack K Bits

2-FSK: File Source -> DC Blocker -> Frequency Xlating FIR -> Quadrature Demod
       -> Symbol Sync -> Binary Slicer
```

The mapping follows GNU Radio’s documented generic demodulator structure and its frequency-discriminator / Constellation Receiver blocks. The current developer runtime does not contain GNU Radio, so the API returns graphs for analyst review rather than claiming execution. The library-backed Python receiver supports the seven declared modes under its documented pulse/timing contract.
