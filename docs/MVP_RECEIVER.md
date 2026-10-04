# DEmod receiver MVP

## Current library-backed receiver (2026-10-05)

The current receiver contract is [DEMODULATION_TOOLING.md](DEMODULATION_TOOLING.md): Komm/SciPy paths now support BPSK/QPSK/8-PSK, 16/64-QAM and unshaped 2/4-FSK. Pulse filtering, fixed or static-search timing/phase, tone hypotheses and mapping provenance are exposed in the API and UI. Reference-bit BER/SER results are in [receiver_results.md](../research/results/receiver_results.md). `src/receive_capture.py` exports the full candidate bitstream. The optional GNU Radio PSK/QAM runner exists but is not runtime-validated here. There is still no continuous/fractional timing tracking in Python or verified framing/FEC/payload decoding.

The description below documents the retained **legacy_fixed** three-mode backend, not the new library receiver's complete capabilities.

The local API now offers a bounded receiver endpoint:

```text
POST /demodulate
```

Required headers are `X-DEmod-IQ-Format`, `X-DEmod-Sample-Rate`, and `X-DEmod-Samples-Per-Symbol`. Use `X-DEmod-Modulation` as `bpsk`, `qpsk`, `2fsk`, or `auto`; use `X-DEmod-Timing-Offset` and `X-DEmod-Carrier-Offset` when an analyst has those values. The MVP demodulator has deliberately **no implicit timing recovery**. A fixed samples-per-symbol/timing override makes every hard decision reproducible and avoids an unsupported blind-decoding claim.

For BPSK/QPSK it applies DC removal, optional fixed CFO correction, integrate-and-dump sampling, fixed hard constellation decisions and EVM. For 2-FSK it uses the conventional one-sample phase-difference frequency discriminator, then integrate-and-dump and a binary slicer. Results contain only candidate hard bits, not a protocol/payload claim.

## GNU Radio path

When installed on the analyst workstation, GNU Radio is the preferred interactive execution engine. DEmod emits the blocks and parameters for a reviewable GRC graph, documented in [gnuradio/README.md](../gnuradio/README.md). The proposed PSK graph follows GNU Radio's generic receiver design: frequency acquisition, PFB clock synchronization, constellation/Costas-loop phase tracking and bit unpacking. The FSK graph uses its quadrature frequency discriminator, Symbol Sync and Binary Slicer. GNU Radio documents these blocks and their functions in its [digital-modulation manual](https://www.gnuradio.org/doc/doxygen/page_digital.html), [Constellation Receiver reference](https://wiki.gnuradio.org/index.php/Constellation_Receiver), and [Quadrature Demod reference](https://wiki.gnuradio.org/index.php/Quadrature_Demod).

The development runtime does not currently include GNU Radio, so the API reports its availability and a graph descriptor rather than falsely claiming execution. Its pure-Python fallback exists only for controlled, manually parameterized captures.

## Validation boundary

Acceptance currently means that synthetic, known BPSK/QPSK samples produce the expected hard-decision sequence and that the configuration/provenance is emitted. The next receiver milestone is genuinely blind timing/carrier recovery and BER/SER on held-out, authorised captures. Experimental [FEC/interleaver identification](FEC_AND_INTERLEAVER_REVIEW.md) now runs on full candidate bits and records an analyst frame-offset hypothesis. It does not establish frame synchronization or exact permutation recovery. FEC decoding, verified deinterleaving, decryption and human-readable telemetry remain outside this MVP.
