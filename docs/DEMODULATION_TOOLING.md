# PSK, QAM and FSK receiver tooling

## Decision and literature

Use conventional library receivers after the existing AMC/CFO/SPS estimators, rather than train an AI model to imitate an available symbol slicer. Modulation classification selects a hypothesis; it is not demodulation. Correct bits additionally require pulse/timing/carrier settings and the transmitted bit mapping.

| Source | Relevant strategy | Implementation and boundary |
| --- | --- | --- |
| [Komm constellation API](https://komm.dev/ref/Constellation/), [QAM](https://komm.dev/ref/QAMConstellation/), [PSK](https://komm.dev/ref/PSKConstellation/) | Nearest constellation decisions, with constellation geometry distinct from bit labeling. | `komm.Constellation.closest_indices` performs hard decisions for all seven receiver paths. No replacement hand-written nearest-neighbour demodulator. |
| [SciPy convolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html) | FIR receive/matched filtering. | Rectangular integrate-and-dump or RRC receive filtering. RRC taps reuse the existing six-symbol-span implementation; rolloff and pulse are analyst hypotheses, not detected automatically. |
| [GNU Radio generic receiver source](https://github.com/gnuradio/gnuradio/blob/main/gr-digital/python/digital/generic_mod_demod.py), [digital manual](https://www.gnuradio.org/doc/doxygen/page_digital.html) | AGC, band-edge frequency acquisition, polyphase RRC clock synchronization, constellation carrier/phase tracking and bit unpacking. | Optional executable `gnuradio/run_receiver.py` delegates PSK/QAM to `digital.generic_demod`. Requires real GNU Radio 3.10; absent and not runtime-validated here. No claim that a graph descriptor executes. |
| [GNU Radio PSK tutorial](https://wiki.gnuradio.org/index.php?title=Guided_Tutorial_PSK_Demodulation) | Carrier/timing recovery does not remove the constellation's rotational bit-mapping ambiguity. | Reports explicitly retain phase and mapping hypotheses. Auto phase searches one fundamental rotation sector; without a known preamble/differential convention it cannot recover absolute bit labels. |
| Gardner, *A BPSK/QPSK Timing-Error Detector for Sampled Receivers*, 1986, [DOI](https://doi.org/10.1109/TCOM.1986.1096561), [GNU Radio Symbol Sync](https://wiki.gnuradio.org/index.php/Symbol_Sync) | Timing-error feedback with an interpolating clock synchronizer. | Continuous/fractional clock tracking is a separate receiver capability. The Python burst path only searches **static integer offsets** and must not be called a Gardner implementation. |
| [GNU Radio Quadrature Demod](https://wiki.gnuradio.org/index.php/Quadrature_Demod), [GFSK source](https://github.com/gnuradio/gnuradio/blob/main/gr-digital/python/digital/gfsk.py) | Complex conjugate-product phase differences form a frequency discriminator before timing and symbol decisions. | Python uses phase differences within symbol windows and Komm real-level decisions for unshaped 2/4-CPFSK. Gaussian-shaped FSK/MSK/GMSK need separate matched/sequence receivers and are rejected by this pulse contract. |
| [O'Shea et al., 2017](https://arxiv.org/abs/1707.06260) | Learned physical-parameter estimation can precede an analytic receiver. | Existing SpecCFO and SPS/AMC supply upstream hypotheses with provenance; this work does not train another demodulation network or claim a neural receiver improvement. |

## Implemented modes and mapping

| Mode | Receive filtering | Symbol-to-bit convention |
| --- | --- | --- |
| BPSK | Rectangular/RRC | Negative = 0, positive = 1 (legacy compatible) |
| QPSK | Rectangular/RRC | I sign then Q sign, 00=(-,-), 01=(-,+), 10=(+,-), 11=(+,+) |
| 8-PSK | Rectangular/RRC | Unit-circle angular order from 0 radians, reflected Gray labels, MSB first |
| 16/64-QAM | Rectangular/RRC | Ascending I then Q axis, independently reflected Gray labels, I bits before Q bits |
| 2/4-FSK | Unshaped CPFSK discriminator | Ascending corrected tone frequencies, reflected Gray labels, MSB first |

Conventions are not universally used by real protocols. Phase rotation, conjugation, axis inversion, differential coding and unknown labeling can give a low EVM yet completely wrong bits. Random or missing tone levels can invalidate FSK clustering. EVM and cluster residual are fit measures, not BER/confidence/proof of payload correctness.

## Simple process

Install in the Python environment running the API (this pinned receiver stack is tested on Python 3.12; the standard-library baseline retains its separate older Python support):

```powershell
python -m pip install -r requirements-receiver.txt
python src/local_comparison_api.py --port 8789
```

The project also supports optional isolated libraries in `.venv/receiver-deps`. `backend=auto` uses Komm/SciPy when available; the old three-mode fixed receiver is retained as an explicitly labeled fallback. New modes, RRC or automatic acquisition require the libraries and fail explicitly if unavailable.

CLI, preserving the original file and exporting **all candidate bits**, not just the UI preview:

```powershell
python src/receive_capture.py capture.iq --output candidate-bits.json --iq-format s16le --sample-rate 250000 --modulation 16qam --sps 8 --pulse rrc --rolloff 0.35 --timing auto --phase auto --cfo 750
```

Output is created exclusively: an existing output file is not overwritten. WAV uses the existing header/provenance parser; stereo I/Q or real-IF interpretation must be explicitly declared. Already-demodulated voice/audio is not sent to RF demodulators.

`POST /demodulate` retains the existing IQ/WAV contract and adds:

| Header | Values |
| --- | --- |
| `X-DEmod-Modulation` | bpsk/qpsk/8psk/16qam/64qam/2fsk/4fsk/auto |
| `X-DEmod-Receiver-Backend` | auto/komm_scipy/legacy_fixed |
| `X-DEmod-Pulse` | rect/rrc |
| `X-DEmod-RRC-Rolloff` | (0,1], default 0.35 |
| `X-DEmod-Timing-Offset` | integer offset, or auto |
| `X-DEmod-Phase-Radians` | fixed correction in radians, or auto |
| `X-DEmod-FSK-Tones-Hz` | ascending comma-separated frequencies **after CFO correction**, optional |

AMC-guided receivers now accept all seven supported modulation labels. Automatic SPS remains restricted to linear PSK/QAM and integer 2/4/8/16; FSK requires manual SPS. Other modes abstain. Explicit zero CFO remains a manual override. UI plots show the actual QAM/PSK points or FSK tone thresholds, with applied settings in JSON. Full receiver bits feed the existing bounded FEC candidate identifier, not its 512-bit display preview.

## Evaluation and remaining work

[Measured BER/SER](../research/results/receiver_results.md) and [recipe/count/provenance JSON](../research/results/receiver_results.json) come from `research/eval_receivers.py`. Tests assert known mappings, seven-mode recovery, RRC/static acquisition, invalid settings, API routing and an intentional phase-ambiguity counterexample. The generator is a controlled test harness, not an intercepted corpus or published benchmark.

Not implemented in the Python path: fractional/noninteger symbol timing, sample-clock drift tracking, fine carrier tracking, adaptive equalization, automatic pulse selection, differential/absolute mapping resolution, frame/CRC validation, FEC decoding or decryption. GNU Radio's optional tracking path is supplied but not falsely counted as tested. Nothing here turns encrypted data into plaintext.
