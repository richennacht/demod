# WAV input workflow

A WAV container supplies encoding, channel count, frame count and sample rate. It does not prove that its channels are I/Q, contain RF rather than audio, or supply RF centre/gain. Decode the header before interpreting waveform samples. Original uploaded bytes and their SHA-256 remain the provenance anchor; conversion runs in memory and no intermediate IQ file is required.

## Supported interpretations

| Declaration | Processing | Limits |
| --- | --- | --- |
| Stereo I/Q or Q/I | Read channel samples directly as I+jQ (or swapped), normalize PCM, retain header Fs | Two channels only; I/Q assignment is an analyst hypothesis unless independently documented. |
| Mono RF/IF | Construct the positive-frequency analytic waveform, optionally translate a declared IF centre to baseband, then use FFT/STFT, rate/AMC and receivers | Requires a properly sampled real RF/IF waveform. Aliased or overlapping spectral images cannot be undone. FFT Hilbert conversion has finite-record edge effects; this does not create independent measured quadrature. |
| Already-demodulated audio | Local playback, level summary, spectrum/waterfall; RF classifier and receiver abstain | Speech/audio does not retain enough information to recreate the original RF modulation. No transcription or telemetry protocol decoding is claimed. |

The implementation uses Python `wave` for uncompressed PCM WAV (8/16/24/32 bit), with a NumPy FFT analytic-signal construction. This follows [Python's PCM/header contract](https://docs.python.org/3/library/wave.html) and the algorithm documented for [SciPy `signal.hilbert`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html). It does not require SciPy at runtime. Compressed, IEEE-float WAV, RF64 and arbitrary multi-channel RF interpretations are unsupported and receive a clear error. PCM extensible support follows the installed Python wave implementation.

## GUI and API

Upload WAV on Capture, read the automatically populated header facts, choose its waveform interpretation, and optionally set recorded IF centre. Header Fs is retained; a conflicting supplied Fs is rejected rather than silently resampling. Centre frequency and gain stay separately supplied values. Analyze renders the same IQ plots/model comparison for stereo/real-IF input, or an audio overview for audio. Receiver → Run AMC-guided receiver uses the selected energy region for eligible BPSK/QPSK. Manual overrides remain available. This is region selection for an inspectable candidate receiver, not a guarantee of maximum recovery.

`POST /analyse` and `/demodulate` accept a WAV body with `X-DEmod-WAV-Role: stereo_iq`, `stereo_qi`, `real_if` or `audio`. `X-DEmod-IF-Centre` is optional. Sample-rate header can be omitted; Fs comes from WAV. `/rates` also accepts a declared waveform role for normalized rate evidence. Undeclared waveform roles are rejected for analysis/demodulation.

The 16 MiB demo cap applies. WAV files over the cap are rejected before upload, rather than slicing a container into an invalid file. Malformed/truncated PCM payloads are rejected. Original-byte hashes, parsed header facts, channel role and conversion/translation are included in JSON exports. The raw analysis branch for mono real-IF is the declared analytic conversion before cleaning; it is not the original real PCM vector.

`python research/make_wav_example.py` wraps the existing synthetic QPSK IQ fixture in stereo PCM WAV without changing its waveform samples. **Load WAV example** demonstrates the full path. Validation covers I/Q channel ordering, PCM24 sign extension, header-rate conflicts, truncation rejection, analytic tone recovery and translation, identical raw/WAV BPSK decisions, and RF abstention for audio. Learned-model real-IF performance remains unvalidated.
