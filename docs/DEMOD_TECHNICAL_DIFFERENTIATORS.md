# DEmod technical brief: evidence, technology and differentiation

## One-line proposition

**DEmod turns an unknown RF recording into an auditable chain of ranked hypotheses—not an unexplained modulation label.** It preserves the raw recording, makes capture-format uncertainty explicit, produces reproducible DSP evidence, and only permits a decoder/FEC claim when measurable gates are passed.

## What exists today

| Capability | Technology | Status |
| --- | --- | --- |
| IQ/WAV ingestion | Python standard library; explicit raw layouts `s16le`, `s16be`, `s8`, `cu8`, `f32le`, `f32be` | Implemented |
| True visual analysis | NumPy complex FFT/PSD, Hann-window STFT waterfall, sampled constellation, robust energy segmentation | Implemented backend |
| Provenance | Immutable raw input, JSON report settings/axes, source manifest, browser run ledger | Implemented; browser preview is explicitly separate from backend DSP |
| Denoising safety rail | Candidate DC/impulse branch with raw preservation, masks and audit | Implemented but disabled/unwired |
| Synthetic corpus | Versioned HF/VHF/UHF recipes, deterministic seeds, hidden audit truth | Implemented |
| Modulation/FEC recovery | Bounded-hypothesis architecture and test plan | Designed; not yet implemented—no performance claim |
| ML | DSP baseline plus compact raw-IQ CNN/feature-fusion proposal | Designed; not yet trained or claimed |

## Why this differs from a run-of-the-mill RF dashboard

| Typical weak solution | DEmod differentiator | Why it matters for NTRO-style unlabeled captures |
| --- | --- | --- |
| Assumes `.iq` means one fixed datatype and rate | Treats representation, endianness and `Fs` as source facts or named hypotheses | Headerless bytes cannot uniquely provide these values; a confident but wrong parser contaminates every later plot/model |
| Shows a spectrum and an AI class label | Stores transform/window parameters, raw/derived branch, plot data, assumptions and limits in a report | An analyst can reproduce, challenge or tune a conclusion |
| Globally smooths the input before classification | Uses raw-first, reversible, gated denoising branches and preserves masks | A strong narrowband component, fast transition or radar pulse may be the target rather than noise |
| Uses “less readable/gibberish” as the decoder objective | Uses bounded model selection and scores sync, likelihood/EVM, cyclic evidence, framing and CRC before low-weight payload plausibility | Encrypted, compressed and binary telemetry are valid outputs that look non-textual |
| Trains only on a huge synthetic dump | Stores signal recipes, seed and audit truth; generates ephemeral batches and measures synthetic-to-real gap on held-out sessions | Avoids petabyte storage and makes every synthetic sample reproducible without leaking generation parameters to the model |
| Claims universal decoding | Uses an explicit supported-family catalogue and `abstain/unsupported` result | Honest failure is operationally safer and technically more credible than fabricated FEC/interleaver labels |
| Replaces DSP with a black-box model | Makes deterministic features the mandatory baseline and fuses them later with a compact raw-IQ model | Enables an interpretable MVP and exposes disagreement for analyst review |

## Novel contribution: the evidence graph

The main novelty is the **evidence graph**, not a claim of inventing FFT or QPSK. Each capture produces linked, versioned nodes:

```text
raw bytes -> representation hypothesis -> raw DSP measures -> optional derived branch
          -> segmentation -> modulation candidates -> supported receiver -> soft bits
          -> framing / FEC candidates -> result or abstention
```

Every arrow records settings, input/derived hashes, feature values, confidence and failure reason. This allows an analyst to compare raw and denoised branches, trace an apparent detection back to a plot and an input interpretation, and reject a faulty assumption without losing the original evidence.

## Research and implementation references

1. **Raw IQ and metadata:** [PySDR, *IQ Files and SigMF*](https://pysdr.org/content/iq_files) — interleaved IQ, explicit dtype, clipping and SigMF metadata practice.
2. **FFT/waterfall implementation:** [NumPy FFT reference](https://numpy.org/doc/stable/reference/routines.fft.html), [SciPy signal documentation](https://docs.scipy.org/doc/scipy/reference/signal.html), and [PySDR frequency-domain guide](https://pysdr.org/content/frequency_domain).
3. **Classical AMC taxonomy:** Dobre, Abdi, Bar-Ness & Su, 2007, [DOI 10.1049/iet-com:20050176](https://doi.org/10.1049/iet-com:20050176).
4. **Higher-order-cumulant AMC:** Abdelmutalab, Assaleh & El‑Tarhuni, 2016, [DOI 10.1016/j.phycom.2016.08.001](https://doi.org/10.1016/j.phycom.2016.08.001).
5. **Cyclostationary AMC:** Dobre et al., 2010, [DOI 10.1007/s11277-009-9776-2](https://doi.org/10.1007/s11277-009-9776-2).
6. **Blind carrier/symbol-rate candidates:** Zhang et al., 2012, [DOI 10.1016/j.proeng.2011.12.753](https://doi.org/10.1016/j.proeng.2011.12.753); Güner, 2014, [DOI 10.1002/dac.2606](https://doi.org/10.1002/dac.2606).
7. **Receiver impairments:** Tarable et al., 2021, [DOI 10.1109/ACCESS.2021.3101845](https://doi.org/10.1109/ACCESS.2021.3101845); Liu & Li, 2011, [DOI 10.1016/j.sigpro.2010.12.002](https://doi.org/10.1016/j.sigpro.2010.12.002); Song et al., 2017, [arXiv](https://arxiv.org/abs/1712.05970).
8. **Impulsive/RFI mitigation:** Hwang et al., 2017, [DOI 10.1587/transfun.E100.A.3041](https://doi.org/10.1587/transfun.E100.A.3041); Nita & Gary, 2010, [paper record](https://digitalcommons.njit.edu/fac_pubs/13405/); Taylor et al., 2018, [arXiv](https://arxiv.org/abs/1808.10365).
9. **Wavelet option, not universal preprocessing:** Baxter & Upton, 2002, [DOI 10.1111/1467-9876.00276](https://doi.org/10.1111/1467-9876.00276).
10. **Hybrid RFML direction and caveats:** Thakur & Imtiaz, 2026, [DOI 10.3390/electronics15102163](https://doi.org/10.3390/electronics15102163); Tian et al., 2026, [DOI 10.1016/j.sigpro.2025.110444](https://doi.org/10.1016/j.sigpro.2025.110444).
11. **Real/proxy dataset roles:** [Panoradio HF](https://panoradio-sdr.de/radio-signal-classification-dataset/), [LoRadar](https://zenodo.org/records/16302856), and the [SigMF RF framework](https://doi.org/10.5281/zenodo.4603987).

## Submission-safe novelty statement

> DEmod is an evidence-first, provenance-preserving RF analysis pipeline for uncertain IQ/WAV captures. Its novelty lies in representing capture-format uncertainty, reversible DSP branches and bounded modulation hypotheses in one auditable evidence graph—combining classical RF measurements with a planned compact AI fusion model, while abstaining instead of fabricating an unsupported decode.

This wording is defensible because it describes the system design and implemented constraints. It does not claim universal decryption, a trained production model, or accuracy on undisclosed NTRO recordings.
