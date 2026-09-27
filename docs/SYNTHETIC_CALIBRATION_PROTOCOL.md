# Synthetic-to-Real Calibration Protocol

## Target and provenance

The initial real target is the MIT-licensed [POWDER Co-Channel Protocol Dataset](https://huggingface.co/datasets/T-Arshad/POWDER_CoChannel_Protocol_Dataset): OTA complex64 captures at 33.33 MS/s on a shared 20 MHz channel, containing 802.11a Wi-Fi, 4G LTE and 5G NR mixtures. Each local calibration window retains the original JSON receiver/transmitter metadata.

This target is deliberately narrower than “all IQ files.” A synthetic generator cannot honestly be declared indistinguishable from arbitrary, unobserved RF environments.

## Change log

| Stage | Change | Why it is present | Provenance |
| --- | --- | --- | --- |
| 0 | Linear PSK/QAM baseline | Establishes basic decoder labels | DEmod MVP design |
| 1 | AWGN, CFO, DC, I/Q imbalance, phase noise, multipath, clipping and interference controls | Models known RF front-end/channel non-idealities | [Tarable et al., 2021](https://doi.org/10.1109/ACCESS.2021.3101845); [Chen et al., 2021](https://doi.org/10.1049/cmu2.12077) |
| 2 | 64-carrier / 16-sample CP, 52-active-carrier OFDM shape at 20 MHz; fractional conversion to 33.333 MS/s; 1–2 co-channel emitters; calibrated receiver RMS | The real corpus is a 20 MHz multi-protocol OFDM environment, so random linear modulation would create a structural shortcut | [POWDER dataset card](https://huggingface.co/datasets/T-Arshad/POWDER_CoChannel_Protocol_Dataset); [RF-Diffusion](https://arxiv.org/abs/2404.09140) |

The stage-2 waveform is an **802.11a-shaped OFDM proxy**, not a standards-compliant Wi-Fi/LTE/NR implementation and not a synthetic reproduction of a particular capture.

The stage-2 `target_rms` values are receiver-scale bins measured from 48 deterministic, provenance-distinct calibration requests (seed 26147, one 4,096-sample range each). They are aggregate calibration statistics, not replayed real I/Q or labels; the validation set must be disjoint from those capture URLs.

The stage-2 carrier-offset range is centred near -0.9 MHz because that same calibration subset had mean adjacent-sample phase change of about -0.17 rad at 33.33 MS/s. This is a corpus-specific frequency-placement estimate and must be re-estimated for each receiver/centre-frequency stratum.

## Acceptance test

1. Split by source capture (and ideally collection round/gain), never random windows from the same capture across train and test.
2. Train multiple held-out discriminators on independent time, spectral, cyclostationary and learned I/Q representations.
3. Report each accuracy with a two-sided 95% Wilson interval.
4. Accept a family only if each preregistered discriminator is near 50% **and** the 95% half-width is at most 5 percentage points on an untouched holdout. For a binomial accuracy near 50%, that needs roughly 384 independent held-out examples; six source captures cannot establish it.
5. Separately verify downstream parameter-estimation and demodulation performance. Making a weak discriminator fail is never enough.

## Current honest status

The first stage-2 run reaches 0.50 on the legacy six-example held-out diagnostic after matching waveform class, sample rate and receiver scale. Its interval is necessarily wide, so it **does not pass** the precision gate. The next required artifact is a larger provenance-stratified holdout, not a claim of indistinguishability.
