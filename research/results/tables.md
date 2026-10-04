## CFO results

#### AWGN, all modulations (3960 signals)

All 11 modulations, SNR sweep, CFO +-0.2, L=1024 (Chen 2023 Fig. 3)

| Method | MSE | RMSE | Median abs err | Gross errors | RMSE 0 dB | 10 dB | 20 dB | 30 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repo DSP (lag-1 on x^4) | 0.0217 | 0.147 | 0.0673 | 70.4% | 0.142 | 0.147 | 0.156 | 0.15 |
| Kay (x) | 0.0025 | 0.05 | 0.00789 | 59.4% | 0.046 | 0.0188 | 0.0192 | 0.0196 |
| Kay (x^M, M known) | 0.00712 | 0.0844 | 0.023 | 69.2% | 0.0853 | 0.0695 | 0.0802 | 0.0772 |
| Luise-Reggiannini (M known) | 0.00939 | 0.0969 | 0.0669 | 85.9% | 0.0973 | 0.0952 | 0.0973 | 0.0934 |
| Periodogram (M known) | 0.0108 | 0.104 | 0.0137 | 60.6% | 0.0952 | 0.0962 | 0.104 | 0.103 |
| Periodogram (M blind) | 0.0142 | 0.119 | 0.0273 | 65.6% | 0.103 | 0.103 | 0.116 | 0.117 |
| IQ-ResNet (Chen 2023) | 0.00128 | 0.0357 | 0.0122 | 77.1% | 0.0224 | 0.0157 | 0.0142 | 0.0167 |
| CNN (O'Shea 2017) | 0.000465 | 0.0216 | 0.00538 | 52.7% | 0.0115 | 0.00953 | 0.00894 | 0.00731 |
| SpecCFO network only | 0.000436 | 0.0209 | 0.000684 | 13.5% | **0.00826** | 0.00406 | 0.00135 | **0.00159** |
| SpecCFO v1 (paper conditions) | 0.000436 | 0.0209 | **0.00035** | 13.5% | 0.00827 | **0.00402** | **0.00134** | 0.00166 |
| SpecCFO v2 (augmented + position channels, proposed) | **0.000274** | **0.0165** | 0.000434 | 16.5% | 0.00968 | 0.00603 | 0.00177 | 0.00199 |

Units are cycles per sample. Multiply by the sample rate for Hz (0.001 is 250 Hz at 250 kS/s). Gross error means more than 0.005 off.

#### AWGN, narrow offset range (easier reading of Chen 2023) (3960 signals)

All modulations at 8 samples/symbol, CFO +-0.025 cycles/sample (= +-0.2 cycles/symbol), the easier reading of Chen 2023

| Method | MSE | RMSE | Median abs err | Gross errors | RMSE 0 dB | 10 dB | 20 dB | 30 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repo DSP (lag-1 on x^4) | 0.00184 | 0.0429 | 0.00529 | 50.9% | 0.0605 | 0.0192 | 0.015 | 0.0114 |
| Kay (x) | **6.36e-05** | **0.00798** | 0.00345 | 36.4% | 0.00741 | 0.00492 | 0.0043 | 0.00486 |
| Kay (x^M, M known) | 8.44e-05 | 0.00919 | 0.00398 | 43.2% | 0.0108 | 0.00691 | 0.00455 | 0.00489 |
| Luise-Reggiannini (M known) | 0.000277 | 0.0166 | 0.00422 | 46.1% | 0.016 | 0.017 | 0.0126 | 0.0133 |
| Periodogram (M known) | 0.00365 | 0.0604 | 0.00631 | 52.0% | 0.0558 | 0.0556 | 0.0392 | 0.0398 |
| Periodogram (M blind) | 0.00425 | 0.0652 | 0.0135 | 56.7% | 0.0541 | 0.0501 | 0.0334 | 0.0365 |
| IQ-ResNet (Chen 2023) | 0.000314 | 0.0177 | 0.0106 | 74.5% | 0.0167 | 0.016 | 0.0151 | 0.0142 |
| CNN (O'Shea 2017) | 0.000176 | 0.0133 | 0.00444 | 45.7% | 0.00856 | 0.00655 | 0.00579 | 0.00664 |
| SpecCFO network only | 0.00036 | 0.019 | 0.000709 | 12.5% | 0.0143 | 0.00186 | 0.00134 | 0.00161 |
| SpecCFO v1 (paper conditions) | 0.00036 | 0.019 | **0.000383** | 12.7% | 0.0143 | **0.00183** | **0.00132** | **0.00158** |
| SpecCFO v2 (augmented + position channels, proposed) | 6.4e-05 | 0.008 | 0.000422 | 13.0% | **0.00641** | 0.00199 | 0.00137 | **0.00158** |

Units are cycles per sample. Multiply by the sample rate for Hz (0.001 is 250 Hz at 250 kS/s). Gross error means more than 0.005 off.

#### Rayleigh multipath, all modulations (1980 signals)

Same with exponential-profile Rayleigh multipath, spread 0.5-2 samples

| Method | MSE | RMSE | Median abs err | Gross errors | RMSE 0 dB | 10 dB | 20 dB | 30 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repo DSP (lag-1 on x^4) | 0.0213 | 0.146 | 0.0651 | 86.2% | 0.128 | 0.151 | 0.158 | 0.152 |
| Kay (x) | 0.00295 | 0.0543 | 0.0134 | 67.6% | 0.0516 | 0.0234 | 0.0247 | 0.032 |
| Kay (x^M, M known) | 0.00714 | 0.0845 | 0.0297 | 72.1% | 0.0834 | 0.0742 | 0.0756 | 0.0745 |
| Luise-Reggiannini (M known) | 0.00948 | 0.0974 | 0.0671 | 86.3% | 0.0957 | 0.0944 | 0.0976 | 0.0975 |
| Periodogram (M known) | 0.0109 | 0.104 | 0.0195 | 65.1% | 0.0923 | 0.109 | 0.0985 | 0.104 |
| Periodogram (M blind) | 0.0134 | 0.116 | 0.0317 | 69.1% | 0.0951 | 0.11 | 0.103 | 0.111 |
| IQ-ResNet (Chen 2023) | 0.00198 | 0.0445 | 0.016 | 82.8% | 0.0392 | 0.0292 | 0.0264 | 0.0386 |
| CNN (O'Shea 2017) | 0.00139 | 0.0372 | 0.00909 | 67.8% | 0.0359 | 0.0307 | 0.0249 | 0.0366 |
| SpecCFO network only | 0.00163 | 0.0404 | 0.00126 | 27.2% | 0.0371 | 0.0327 | **0.0144** | 0.037 |
| SpecCFO v1 (paper conditions) | 0.00164 | 0.0405 | **0.000979** | 27.6% | 0.0371 | 0.0327 | **0.0144** | 0.037 |
| SpecCFO v2 (augmented + position channels, proposed) | **0.0007** | **0.0265** | 0.00137 | 31.8% | **0.0308** | **0.0221** | 0.0159 | **0.0199** |

Units are cycles per sample. Multiply by the sample rate for Hz (0.001 is 250 Hz at 250 kS/s). Gross error means more than 0.005 off.

#### RMSE by modulation, SNR 10 dB and above, AWGN

| Method | 16qam | 2fsk | 4fsk | 64qam | 8psk | am | bpsk | fm | gmsk | ofdm | qpsk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repo DSP (lag-1 on x^4) | 0.144 | 0.147 | 0.154 | 0.15 | 0.15 | 0.164 | 0.149 | 0.161 | 0.147 | 0.146 | 0.155 |
| Kay (x) | 0.0102 | 0.00578 | 0.00922 | 0.00958 | 0.0101 | 0.000406 | 0.0378 | 0.004 | 0.00377 | 0.0423 | 0.0149 |
| Kay (x^M, M known) | 0.119 | 0.00578 | 0.00922 | 0.129 | 0.116 | 0.000406 | 0.0248 | 0.004 | 0.00377 | 0.0423 | 0.131 |
| Luise-Reggiannini (M known) | 0.113 | 0.0747 | 0.0727 | 0.121 | 0.116 | 0.0879 | 0.0994 | 0.0694 | 0.0625 | 0.0846 | 0.122 |
| Periodogram (M known) | 0.141 | 0.084 | 0.0599 | 0.152 | 0.118 | 3.34e-06 | 2.44e-06 | 0.00952 | 0.0166 | 0.15 | 0.155 |
| Periodogram (M blind) | 0.127 | 0.0967 | 0.127 | 0.127 | 0.115 | 3.34e-06 | 2.44e-06 | 0.105 | 0.119 | 0.14 | 0.155 |
| IQ-ResNet (Chen 2023) | 0.0137 | 0.0157 | 0.0183 | 0.0131 | 0.0133 | 0.0111 | 0.0157 | 0.0137 | 0.0126 | 0.0241 | 0.0136 |
| CNN (O'Shea 2017) | 0.00557 | 0.0111 | 0.0105 | 0.0062 | 0.00561 | 0.00458 | 0.00587 | 0.00495 | 0.00519 | 0.0159 | 0.00624 |
| SpecCFO network only | 0.000665 | 0.00179 | 0.00634 | 0.00081 | 0.00817 | 0.00027 | 0.00047 | 0.00449 | 0.000589 | 0.00129 | 0.000538 |
| SpecCFO v1 (paper conditions) | 0.00048 | 0.0019 | 0.0065 | 0.000697 | 0.00815 | 3.34e-06 | 2.44e-06 | 0.0045 | 0.0001 | 0.00141 | 5.61e-06 |
| SpecCFO v2 (augmented + position channels, proposed) | 0.000291 | 0.00193 | 0.00809 | 0.000608 | 0.00468 | 3.34e-06 | 2.44e-06 | 0.00494 | 0.000284 | 0.00139 | 5.61e-06 |

#### Oversampling (BPSK, Chen 2023 Fig. 5)

RMSE averaged over SNR 0 to 30 dB.

| Method | sps 4 | sps 8 | sps 16 |
| --- | ---: | ---: | ---: |
| Repo DSP (lag-1 on x^4) | 0.154 | 0.151 | 0.154 |
| Kay (x) | 0.0375 | 0.0225 | 0.0199 |
| Kay (x^M, M known) | 0.053 | 0.044 | 0.048 |
| Luise-Reggiannini (M known) | 0.107 | 0.0989 | 0.096 |
| Periodogram (M known) | **6.68e-06** | **6.08e-06** | **6.78e-06** |
| Periodogram (M blind) | 0.0211 | 0.0122 | 0.0386 |
| IQ-ResNet (Chen 2023) | 0.0204 | 0.0142 | 0.0135 |
| CNN (O'Shea 2017) | 0.00735 | 0.00497 | 0.00635 |
| SpecCFO network only | 0.00045 | 0.000399 | 0.000375 |
| SpecCFO v1 (paper conditions) | **6.68e-06** | **6.08e-06** | **6.78e-06** |
| SpecCFO v2 (augmented + position channels, proposed) | **6.68e-06** | **6.08e-06** | **6.78e-06** |

#### Signal length (BPSK, Chen 2023 Fig. 6)

RMSE averaged over SNR 0 to 30 dB.

| Method | L 512 | L 1024 | L 2048 |
| --- | ---: | ---: | ---: |
| Repo DSP (lag-1 on x^4) | 0.155 | 0.144 | 0.156 |
| Kay (x) | 0.0254 | 0.0222 | 0.024 |
| Kay (x^M, M known) | 0.0512 | 0.045 | 0.0503 |
| Luise-Reggiannini (M known) | 0.105 | 0.0968 | 0.0992 |
| Periodogram (M known) | **1.93e-05** | **6.5e-06** | **2.3e-06** |
| Periodogram (M blind) | 0.0366 | 0.0211 | 0.0122 |
| IQ-ResNet (Chen 2023) | n/a | 0.0155 | n/a |
| CNN (O'Shea 2017) | n/a | 0.00484 | n/a |
| SpecCFO network only | 0.000528 | 0.000413 | 0.000352 |
| SpecCFO v1 (paper conditions) | **1.93e-05** | **6.5e-06** | **2.3e-06** |
| SpecCFO v2 (augmented + position channels, proposed) | 0.000366 | **6.5e-06** | **2.3e-06** |

#### O'Shea 2017 setup: error standard deviation in Hz at 400 kS/s, SNR 5 dB

QPSK, RRC 0.25, 4 samples/symbol, CFO within +-50 kHz. Lower is better. CNNs trained at 1024 samples are scored only there.

**awgn**

| Method | L=64 | 128 | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Periodogram (M known) | 3.729e+04 | 2.825e+04 | 2.071e+04 | 9.64e+03 | **7.37** |
| Periodogram (M blind) | 3.021e+04 | 2.451e+04 | 2.341e+04 | 9.49e+03 | **7.37** |
| Kay (x^M, M known) | 2.702e+04 | 2.913e+04 | 2.758e+04 | 2.906e+04 | 2.839e+04 |
| IQ-ResNet (Chen 2023) | n/a | n/a | n/a | n/a | 6.02e+03 |
| CNN (O'Shea 2017) | n/a | n/a | n/a | n/a | 2.21e+03 |
| SpecCFO network only | 4.548e+04 | 2.559e+04 | 2.72e+03 | 756 | 482 |
| SpecCFO v1 (paper conditions) | 4.548e+04 | 2.559e+04 | 2.73e+03 | **688** | 420 |
| SpecCFO v2 (augmented + position channels, proposed) | **1.069e+04** | **6.4e+03** | **2.51e+03** | 721 | 312 |

**sigma0.5**

| Method | L=64 | 128 | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Periodogram (M known) | 3.750e+04 | 3.599e+04 | 2.971e+04 | 1.907e+04 | 1.228e+04 |
| Periodogram (M blind) | 3.049e+04 | 2.940e+04 | 2.571e+04 | 1.931e+04 | 1.099e+04 |
| Kay (x^M, M known) | 2.847e+04 | 2.653e+04 | 2.901e+04 | 2.787e+04 | 2.785e+04 |
| IQ-ResNet (Chen 2023) | n/a | n/a | n/a | n/a | 1.058e+04 |
| CNN (O'Shea 2017) | n/a | n/a | n/a | n/a | 8.38e+03 |
| SpecCFO network only | 4.951e+04 | 3.464e+04 | 1.525e+04 | **4.21e+03** | 3.52e+03 |
| SpecCFO v1 (paper conditions) | 4.951e+04 | 3.464e+04 | 1.526e+04 | **4.21e+03** | **3.49e+03** |
| SpecCFO v2 (augmented + position channels, proposed) | **1.299e+04** | **8.7e+03** | **7.49e+03** | 5.15e+03 | 3.84e+03 |

**sigma1**

| Method | L=64 | 128 | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Periodogram (M known) | 4.328e+04 | 3.695e+04 | 3.258e+04 | 2.864e+04 | 2.219e+04 |
| Periodogram (M blind) | 2.909e+04 | 3.091e+04 | 3.271e+04 | 2.542e+04 | 2.339e+04 |
| Kay (x^M, M known) | 3.116e+04 | 2.810e+04 | 2.755e+04 | 2.735e+04 | 2.891e+04 |
| IQ-ResNet (Chen 2023) | n/a | n/a | n/a | n/a | 1.387e+04 |
| CNN (O'Shea 2017) | n/a | n/a | n/a | n/a | 1.208e+04 |
| SpecCFO network only | 4.468e+04 | 3.482e+04 | 1.676e+04 | 5.69e+03 | **5.2e+03** |
| SpecCFO v1 (paper conditions) | 4.468e+04 | 3.482e+04 | 1.676e+04 | **5.68e+03** | 5.21e+03 |
| SpecCFO v2 (augmented + position channels, proposed) | **1.437e+04** | **1.254e+04** | **8.48e+03** | 6.96e+03 | 5.62e+03 |

**sigma2**

| Method | L=64 | 128 | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Periodogram (M known) | 3.511e+04 | 3.758e+04 | 3.287e+04 | 3.569e+04 | 2.847e+04 |
| Periodogram (M blind) | 2.993e+04 | 3.249e+04 | 2.827e+04 | 2.728e+04 | 2.656e+04 |
| Kay (x^M, M known) | 2.829e+04 | 2.770e+04 | 2.944e+04 | 2.969e+04 | 2.841e+04 |
| IQ-ResNet (Chen 2023) | n/a | n/a | n/a | n/a | 1.309e+04 |
| CNN (O'Shea 2017) | n/a | n/a | n/a | n/a | 1.086e+04 |
| SpecCFO network only | 4.747e+04 | 4.539e+04 | 1.636e+04 | 1.073e+04 | **6.28e+03** |
| SpecCFO v1 (paper conditions) | 4.747e+04 | 4.539e+04 | 1.636e+04 | 1.073e+04 | **6.28e+03** |
| SpecCFO v2 (augmented + position channels, proposed) | **1.631e+04** | **1.498e+04** | **1.037e+04** | **7.41e+03** | 7.41e+03 |

#### The repo's own recipes and metrics (120 signals, 1 MS/s, 4096 samples)

The README's CFO agreement tolerance is 250 Hz and its DC tolerance is 0.05. Signals come from the repo's recipe generator, which SpecCFO never saw in training.

| Carrier offset | RMSE (Hz) | Median abs err (Hz) | Within 250 Hz | Within 25 Hz |
| --- | ---: | ---: | ---: | ---: |
| Repo DSP (lag-1 on x^4) | 3.483e+04 | 2.51e+03 | 28% | 18% |
| Shipped TinyMLP | 1.26e+03 | 622 | 18% | 1% |
| SpecCFO v1 (paper conditions) | 2.515e+05 | 1.250e+05 | 23% | 20% |
| Ablation: v2 data, no position channels | 2.975e+05 | 2.500e+05 | 18% | 17% |
| SpecCFO v2 (augmented + position channels, proposed) | 1.568e+04 | 7.99 | 64% | 62% |

| DC offset | RMSE | Within 0.05 | Within 0.005 |
| --- | ---: | ---: | ---: |
| Sample mean (DSP) | 0.0212 | 99% | 5% |
| Shipped TinyMLP | 0.0158 | 100% | 18% |


## Modulation classification results

#### In distribution (CFO within +-0.05)

| Model | Accuracy | Accuracy SNR>=0 | Macro F1 | ECE | Coverage / accuracy kept |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shipped centroid (4 classes) | 0.252 | 0.25 | 0.199 | n/a | 67% / 24.7% |
| Centroid retrained | 0.227 | 0.25 | 0.18 | n/a | n/a |
| Cumulants (Swami 2000) | 0.272 | 0.299 | 0.155 | n/a | n/a |
| Cumulants + SpecCFO derotation | 0.401 | 0.476 | 0.365 | n/a | n/a |
| VT-CNN2 (O'Shea 2016) | 0.265 | 0.306 | 0.205 | 0.0495 | n/a |
| LSTM (Rajendran 2018) | 0.354 | 0.41 | 0.311 | 0.0718 | n/a |
| DemodAMC, no CFO removal | **0.777** | **0.903** | **0.775** | 0.0228 | n/a |
| DemodAMC v1 (paper conditions) | 0.762 | **0.903** | 0.761 | 0.0181 | 63% / 95.1% |
| DemodAMC v2 (augmented, proposed) | 0.762 | 0.899 | 0.76 | 0.0113 | 62% / 96.2% |

Classes covered differ by baseline: cumulants cover 5 linear modulations only, the shipped centroid 4. Their rows are scored on those classes only. The 12-class models are scored on all 12, and the next table scores them on the same subsets.

| Same subset, 12-class models restricted | 5 linear classes | BPSK/QPSK/8PSK/16QAM |
| --- | ---: | ---: |
| Shipped centroid | n/a | 0.252 |
| Cumulants (Swami 2000) | 0.272 | n/a |
| Cumulants + SpecCFO derotation | 0.401 | n/a |
| VT-CNN2 (O'Shea 2016) | 0.227 | 0.284 |
| LSTM (Rajendran 2018) | 0.267 | 0.33 |
| DemodAMC, no CFO removal | 0.699 | 0.79 |
| DemodAMC v1 (paper conditions) | 0.72 | 0.826 |
| DemodAMC v2 (augmented, proposed) | 0.691 | 0.812 |

#### Carrier offset shift (CFO within +-0.2, 4x wider than training)

| Model | Accuracy | Accuracy SNR>=0 | Macro F1 | ECE | Coverage / accuracy kept |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shipped centroid (4 classes) | 0.242 | 0.246 | 0.189 | n/a | 67% / 25.5% |
| Centroid retrained | 0.234 | 0.261 | 0.185 | n/a | n/a |
| Cumulants (Swami 2000) | 0.247 | 0.265 | 0.142 | n/a | n/a |
| Cumulants + SpecCFO derotation | 0.389 | 0.46 | 0.353 | n/a | n/a |
| VT-CNN2 (O'Shea 2016) | 0.21 | 0.243 | 0.151 | 0.109 | n/a |
| LSTM (Rajendran 2018) | 0.214 | 0.244 | 0.167 | 0.128 | n/a |
| DemodAMC, no CFO removal | 0.708 | 0.822 | 0.696 | 0.0308 | n/a |
| DemodAMC v1 (paper conditions) | **0.77** | **0.908** | **0.768** | 0.0205 | 64% / 95.1% |
| DemodAMC v2 (augmented, proposed) | 0.768 | 0.905 | 0.766 | 0.0195 | 62% / 96.7% |

Classes covered differ by baseline: cumulants cover 5 linear modulations only, the shipped centroid 4. Their rows are scored on those classes only. The 12-class models are scored on all 12, and the next table scores them on the same subsets.

| Same subset, 12-class models restricted | 5 linear classes | BPSK/QPSK/8PSK/16QAM |
| --- | ---: | ---: |
| Shipped centroid | n/a | 0.242 |
| Cumulants (Swami 2000) | 0.247 | n/a |
| Cumulants + SpecCFO derotation | 0.389 | n/a |
| VT-CNN2 (O'Shea 2016) | 0.225 | 0.276 |
| LSTM (Rajendran 2018) | 0.248 | 0.298 |
| DemodAMC, no CFO removal | 0.621 | 0.723 |
| DemodAMC v1 (paper conditions) | 0.716 | 0.821 |
| DemodAMC v2 (augmented, proposed) | 0.678 | 0.795 |


DemodAMC v1 (paper conditions) calibration (validation set, disjoint from test): temperature 0.97, abstain below 0.72 top probability, chosen so accepted predictions reach 95% accuracy on validation.


DemodAMC v2 (augmented, proposed) calibration (validation set, disjoint from test): temperature 0.94, abstain below 0.74 top probability, chosen so accepted predictions reach 95% accuracy on validation.


#### The repo's own recipe signals (120 signals, 4 classes, 1 MS/s, 4096 samples)

The shipped centroid classifier was trained on these recipes, so this is its home distribution. The generator emits one independent symbol per sample (see research/README.md), a case only the v2 models were trained on.

| Model | Accuracy | Clean stage | Impaired stage | Coverage / accuracy kept |
| --- | ---: | ---: | ---: | ---: |
| Shipped centroid (4 classes) | 0.758 | 0.933 | 0.583 | 68% / 88.9% |
| VT-CNN2 (O'Shea 2016) | 0.267 | 0.167 | 0.367 | n/a |
| LSTM (Rajendran 2018) | 0.3 | 0.333 | 0.267 | n/a |
| DemodAMC, no CFO removal | 0.417 | 0.233 | 0.6 | n/a |
| DemodAMC v1 (paper conditions) | 0.683 | 0.75 | 0.617 | n/a |
| DemodAMC v2 (augmented, proposed) | **0.95** | **1** | **0.9** | n/a |
