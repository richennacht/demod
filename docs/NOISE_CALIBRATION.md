# Noise and Interference Calibration

## Why the first discriminator separated domains perfectly

The `1.0` held-out accuracy came from only six test windows, so it is not a reliable accuracy estimate. More importantly, the real subset contains OTA Wi-Fi/LTE/5G protocol mixtures at 33.333 MS/s while the first recipe family contains isolated rectangular-pulse PSK/QAM at 1 MS/s. It can therefore exploit waveform family, occupied bandwidth, front-end scaling/quantization, co-channel activity, and unmodelled receiver/channel effects—not just noise. The score is a gap detector.

## Five additions

| Addition | Recipe keys | Why it matters |
| --- | --- | --- |
| Coloured receiver noise | `colored_noise_std`, `colored_noise_rho` | White Gaussian noise alone misses temporally correlated receiver/background noise. |
| Narrowband blocker / spur | `tone_interferer_amplitude`, `tone_interferer_hz` | Models a narrow interfering carrier or oscillator spur. |
| Bursty broadband interference | `burst_probability`, `burst_length`, `burst_amplitude` | Models temporally clustered interference rather than isolated samples. |
| Co-channel modulated interferer | `cochannel_interferer_amplitude`, `cochannel_interferer_hz` | Models a second QPSK signal sharing the capture band. |
| ADC quantization | `adc_bits` | Models finite ADC resolution after clipping. |

Existing recipe effects remain AWGN, CFO, DC offset, I/Q gain/phase imbalance, phase noise, isolated impulsive noise, clipping, and static multipath taps.

Phase noise, CFO, and I/Q imbalance are established RF-front-end impairment classes; their signal models and impact on EVM/SINR are surveyed in [Tarable et al., 2021](https://doi.org/10.1109/ACCESS.2021.3101845). Impulsive interference is commonly represented with Bernoulli-Gaussian or Middleton-class models, motivating separate isolated and bursty disturbance controls. [Chen et al., 2021](https://doi.org/10.1049/cmu2.12077) The real POWDER source explicitly contains co-channel OTA captures, so co-channel interference is essential rather than decorative. [POWDER dataset](https://huggingface.co/datasets/T-Arshad/POWDER_CoChannel_Protocol_Dataset)

## Calibration protocol

Enable one addition at a time, retrain the discriminator, and record its held-out score plus the downstream parameter-estimation error. Keep an effect only when it reduces the domain gap without degrading held-out real parameter/demodulation performance. Do not tune against the final real holdout set.
