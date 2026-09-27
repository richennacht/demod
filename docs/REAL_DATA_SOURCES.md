# Real Calibration Data

The local development subset is drawn from the MIT-licensed **POWDER Co-Channel Protocol Dataset**: six real OTA windows, each limited to the first 1,048,576 complex64 samples (8 MiB) from Round 1 / Gain 60. The subset totals 48 MiB and is Git-ignored.

Source: https://huggingface.co/datasets/T-Arshad/POWDER_CoChannel_Protocol_Dataset

The source comprises 768 OTA captures with mixtures of 802.11a Wi-Fi, 4G LTE, and 5G NR. Each local file has its source JSON metadata retained beside it, including sample rate (33.333 MS/s), centre frequency (2.425 GHz), gain, receiver, and active-transmitter state.

This is a calibration corpus, not ground truth for the first PSK/QAM demodulator. A high real-vs-synthetic discrimination score is expected initially because the real examples are protocol mixtures while the first recipes generate linear modulation. We will reduce that gap by adding one measured phenomenon per recipe stage and, later, protocol-matched OFDM/LoRa recipe families.
