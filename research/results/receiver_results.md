# Controlled receiver BER/SER

Controlled same-generator burst BER/SER, not real intercept or blind phase/mapping recovery.

Four captures per row, 1024 transmitted symbols, SPS 8, Fs 250 kHz, CFO +750 Hz, phase +0.17 radians, leading offset 3 samples. CFO and FSK tone levels are supplied. Manual mode supplies phase/timing; auto searches static phase/timing. The first/last 12 symbols are excluded by a fixed rule; no oracle alignment or rotation is performed. RRC uses rolloff 0.35. Independent random data per row, so manual/auto rows are not paired comparisons.

| Mode | Pulse | SNR dB | Acquisition | Bit errors / compared | BER | SER |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| bpsk | rect | 15 | manual | 0/4000 | 0.000000 | 0.000000 |
| bpsk | rect | 15 | static_auto | 0/4000 | 0.000000 | 0.000000 |
| bpsk | rect | 25 | manual | 0/4000 | 0.000000 | 0.000000 |
| bpsk | rect | 25 | static_auto | 0/4000 | 0.000000 | 0.000000 |
| bpsk | rrc | 15 | manual | 0/4000 | 0.000000 | 0.000000 |
| bpsk | rrc | 15 | static_auto | 0/4000 | 0.000000 | 0.000000 |
| bpsk | rrc | 25 | manual | 0/4000 | 0.000000 | 0.000000 |
| bpsk | rrc | 25 | static_auto | 0/4000 | 0.000000 | 0.000000 |
| qpsk | rect | 15 | manual | 0/8000 | 0.000000 | 0.000000 |
| qpsk | rect | 15 | static_auto | 0/8000 | 0.000000 | 0.000000 |
| qpsk | rect | 25 | manual | 0/8000 | 0.000000 | 0.000000 |
| qpsk | rect | 25 | static_auto | 0/8000 | 0.000000 | 0.000000 |
| qpsk | rrc | 15 | manual | 0/8000 | 0.000000 | 0.000000 |
| qpsk | rrc | 15 | static_auto | 0/8000 | 0.000000 | 0.000000 |
| qpsk | rrc | 25 | manual | 0/8000 | 0.000000 | 0.000000 |
| qpsk | rrc | 25 | static_auto | 0/8000 | 0.000000 | 0.000000 |
| 8psk | rect | 15 | manual | 0/12000 | 0.000000 | 0.000000 |
| 8psk | rect | 15 | static_auto | 0/12000 | 0.000000 | 0.000000 |
| 8psk | rect | 25 | manual | 0/12000 | 0.000000 | 0.000000 |
| 8psk | rect | 25 | static_auto | 0/12000 | 0.000000 | 0.000000 |
| 8psk | rrc | 15 | manual | 0/12000 | 0.000000 | 0.000000 |
| 8psk | rrc | 15 | static_auto | 0/12000 | 0.000000 | 0.000000 |
| 8psk | rrc | 25 | manual | 0/12000 | 0.000000 | 0.000000 |
| 8psk | rrc | 25 | static_auto | 0/12000 | 0.000000 | 0.000000 |
| 16qam | rect | 15 | manual | 0/16000 | 0.000000 | 0.000000 |
| 16qam | rect | 15 | static_auto | 0/16000 | 0.000000 | 0.000000 |
| 16qam | rect | 25 | manual | 0/16000 | 0.000000 | 0.000000 |
| 16qam | rect | 25 | static_auto | 0/16000 | 0.000000 | 0.000000 |
| 16qam | rrc | 15 | manual | 0/16000 | 0.000000 | 0.000000 |
| 16qam | rrc | 15 | static_auto | 0/16000 | 0.000000 | 0.000000 |
| 16qam | rrc | 25 | manual | 0/16000 | 0.000000 | 0.000000 |
| 16qam | rrc | 25 | static_auto | 0/16000 | 0.000000 | 0.000000 |
| 64qam | rect | 15 | manual | 9/24000 | 0.000375 | 0.002250 |
| 64qam | rect | 15 | static_auto | 21/24000 | 0.000875 | 0.005250 |
| 64qam | rect | 25 | manual | 0/24000 | 0.000000 | 0.000000 |
| 64qam | rect | 25 | static_auto | 0/24000 | 0.000000 | 0.000000 |
| 64qam | rrc | 15 | manual | 7/24000 | 0.000292 | 0.001750 |
| 64qam | rrc | 15 | static_auto | 17/24000 | 0.000708 | 0.004250 |
| 64qam | rrc | 25 | manual | 0/24000 | 0.000000 | 0.000000 |
| 64qam | rrc | 25 | static_auto | 0/24000 | 0.000000 | 0.000000 |
| 2fsk | rect | 15 | manual | 0/4000 | 0.000000 | 0.000000 |
| 2fsk | rect | 15 | static_auto | 0/3999 | 0.000000 | 0.000000 |
| 2fsk | rect | 25 | manual | 0/4000 | 0.000000 | 0.000000 |
| 2fsk | rect | 25 | static_auto | 0/3999 | 0.000000 | 0.000000 |
| 4fsk | rect | 15 | manual | 0/8000 | 0.000000 | 0.000000 |
| 4fsk | rect | 15 | static_auto | 0/7996 | 0.000000 | 0.000000 |
| 4fsk | rect | 25 | manual | 0/8000 | 0.000000 | 0.000000 |
| 4fsk | rect | 25 | static_auto | 0/7994 | 0.000000 | 0.000000 |

Zero measured errors is not a zero-error guarantee. Static phase search is M-fold ambiguous; these phases lie inside the selected fundamental sector. Tests do not establish arbitrary-phase recovery, clock-drift/fractional timing recovery, multipath equalisation, unknown mapping or protocol/FEC decoding. GNU Radio execution is not validated here. Recipes and per-capture counts/hashes are in receiver_results.json.
