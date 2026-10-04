"""Vectorised synthetic waveform simulator for DEmod model research.

Generates complex-baseband examples with exact labels for modulation, carrier
frequency offset (CFO), SNR, samples per symbol and channel. Everything is in
normalised units: CFO is in cycles per sample (multiply by the sample rate to
get Hz) and SNR is total signal power over total noise power in the sampled
band. Nothing is written to disk.

The waveform models follow the settings used by the recreated papers where
they state them: root-raised-cosine pulses with a 6-symbol span and roll-off
0.2 to 0.7 (Chen et al. 2023), random carrier phase, and exponential-profile
Rayleigh multipath described by a mean delay spread in samples (O'Shea et al.
2017). These are simulations, not captures, and results on them are not
evidence of field performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MODULATIONS = ("bpsk", "qpsk", "8psk", "16qam", "64qam", "2fsk", "4fsk", "gmsk", "ofdm", "am", "fm", "noise")
LINEAR = ("bpsk", "qpsk", "8psk", "16qam", "64qam")


def _constellation(name: str) -> np.ndarray:
    if name == "bpsk":
        points = np.array([-1, 1], dtype=complex)
    elif name == "qpsk":
        points = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    elif name == "8psk":
        points = np.exp(1j * np.pi / 4 * np.arange(8))
    elif name in ("16qam", "64qam"):
        side = 4 if name == "16qam" else 8
        levels = np.arange(-(side - 1), side, 2)
        points = (levels[:, None] + 1j * levels[None, :]).ravel()
    else:
        raise ValueError(name)
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


CONSTELLATIONS = {name: _constellation(name) for name in LINEAR}


def rrc_taps(sps: int, rolloff: float, span: int = 6) -> np.ndarray:
    """Root-raised-cosine impulse response, unit energy, span in symbols."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    taps = np.zeros_like(t)
    b = rolloff
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1 - b + 4 * b / np.pi
        elif b > 0 and abs(abs(4 * b * ti) - 1) < 1e-9:
            taps[i] = b / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * b)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * b)))
        else:
            taps[i] = (np.sin(np.pi * ti * (1 - b)) + 4 * b * ti * np.cos(np.pi * ti * (1 + b))) / (np.pi * ti * (1 - (4 * b * ti) ** 2))
    return taps / np.sqrt(np.sum(taps**2))


PULSES = ("rrc", "rect", "rc", "halfsine", "gauss")


def pulse_taps(kind: str, sps: int, rolloff: float, span: int = 6) -> np.ndarray:
    """Unit-energy transmit pulse. 'rrc' is the paper setting, the rest are for robustness training."""
    if kind == "rrc":
        return rrc_taps(sps, rolloff, span)
    if kind == "rect":
        taps = np.ones(sps)
    elif kind == "halfsine":
        taps = np.sin(np.pi * (np.arange(sps) + 0.5) / sps)
    elif kind == "rc":
        t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
        den = 1 - (2 * rolloff * t) ** 2
        taps = np.where(np.abs(den) < 1e-9, np.pi / 4 * np.sinc(1 / (2 * max(rolloff, 1e-3))), np.sinc(t) * np.cos(np.pi * rolloff * t) / np.where(np.abs(den) < 1e-9, 1.0, den))
    elif kind == "gauss":
        t = np.arange(-2 * sps, 2 * sps + 1) / sps
        taps = np.exp(-t**2 / (2 * (np.sqrt(np.log(2)) / (2 * np.pi * 0.5)) ** 2))
    else:
        raise ValueError(kind)
    return taps / np.sqrt(np.sum(taps**2))


def _lowpass_noise(rng: np.random.Generator, n: int, bandwidth: float) -> np.ndarray:
    """Real Gaussian message band-limited to `bandwidth` cycles/sample, peak-normalised."""
    spectrum = np.fft.rfft(rng.standard_normal(n + 64))
    freqs = np.fft.rfftfreq(n + 64)
    spectrum[freqs > bandwidth] = 0
    message = np.fft.irfft(spectrum, n + 64)[32:32 + n]
    return message / (np.max(np.abs(message)) + 1e-12)


def _linear(rng, name, length, sps, rolloff, pulse="rrc"):
    span = 6
    symbols_needed = length // sps + span + 4
    symbols = CONSTELLATIONS[name][rng.integers(0, len(CONSTELLATIONS[name]), symbols_needed)]
    up = np.zeros(symbols_needed * sps, dtype=complex)
    up[::sps] = symbols
    shaped = np.convolve(up, pulse_taps(pulse, sps, rolloff, span))
    start = span * sps + int(rng.integers(0, sps))
    return shaped[start:start + length]


def _cpfsk(rng, length, sps, levels, h, gaussian_bt=None):
    symbols_needed = length // sps + 8
    symbols = rng.choice(levels, symbols_needed).astype(float)
    nrz = np.repeat(symbols, sps)
    if gaussian_bt is not None:
        t = np.arange(-2 * sps, 2 * sps + 1) / sps
        sigma = np.sqrt(np.log(2)) / (2 * np.pi * gaussian_bt)
        g = np.exp(-t**2 / (2 * sigma**2))
        nrz = np.convolve(nrz, g / g.sum(), mode="same")
    freq = h / (2 * sps) * nrz  # cycles/sample, symbol phase step pi*h*symbol
    phase = 2 * np.pi * np.cumsum(freq)
    start = 4 * sps + int(rng.integers(0, sps))
    return np.exp(1j * phase[start:start + length])


def _ofdm(rng, length):
    nfft = 64
    oversample = int(rng.choice([1, 2, 4]))
    used = int(rng.integers(40, 57))
    cp = nfft * oversample // 4
    carriers = np.r_[np.arange(1, used // 2 + 1), np.arange(-(used - used // 2), 0)]
    qam = CONSTELLATIONS["16qam"] if rng.random() < 0.5 else CONSTELLATIONS["qpsk"]
    out = []
    total = 0
    while total < length + nfft * oversample:
        bins = np.zeros(nfft * oversample, dtype=complex)
        bins[carriers % (nfft * oversample)] = qam[rng.integers(0, len(qam), used)]
        symbol = np.fft.ifft(bins)
        out.append(np.r_[symbol[-cp:], symbol])
        total += len(out[-1])
    wave = np.concatenate(out)
    start = int(rng.integers(0, nfft * oversample))
    return wave[start:start + length]


@dataclass
class SimConfig:
    length: int = 1024
    modulations: tuple[str, ...] = MODULATIONS
    snr_db: tuple[float, float] = (-10.0, 30.0)
    cfo: tuple[float, float] = (-0.2, 0.2)
    sps: tuple[int, int] = (2, 16)
    rolloff: tuple[float, float] = (0.2, 0.7)
    rayleigh_probability: float = 0.0
    delay_spread: tuple[float, float] = (0.5, 2.0)
    hardware_probability: float = 0.0
    # Robustness augmentation. All default to the paper setting and consume no random numbers when off,
    # so existing seeded test sets are reproduced exactly.
    pulses: tuple[str, ...] = ("rrc",)
    phase_noise_probability: float = 0.0
    phase_noise_std: tuple[float, float] = (0.001, 0.02)
    interferer_probability: float = 0.0
    interferer_amplitude: tuple[float, float] = (0.05, 0.4)
    impulse_probability: float = 0.0
    dc_probability: float = 0.0
    fixed: dict = field(default_factory=dict)


def simulate(rng: np.random.Generator, count: int, cfg: SimConfig) -> dict[str, np.ndarray]:
    """Return a batch: x (count, length) complex64 plus per-example labels."""
    x = np.empty((count, cfg.length), dtype=np.complex64)
    labels = {k: np.empty(count) for k in ("cfo", "snr_db", "sps", "rolloff", "delay_spread")}
    mod_index = np.empty(count, dtype=np.int32)
    for i in range(count):
        name = cfg.fixed.get("modulation") or cfg.modulations[int(rng.integers(0, len(cfg.modulations)))]
        sps = int(cfg.fixed.get("sps", rng.integers(cfg.sps[0], cfg.sps[1] + 1)))
        rolloff = float(cfg.fixed.get("rolloff", rng.uniform(*cfg.rolloff)))
        snr = float(cfg.fixed.get("snr_db", rng.uniform(*cfg.snr_db)))
        cfo = float(cfg.fixed.get("cfo", rng.uniform(*cfg.cfo)))
        n = cfg.length
        if sps < 2 and name not in LINEAR:
            sps = 2  # symbol-spaced (1 sample/symbol) only makes sense for linear modulations
        if name in LINEAR:
            pulse = cfg.pulses[int(rng.integers(0, len(cfg.pulses)))] if len(cfg.pulses) > 1 else cfg.pulses[0]
            if sps == 1:
                pulse = "rect"  # one sample per symbol: no pulse shaping is possible (this is what the repo's recipe generator emits)
            s = _linear(rng, name, n, sps, rolloff, pulse)
        elif name == "2fsk":
            s = _cpfsk(rng, n, sps, np.array([-1, 1]), rng.uniform(0.5, 1.0))
        elif name == "4fsk":
            s = _cpfsk(rng, n, sps, np.array([-3, -1, 1, 3]), rng.uniform(0.25, 0.5))
        elif name == "gmsk":
            s = _cpfsk(rng, n, sps, np.array([-1, 1]), 0.5, gaussian_bt=0.3)
        elif name == "ofdm":
            s = _ofdm(rng, n)
        elif name == "am":
            s = (1 + rng.uniform(0.3, 0.9) * _lowpass_noise(rng, n, rng.uniform(0.005, 0.05))).astype(complex)
        elif name == "fm":
            s = np.exp(2j * np.pi * np.cumsum(rng.uniform(0.01, 0.08) * _lowpass_noise(rng, n, rng.uniform(0.005, 0.03))))
        else:
            s = np.zeros(n, dtype=complex)
        spread = 0.0
        if name != "noise" and rng.random() < cfg.rayleigh_probability:
            spread = float(cfg.fixed.get("delay_spread", rng.uniform(*cfg.delay_spread)))
            taps_n = max(2, int(np.ceil(spread * 5)) + 1)
            profile = np.exp(-np.arange(taps_n) / spread)
            taps = (rng.standard_normal(taps_n) + 1j * rng.standard_normal(taps_n)) * np.sqrt(profile / profile.sum() / 2)
            s = np.convolve(s, taps)[:n]
        power = np.mean(np.abs(s) ** 2)
        if power > 0:
            s = s / np.sqrt(power)
        if name != "noise" and cfg.phase_noise_probability > 0 and rng.random() < cfg.phase_noise_probability:
            s = s * np.exp(1j * np.cumsum(rng.uniform(*cfg.phase_noise_std) * rng.standard_normal(n)))
        s = s * np.exp(1j * (2 * np.pi * cfo * np.arange(n) + rng.uniform(0, 2 * np.pi)))
        if name != "noise" and cfg.interferer_probability > 0 and rng.random() < cfg.interferer_probability:
            # A strong tone elsewhere in the band. The label stays the carrier offset of the wanted signal.
            s = s + rng.uniform(*cfg.interferer_amplitude) * np.exp(1j * (2 * np.pi * rng.uniform(-0.5, 0.5) * np.arange(n) + rng.uniform(0, 2 * np.pi)))
        noise_var = 10 ** (-snr / 10) if name != "noise" else 1.0
        y = s + np.sqrt(noise_var / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        if cfg.impulse_probability > 0 and rng.random() < cfg.impulse_probability:
            hits = rng.integers(0, n, int(rng.integers(2, 12)))
            y[hits] += rng.uniform(3, 10) * np.sqrt(np.mean(np.abs(y) ** 2)) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(hits)))
        if cfg.dc_probability > 0 and rng.random() < cfg.dc_probability:
            y = y + (rng.uniform(-0.1, 0.1) + 1j * rng.uniform(-0.1, 0.1)) * np.sqrt(np.mean(np.abs(y) ** 2))
        if rng.random() < cfg.hardware_probability:
            g = 10 ** (rng.uniform(-1, 1) / 20)
            phi = np.deg2rad(rng.uniform(-5, 5))
            y = y.real + 1j * g * (np.sin(phi) * y.real + np.cos(phi) * y.imag)
            y = y + (rng.uniform(-0.05, 0.05) + 1j * rng.uniform(-0.05, 0.05)) * np.sqrt(np.mean(np.abs(y) ** 2))
        x[i] = y
        mod_index[i] = MODULATIONS.index(name)
        labels["cfo"][i], labels["snr_db"][i], labels["sps"][i], labels["rolloff"][i], labels["delay_spread"][i] = cfo, snr, sps, rolloff, spread
    return {"x": x, "modulation": mod_index, **labels}
