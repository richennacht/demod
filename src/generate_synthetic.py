#!/usr/bin/env python3
"""Recipe-driven, in-memory synthetic IQ generation for DEmod.

Recipes are the dataset. Generated IQ exists only in RAM for a training batch;
the model receives samples and task labels, never the recipe or provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def constellation(modulation: str) -> list[complex]:
    if modulation == "bpsk":
        return [complex(-1, 0), complex(1, 0)]
    if modulation == "qpsk":
        scale = 1 / math.sqrt(2)
        return [complex(i * scale, q * scale) for i in (-1, 1) for q in (-1, 1)]
    if modulation == "8psk":
        return [complex(math.cos(2 * math.pi * index / 8), math.sin(2 * math.pi * index / 8)) for index in range(8)]
    if modulation == "16qam":
        scale = 1 / math.sqrt(10)
        return [complex(i * scale, q * scale) for i in (-3, -1, 1, 3) for q in (-3, -1, 1, 3)]
    raise ValueError(f"Unsupported modulation: {modulation}")


def ofdm_waveform(symbols: int, rng: random.Random) -> list[complex]:
    """Generate an 802.11a-shaped, 20 MHz OFDM baseband waveform.

    This is deliberately a *shape* model, not a claim of standards-compliant
    Wi-Fi packets: 64-point IFFT, 52 occupied carriers and a 16-sample cyclic
    prefix reproduce the principal occupied-bandwidth and PAPR mechanisms.
    """
    fft_size, cp, active = 64, 16, [*range(-26, 0), *range(1, 27)]
    qpsk = constellation("qpsk")
    waveform: list[complex] = []
    for _ in range(symbols):
        bins = [0j] * fft_size
        for carrier in active:
            bins[carrier % fft_size] = qpsk[rng.randrange(len(qpsk))]
        time = [sum(bins[k] * complex(math.cos(2 * math.pi * k * n / fft_size), math.sin(2 * math.pi * k * n / fft_size)) for k in range(fft_size)) / math.sqrt(len(active)) for n in range(fft_size)]
        waveform.extend(time[-cp:] + time)
    return waveform


def resample_linear(signal: list[complex], ratio: float) -> list[complex]:
    """Deterministic fractional-rate resampler for recipe generation."""
    count = max(2, int(len(signal) * ratio))
    result: list[complex] = []
    for index in range(count):
        source = index / ratio
        left = min(int(source), len(signal) - 1)
        right = min(left + 1, len(signal) - 1)
        fraction = source - left
        result.append(signal[left] * (1 - fraction) + signal[right] * fraction)
    return result


def choose(value: Any, rng: random.Random) -> Any:
    """Sample a scalar, choice list, or inclusive [low, high] numeric range."""
    if isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
        return rng.uniform(value[0], value[1])
    if isinstance(value, list):
        return rng.choice(value)
    return value


def load_recipes(path: Path) -> list[dict[str, Any]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if loaded.get("recipe_version") != "0.1.0" or not loaded.get("recipes"):
        raise ValueError("Expected a non-empty DEmod recipe document version 0.1.0.")
    return loaded["recipes"]


def generate_example(recipe: dict[str, Any], seed: int) -> tuple[list[complex], dict[str, Any]]:
    """Generate one ephemeral IQ example and its hidden audit truth."""
    rng = random.Random(seed)
    modulation = choose(recipe["modulation"], rng)
    waveform = recipe.get("waveform", "linear")
    sample_rate = int(recipe["sample_rate_hz"])
    symbols, sps = int(recipe["symbols"]), int(recipe["samples_per_symbol"])
    settings = {key: choose(value, rng) for key, value in recipe.get("impairments", {}).items() if key not in ("multipath_taps", "dc_offset")}
    dc_distribution = recipe.get("impairments", {}).get("dc_offset", [[0.0, 0.0], [0.0, 0.0]])
    settings["dc_offset"] = [choose(dc_distribution[0], rng), choose(dc_distribution[1], rng)]
    points = constellation(modulation)
    if waveform == "ofdm_20mhz":
        # POWDER captures run at 33.333 MS/s while the 802.11a-like component
        # occupies a 20 MHz channel. 5/3 is the nominal rate conversion.
        signal = resample_linear(ofdm_waveform(symbols, rng), sample_rate / 20_000_000)
    else:
        signal = [points[rng.randrange(len(points))] for _ in range(symbols) for _ in range(sps)]

    taps = recipe.get("impairments", {}).get("multipath_taps", [[0, 1.0, 0.0]])
    channelled = [sum((signal[index - int(delay)] if index >= int(delay) else 0j) * complex(real, imag) for delay, real, imag in taps) for index in range(len(signal))]

    snr_db = float(settings.get("snr_db", 30.0))
    noise_sigma = math.sqrt(1 / (2 * 10 ** (snr_db / 10)))
    cfo_hz = float(settings.get("carrier_offset_hz", 0.0))
    dc_i, dc_q = settings.get("dc_offset", [0.0, 0.0])
    gain_db, phase_noise_std = float(settings.get("iq_gain_imbalance_db", 0.0)), float(settings.get("phase_noise_std_rad", 0.0))
    phase_error = math.radians(float(settings.get("iq_phase_imbalance_deg", 0.0)))
    impulse_probability, impulse_amplitude = float(settings.get("impulse_probability", 0.0)), float(settings.get("impulse_amplitude", 0.0))
    clip_level, gain, phase_noise = float(settings.get("clip_level", 1000.0)), 10 ** (gain_db / 20), 0.0
    colored_std, colored_rho = float(settings.get("colored_noise_std", 0.0)), float(settings.get("colored_noise_rho", 0.0))
    tone_amplitude, tone_hz = float(settings.get("tone_interferer_amplitude", 0.0)), float(settings.get("tone_interferer_hz", 0.0))
    burst_probability, burst_length, burst_amplitude = float(settings.get("burst_probability", 0.0)), int(settings.get("burst_length", 0)), float(settings.get("burst_amplitude", 0.0))
    cochannel_amplitude, cochannel_hz = float(settings.get("cochannel_interferer_amplitude", 0.0)), float(settings.get("cochannel_interferer_hz", 0.0))
    adc_bits = int(settings.get("adc_bits", 32))
    target_rms = float(settings.get("target_rms", 0.0))
    colored_i, colored_q, burst_remaining = 0.0, 0.0, 0
    cochannel_points = constellation("qpsk")
    emitters = int(settings.get("cochannel_emitters", 1))
    samples: list[complex] = []
    for index, sample in enumerate(channelled):
        phase_noise += rng.gauss(0, phase_noise_std)
        phase = 2 * math.pi * cfo_hz * index / sample_rate + phase_noise
        sample *= complex(math.cos(phase), math.sin(phase))
        i = sample.real * gain + float(dc_i) + rng.gauss(0, noise_sigma)
        q = sample.imag * math.cos(phase_error) + sample.real * math.sin(phase_error) + float(dc_q) + rng.gauss(0, noise_sigma)
        # Colored receiver noise: first-order autoregressive complex noise.
        colored_i = colored_rho * colored_i + math.sqrt(max(0.0, 1 - colored_rho**2)) * rng.gauss(0, colored_std)
        colored_q = colored_rho * colored_q + math.sqrt(max(0.0, 1 - colored_rho**2)) * rng.gauss(0, colored_std)
        i, q = i + colored_i, q + colored_q
        # Narrowband blocker / oscillator spur.
        tone_phase = 2 * math.pi * tone_hz * index / sample_rate
        i, q = i + tone_amplitude * math.cos(tone_phase), q + tone_amplitude * math.sin(tone_phase)
        # Bursty broadband interference; a short run is more realistic than isolated spikes alone.
        if burst_remaining == 0 and rng.random() < burst_probability:
            burst_remaining = burst_length
        if burst_remaining:
            i, q, burst_remaining = i + rng.gauss(0, burst_amplitude), q + rng.gauss(0, burst_amplitude), burst_remaining - 1
        # A second offset QPSK stream is a controlled co-channel interferer.
        for emitter in range(emitters):
            interferer = cochannel_points[rng.randrange(len(cochannel_points))]
            interference_phase = 2 * math.pi * (cochannel_hz * (emitter + 1)) * index / sample_rate
            interferer *= complex(math.cos(interference_phase), math.sin(interference_phase))
            i, q = i + cochannel_amplitude * interferer.real, q + cochannel_amplitude * interferer.imag
        if rng.random() < impulse_probability:
            i, q = i + rng.gauss(0, impulse_amplitude), q + rng.gauss(0, impulse_amplitude)
        i, q = max(-clip_level, min(clip_level, i)), max(-clip_level, min(clip_level, q))
        if adc_bits < 32:
            levels, step = 2**adc_bits - 1, 2 * clip_level / (2**adc_bits - 1)
            i, q = round((i + clip_level) / step) * step - clip_level, round((q + clip_level) / step) * step - clip_level
            i, q = max(-clip_level, min(clip_level, i)), max(-clip_level, min(clip_level, q))
        samples.append(complex(i, q))
    # Match the observed receiver scale after channel/front-end effects. This is
    # a recorded calibration parameter, not a label exposed to a downstream model.
    if target_rms > 0:
        observed_rms = math.sqrt(sum(abs(sample) ** 2 for sample in samples) / len(samples))
        if observed_rms > 0:
            samples = [sample * target_rms / observed_rms for sample in samples]
    audit_truth = {"recipe_id": recipe["recipe_id"], "seed": seed, "modulation": modulation, "symbol_rate_baud": sample_rate / sps, "impairments": settings}
    return samples, audit_truth


def generate_batch(recipes: list[dict[str, Any]], count: int, seed: int) -> list[tuple[list[complex], dict[str, Any]]]:
    if count < 1:
        raise ValueError("count must be at least 1")
    return [generate_example(recipes[index % len(recipes)], seed + index) for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ephemeral IQ batches from DEmod recipes.")
    parser.add_argument("--recipes", type=Path, default=Path("data/recipes/mvp-recipes.json"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=26147)
    args = parser.parse_args()
    batch = generate_batch(load_recipes(args.recipes), args.count, args.seed)
    print(json.dumps({"generated_in_memory": len(batch), "sample_counts": [len(samples) for samples, _ in batch], "recipe_ids": [truth["recipe_id"] for _, truth in batch]}, indent=2))


if __name__ == "__main__":
    main()
