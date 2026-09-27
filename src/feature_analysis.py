#!/usr/bin/env python3
"""Deterministic, inspectable IQ features for DEmod's first DSP stage."""

from __future__ import annotations

import cmath
import math
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))]


def _fft_power(samples: list[complex], bins: int = 512) -> list[float]:
    """Small direct DFT: dependency-free and intended for report previews."""
    window = samples[: min(len(samples), bins)]
    size = len(window)
    if not size:
        return []
    tapered = [sample * (0.5 - 0.5 * math.cos(2 * math.pi * index / max(1, size - 1))) for index, sample in enumerate(window)]
    return [abs(sum(value * cmath.exp(-2j * math.pi * index * k / size) for index, value in enumerate(tapered))) ** 2 for k in range(size)]


def _wrapped_difference(right: float, left: float) -> float:
    return (right - left + math.pi) % (2 * math.pi) - math.pi


def _rank_modulation(amplitude_cv: float, phase_step_std: float, frequency_std_hz: float, occupied_fraction: float) -> list[dict[str, Any]]:
    """Explainable triage only; this is not a decoder or trained classifier."""
    raw = {
        "fsk": 0.55 * min(1.0, frequency_std_hz / 2_000.0) + 0.25 * (1 - min(1.0, amplitude_cv)) + 0.20 * min(1.0, phase_step_std / 1.2),
        "psk": 0.55 * (1 - min(1.0, amplitude_cv)) + 0.45 * (1 - min(1.0, frequency_std_hz / 2_000.0)),
        "qam": 0.65 * min(1.0, amplitude_cv / 0.55) + 0.35 * (1 - min(1.0, frequency_std_hz / 2_000.0)),
        "ofdm_or_multicarrier": min(1.0, 0.7 * occupied_fraction + 0.3 * min(1.0, amplitude_cv / 0.8)),
    }
    total = sum(raw.values()) or 1.0
    return [{"family": name, "score": round(value / total, 3), "basis": "amplitude, phase-step, instantaneous-frequency and occupied-spectrum statistics"} for name, value in sorted(raw.items(), key=lambda item: item[1], reverse=True)]


def analyse_iq(samples: list[complex], sample_rate_hz: float) -> dict[str, Any]:
    if len(samples) < 4:
        raise ValueError("At least four complex samples are required for IQ feature analysis.")
    dc = sum(samples) / len(samples)
    centered = [sample - dc for sample in samples]
    magnitudes = [abs(sample) for sample in centered]
    powers = [value * value for value in magnitudes]
    rms = math.sqrt(_mean(powers))
    amplitude_cv = math.sqrt(_mean([(value - _mean(magnitudes)) ** 2 for value in magnitudes])) / max(rms, 1e-12)
    phases = [math.atan2(sample.imag, sample.real) for sample in centered]
    phase_steps = [_wrapped_difference(right, left) for left, right in zip(phases, phases[1:])]
    inst_freq = [step * sample_rate_hz / (2 * math.pi) for step in phase_steps]
    freq_mean = _mean(inst_freq)
    freq_std = math.sqrt(_mean([(value - freq_mean) ** 2 for value in inst_freq]))
    spectrum = _fft_power(centered)
    spectrum_total = sum(spectrum) or 1.0
    peak = max(range(len(spectrum)), key=lambda index: spectrum[index]) if spectrum else 0
    signed_peak = peak if peak <= len(spectrum) // 2 else peak - len(spectrum)
    peak_hz = signed_peak * sample_rate_hz / max(1, len(spectrum))
    ordered = sorted(enumerate(spectrum), key=lambda item: item[1], reverse=True)
    selected: set[int] = set()
    energy = 0.0
    for index, value in ordered:
        selected.add(index)
        energy += value
        if energy >= 0.99 * spectrum_total:
            break
    occupied_bw = (max(selected) - min(selected) + 1) * sample_rate_hz / max(1, len(spectrum)) if selected else 0.0
    occupied_fraction = occupied_bw / max(sample_rate_hz, 1.0)
    return {
        "sample_count": len(samples),
        "sample_rate_hz": sample_rate_hz,
        "dc_offset": {"i": round(dc.real, 8), "q": round(dc.imag, 8)},
        "rms": round(rms, 8),
        "peak": round(max(magnitudes), 8),
        "crest_factor": round(max(magnitudes) / max(rms, 1e-12), 5),
        "amplitude_cv": round(amplitude_cv, 5),
        "instantaneous_frequency_hz": {"mean": round(freq_mean, 3), "std": round(freq_std, 3)},
        "spectrum": {"preview_bins": len(spectrum), "peak_frequency_hz": round(peak_hz, 3), "occupied_bandwidth_99pct_hz": round(occupied_bw, 3)},
        "modulation_hypotheses": _rank_modulation(amplitude_cv, math.sqrt(_mean([(value - _mean(phase_steps)) ** 2 for value in phase_steps])), freq_std, occupied_fraction),
        "limitations": ["Centre frequency cannot be recovered from baseband IQ without capture metadata or an external reference.", "A raw byte stream cannot establish an absolute sample rate; sample_rate_hz is a supplied hypothesis.", "Modulation hypotheses are DSP triage scores, not a demodulation or FEC result."],
    }
