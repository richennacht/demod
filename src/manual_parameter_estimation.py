"""Inspectable manual parameter estimators used by the DEmod comparison path."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from dual_parameter_estimator import dsp_estimate


def _transition_period_candidates(samples: list[complex], sample_rate_hz: float) -> list[dict[str, float]]:
    """Rank repeated-transition intervals; only a coarse symbol-rate proposal."""
    if len(samples) < 32:
        return []
    transitions = [abs(right - left) for left, right in zip(samples, samples[1:])]
    mean = sum(transitions) / len(transitions)
    centered = [value - mean for value in transitions]
    energy = sum(value * value for value in centered) or 1.0
    maximum_lag = min(256, len(centered) // 4)
    scored = []
    for lag in range(2, maximum_lag + 1):
        score = sum(centered[index] * centered[index - lag] for index in range(lag, len(centered))) / energy
        scored.append((score, lag))
    selected: list[dict[str, float]] = []
    for score, lag in sorted(scored, reverse=True):
        if all(abs(lag - item["samples_per_symbol_candidate"]) > 1 for item in selected):
            selected.append({"samples_per_symbol_candidate": float(lag), "symbol_rate_baud_candidate": round(sample_rate_hz / lag, 3), "normalized_transition_correlation": round(score, 5)})
        if len(selected) == 5:
            break
    return selected


def estimate(samples: list[complex], sample_rate_hz: float, spectral: dict[str, Any] | None = None) -> dict[str, Any]:
    if len(samples) < 8:
        raise ValueError("At least eight complex samples are required.")
    dc = sum(samples) / len(samples)
    centered = [sample - dc for sample in samples]
    powers = [abs(sample) ** 2 for sample in centered]
    floor = median(powers)
    peak = max(powers)
    result: dict[str, Any] = {
        "method_version": "manual_estimators_v1",
        "dc_iq": {"i": round(dc.real, 8), "q": round(dc.imag, 8), "method": "complex_sample_mean"},
        "coarse_carrier_offset_hz": {**dsp_estimate(samples, sample_rate_hz), "method": "fourth_power_phase_increment", "validity": "PSK-like signals; ambiguous for QAM/FSK/OFDM or low SNR"},
        "power": {"median_centered_power": round(floor, 8), "peak_centered_power": round(peak, 8), "peak_to_median_db": round(10 * math.log10(max(peak / max(floor, 1e-15), 1e-15)), 4)},
        "symbol_rate_candidates": _transition_period_candidates(centered, sample_rate_hz),
        "limitations": ["Symbol-rate candidates use transition periodicity only; they are not timing recovery.", "Absolute sample rate and centre frequency remain metadata/analyst inputs, not inferred facts."],
    }
    if spectral:
        result["spectrum"] = {"peak_frequency_hz": spectral["spectrum"]["peak_frequency_hz"], "occupied_bandwidth_99pct_hz": spectral["spectrum"]["occupied_bandwidth_99pct_hz"], "method": "Hann-window FFT 99%-energy occupancy"}
        result["burst_candidates"] = spectral["segmentation"]
    return result
