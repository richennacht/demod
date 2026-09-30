"""Connected provenance-first IQ analysis pipeline for the local DEmod API."""

from __future__ import annotations

from typing import Any

from denoise_iq import apply_pipeline
from feature_analysis import analyse_iq
from manual_parameter_estimation import estimate as manual_estimate

try:
    from spectral_analysis import analyse_spectrum
except ModuleNotFoundError:
    analyse_spectrum = None


def analyse(samples: list[complex], sample_rate_hz: float, denoise_config: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    derived, denoise_audit = apply_pipeline(samples, denoise_config)
    raw_spectrum = analyse_spectrum(samples, sample_rate_hz) if analyse_spectrum else None
    derived_spectrum = analyse_spectrum(derived, sample_rate_hz) if analyse_spectrum else None
    return {
        "raw_branch": {"features": analyse_iq(samples, sample_rate_hz), "manual_parameters": manual_estimate(samples, sample_rate_hz, raw_spectrum), "visualization": raw_spectrum},
        "derived_branch": {"features": analyse_iq(derived, sample_rate_hz), "manual_parameters": manual_estimate(derived, sample_rate_hz, derived_spectrum), "visualization": derived_spectrum},
        "denoising": denoise_audit,
        "comparison_rule": "Raw branch is always retained. Derived results are an opt-in aid and never replace raw evidence.",
    }
