#!/usr/bin/env python3
"""Reversible, opt-in IQ denoising branches.

Nothing imports or executes this module in the analysis path.  Its default
configuration returns an exact copy of input samples and audit findings only.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any


DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "dc_offset": {"enabled": False, "method": "component_median"},
    "impulse_blanking": {"enabled": False, "mad_threshold": 8.0, "replacement": "linear_interpolation"},
    "iq_imbalance": {"enabled": False, "status": "planned_requires_validation"},
    "rfi_mask": {"enabled": False, "status": "planned_detection_only"},
}


def _robust_location(samples: list[complex]) -> complex:
    return complex(median([sample.real for sample in samples]), median([sample.imag for sample in samples]))


def inspect(samples: list[complex]) -> dict[str, Any]:
    """Return candidate artifact measurements without changing a sample."""
    if not samples:
        raise ValueError("Cannot inspect an empty IQ sequence.")
    location = _robust_location(samples)
    radii = [abs(sample - location) for sample in samples]
    radius_median = median(radii)
    mad = median([abs(radius - radius_median) for radius in radii])
    scale = max(1.4826 * mad, 1e-12)
    impulse_count = sum(radius > radius_median + 8.0 * scale for radius in radii)
    return {
        "dc_offset_candidate": {"i": location.real, "q": location.imag, "method": "component_median"},
        "impulse_candidate": {"median_radius": radius_median, "mad_scale": scale, "count_at_8_mad": impulse_count, "fraction_at_8_mad": impulse_count / len(samples)},
        "notes": ["Measurements are candidates, not proof that a component is noise.", "No samples were modified during inspection."],
    }


def _blank_impulses(samples: list[complex], threshold: float) -> tuple[list[complex], list[int], dict[str, float]]:
    location = _robust_location(samples)
    radii = [abs(sample - location) for sample in samples]
    # Compute the median once: recomputing it per element made this O(n^2 log n).
    center = median(radii)
    scale = max(1.4826 * median([abs(radius - center) for radius in radii]), 1e-12)
    masked = [index for index, radius in enumerate(radii) if radius > center + threshold * scale]
    result = list(samples)
    for index in masked:
        left = result[index - 1] if index else result[min(1, len(result) - 1)]
        right = result[index + 1] if index + 1 < len(result) else result[max(0, len(result) - 2)]
        result[index] = (left + right) / 2
    return result, masked, {"median_radius": center, "mad_scale": scale}


def apply_pipeline(samples: list[complex], config: dict[str, dict[str, Any]] | None = None) -> tuple[list[complex], dict[str, Any]]:
    """Apply only explicitly enabled reversible branches and return their audit."""
    if not samples:
        raise ValueError("Cannot denoise an empty IQ sequence.")
    options = {name: dict(values) for name, values in DEFAULT_CONFIG.items()}
    for name, values in (config or {}).items():
        if name not in options:
            raise ValueError(f"Unknown denoising stage: {name}")
        options[name].update(values)
    raw_measurements = inspect(samples)
    derived = list(samples)
    actions: list[dict[str, Any]] = []
    if options["dc_offset"]["enabled"]:
        offset = _robust_location(derived)
        derived = [sample - offset for sample in derived]
        actions.append({"stage": "dc_offset", "applied": True, "removed": {"i": offset.real, "q": offset.imag}})
    else:
        actions.append({"stage": "dc_offset", "applied": False, "reason": "disabled_by_default"})
    if options["impulse_blanking"]["enabled"]:
        derived, mask, statistics = _blank_impulses(derived, float(options["impulse_blanking"]["mad_threshold"]))
        actions.append({"stage": "impulse_blanking", "applied": True, "masked_indices": mask, "masked_count": len(mask), **statistics})
    else:
        actions.append({"stage": "impulse_blanking", "applied": False, "reason": "disabled_by_default"})
    for name in ("iq_imbalance", "rfi_mask"):
        actions.append({"stage": name, "applied": False, "reason": options[name]["status"] if options[name]["enabled"] else "disabled_by_default"})
    return derived, {"raw_measurements": raw_measurements, "config": options, "actions": actions, "raw_preserved": True}
