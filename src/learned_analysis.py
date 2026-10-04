"""Run DEmod's learned carrier-offset estimator and modulation classifier on a capture.

Both models are evaluated in research/README.md against recreated paper
baselines. They are trained only on simulated signals, so their outputs are
labelled as model estimates with confidence, never as ground truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from amc_models import CLASSES, default_model_amc
import cfo_estimators
import amc_models
from cfo_estimators import default_model

MAX_REGION = 65536
AMC_CHUNK = 1024
AMC_MAX_CHUNKS = 16


def analysis_region(samples: np.ndarray, segments: list[dict[str, Any]] | None) -> tuple[np.ndarray, dict[str, Any]]:
    """Prefer the longest energy segment so silence does not dilute the estimate."""
    if segments:
        seg = max(segments, key=lambda s: s["sample_count"])
        start, count = int(seg["sample_start"]), int(seg["sample_count"])
        if count >= 1024:
            count = min(count, MAX_REGION)
            return samples[start:start + count], {"source": "longest_energy_segment", "sample_start": start, "sample_count": count}
    count = min(len(samples), MAX_REGION)
    return samples[:count], {"source": "capture_start", "sample_start": 0, "sample_count": count}


def carrier_offset(region: np.ndarray, sample_rate_hz: float) -> dict[str, Any] | None:
    model = default_model()
    if model is None or len(region) < 256:
        return None
    r = model.estimate(region[None, :])
    cfo = float(r["cfo"][0])
    return {
        "carrier_offset_hz": cfo * sample_rate_hz,
        "normalised_cycles_per_sample": cfo,
        "confidence": round(float(r["confidence"][0]), 4),
        "refined_on_power": int(r["order"][0]) or None,
        "method": "SpecCFO: CNN over x**M spectra (M=1,2,4,8), then periodogram refinement on the strongest line" if bool(r["refined"][0]) else "SpecCFO network estimate (no spectral line strong enough to refine)",
        "model_file": cfo_estimators.MODEL_PATH.name,
        "range_hz": [-0.5 * sample_rate_hz, 0.5 * sample_rate_hz],
        "trained_range_hz": [-0.2 * sample_rate_hz, 0.2 * sample_rate_hz],
    }


def classify(region: np.ndarray, cfo_cycles: float | None) -> dict[str, Any] | None:
    model = default_model_amc()
    if model is None or len(region) < AMC_CHUNK:
        return None
    count = min(AMC_MAX_CHUNKS, len(region) // AMC_CHUNK)
    starts = np.linspace(0, len(region) - AMC_CHUNK, count).astype(int)
    chunks = np.stack([region[s:s + AMC_CHUNK] for s in starts])
    cfo = None
    if cfo_cycles is not None:
        cfo = np.full(count, cfo_cycles)
    probs = model.probabilities(chunks, cfo)
    logp = np.log(probs + 1e-12).mean(axis=0)
    p = np.exp(logp - logp.max())
    p = p / p.sum()
    order = np.argsort(p)[::-1]
    best = int(order[0])
    abstain = float(p[best]) < model.abstain_below
    calibrated = bool(model.meta.get("calibrated_on"))
    return {
        "predicted_modulation": None if abstain else CLASSES[best],
        "confidence": round(float(p[best]), 4),
        "abstained": abstain,
        "reason": f"top probability below the abstention threshold {model.abstain_below:.2f}" if abstain else ("highest calibrated probability" if calibrated else "highest probability (model not calibrated)"),
        "ranked_candidates": [{"modulation": CLASSES[i], "probability": round(float(p[i]), 4)} for i in order[:6]],
        "chunks_used": int(count),
        "model": {
            "type": "DemodAMC: CFO-compensated two-branch CNN (time and x**M spectra)",
            "classes": list(CLASSES),
            "model_file": amc_models.MODEL_PATH.name,
            "calibrated": calibrated,
            "temperature": model.temperature,
            "abstain_below": model.abstain_below,
            "scope": "Trained and evaluated on simulated signals only (research/README.md). Needs held-out real captures before operational use.",
        },
    }
