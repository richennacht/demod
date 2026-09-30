"""Small transparent synthetic-trained modulation classifier with abstention."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from generate_synthetic import generate_batch


def features(samples: list[complex]) -> list[float]:
    dc = sum(samples) / len(samples)
    values = [sample - dc for sample in samples]
    magnitudes = [abs(value) for value in values]
    mean_mag = sum(magnitudes) / len(magnitudes)
    amplitude_cv = math.sqrt(sum((value - mean_mag) ** 2 for value in magnitudes) / len(magnitudes)) / max(mean_mag, 1e-12)
    phases = [math.atan2(value.imag, value.real) for value in values]
    steps = [((right - left + math.pi) % (2 * math.pi)) - math.pi for left, right in zip(phases, phases[1:])]
    phase_std = math.sqrt(sum((value - sum(steps) / len(steps)) ** 2 for value in steps) / max(1, len(steps))) if steps else 0.0
    fourth = sum(value ** 4 for value in values) / len(values)
    second = sum(value ** 2 for value in values) / len(values)
    return [amplitude_cv, phase_std, abs(second), abs(fourth), sum(abs(value) ** 2 for value in values) / len(values)]


class CentroidAMC:
    def __init__(self, centroids: dict[str, list[float]], scales: list[float], training_examples: int):
        self.centroids, self.scales, self.training_examples = centroids, scales, training_examples

    @classmethod
    def train(cls, recipes: list[dict[str, Any]], examples: int = 96, seed: int = 26147) -> "CentroidAMC":
        grouped: dict[str, list[list[float]]] = defaultdict(list)
        for samples, audit in generate_batch(recipes, examples, seed):
            grouped[str(audit["modulation"])].append(features(samples))
        all_rows = [row for rows in grouped.values() for row in rows]
        scales = [math.sqrt(sum((row[index] - sum(item[index] for item in all_rows) / len(all_rows)) ** 2 for row in all_rows) / len(all_rows)) or 1.0 for index in range(len(all_rows[0]))]
        centroids = {name: [sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0]))] for name, rows in grouped.items()}
        return cls(centroids, scales, examples)

    def predict(self, samples: list[complex]) -> dict[str, Any]:
        row = features(samples)
        distances = {name: math.sqrt(sum(((value - centroid[index]) / self.scales[index]) ** 2 for index, value in enumerate(row))) for name, centroid in self.centroids.items()}
        ranked = sorted(distances.items(), key=lambda item: item[1])
        best, runner_up = ranked[0], ranked[1] if len(ranked) > 1 else ("none", float("inf"))
        margin = (runner_up[1] - best[1]) / max(runner_up[1], 1e-12)
        confidence = max(0.0, min(1.0, margin))
        abstain = confidence < 0.12
        return {"predicted_modulation": None if abstain else best[0], "confidence": round(confidence, 4), "abstained": abstain, "reason": "nearest-centroid margin below 0.12" if abstain else "highest normalized feature-centroid similarity", "ranked_candidates": [{"modulation": name, "distance": round(distance, 5)} for name, distance in ranked], "model": {"type": "normalized_feature_centroid", "training_examples": self.training_examples, "classes": sorted(self.centroids), "scope": "Synthetic-recipe baseline; requires held-out real-data calibration before operational use."}}
