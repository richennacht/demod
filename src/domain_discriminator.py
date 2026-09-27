#!/usr/bin/env python3
"""Small dependency-free real-vs-synthetic calibration diagnostic.

It is deliberately not part of the final signal decoder. High held-out domain
accuracy means the synthetic recipe family is missing real-world effects.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path

from generate_synthetic import generate_batch, load_recipes
from powder_stream import sampled_windows


FEATURE_NAMES = ("mean_i", "mean_q", "rms", "variance_i", "variance_q", "iq_covariance", "mean_phase_step", "kurtosis", "band_0", "band_1", "band_2", "band_3", "band_4", "band_5", "band_6", "band_7")


def read_complex64_window(path: Path, offset_samples: int, length: int) -> list[complex]:
    with path.open("rb") as source:
        source.seek(offset_samples * 8)
        raw = source.read(length * 8)
    values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    return [complex(values[index], values[index + 1]) for index in range(0, len(values), 2)]


def features(samples: list[complex]) -> list[float]:
    if len(samples) < 4:
        raise ValueError("Need at least four complex samples.")
    n = len(samples)
    mean_i, mean_q = sum(item.real for item in samples) / n, sum(item.imag for item in samples) / n
    powers = [abs(item) ** 2 for item in samples]
    mean_power = sum(powers) / n
    variance_i = sum((item.real - mean_i) ** 2 for item in samples) / n
    variance_q = sum((item.imag - mean_q) ** 2 for item in samples) / n
    correlation_iq = sum((item.real - mean_i) * (item.imag - mean_q) for item in samples) / n
    phase_steps = [samples[index] * samples[index - 1].conjugate() for index in range(1, n)]
    mean_phase_step = sum(math.atan2(step.imag, step.real) for step in phase_steps) / len(phase_steps)
    kurtosis = sum(power * power for power in powers) / (n * max(mean_power * mean_power, 1e-12))
    # Eight low-cost coarse spectral-band powers; enough for a diagnostic, not AMC.
    probe = samples[: min(256, n)]
    bands = []
    for bin_index in range(8):
        frequency = 2 * math.pi * bin_index / 8
        coefficient = sum(value * complex(math.cos(-frequency * index), math.sin(-frequency * index)) for index, value in enumerate(probe))
        bands.append(abs(coefficient) ** 2 / len(probe))
    return [mean_i, mean_q, math.sqrt(mean_power), variance_i, variance_q, correlation_iq, mean_phase_step, kurtosis, *bands]


def normalize(rows: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    width = len(rows[0])
    means = [sum(row[column] for row in rows) / len(rows) for column in range(width)]
    scales = [math.sqrt(sum((row[column] - means[column]) ** 2 for row in rows) / len(rows)) or 1.0 for column in range(width)]
    return [[(value - means[index]) / scales[index] for index, value in enumerate(row)] for row in rows], means, scales


def train_logistic(rows: list[list[float]], labels: list[int], epochs: int = 300, learning_rate: float = 0.08) -> list[float]:
    weights = [0.0] * (len(rows[0]) + 1)
    for _ in range(epochs):
        for row, label in zip(rows, labels):
            score = weights[0] + sum(weight * value for weight, value in zip(weights[1:], row))
            probability = 1 / (1 + math.exp(-max(-30, min(30, score))))
            error = label - probability
            weights[0] += learning_rate * error
            for index, value in enumerate(row, start=1):
                weights[index] += learning_rate * error * value
    return weights


def accuracy(rows: list[list[float]], labels: list[int], weights: list[float]) -> float:
    predicted = [int(weights[0] + sum(weight * value for weight, value in zip(weights[1:], row)) >= 0) for row in rows]
    return sum(left == right for left, right in zip(predicted, labels)) / len(labels)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided 95% Wilson interval for a binomial held-out accuracy."""
    if total < 1:
        raise ValueError("total must be positive")
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def single_feature_attribution(rows: list[list[float]], labels: list[int], train_indices: list[int], test_indices: list[int]) -> dict[str, float]:
    """Held-out one-feature accuracies: a shortcut diagnostic, not causality."""
    scores: dict[str, float] = {}
    for column, name in enumerate(FEATURE_NAMES):
        train, means, scales = normalize([[rows[index][column]] for index in train_indices])
        test = [[(rows[index][column] - means[0]) / scales[0]] for index in test_indices]
        scores[name] = round(accuracy(test, [labels[index] for index in test_indices], train_logistic(train, [labels[index] for index in train_indices])), 4)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def calibration_report_samples(real: list[list[complex]], recipes_path: Path, seed: int = 26147, window: int = 4096, recipe_id: str | None = None) -> dict[str, float | int | list[float] | bool]:
    recipes = load_recipes(recipes_path)
    if recipe_id:
        recipes = [recipe for recipe in recipes if recipe["recipe_id"] == recipe_id]
        if not recipes:
            raise ValueError(f"Unknown recipe_id: {recipe_id}")
    synthetic = [samples for samples, _ in generate_batch(recipes, len(real), seed)]
    rows, labels = [features(item) for item in real] + [features(item[:window]) for item in synthetic], [1] * len(real) + [0] * len(synthetic)
    # Deterministic stratified split: even-index examples train, odd-index examples evaluate.
    train_indices = [index for index in range(len(rows)) if index % 2 == 0]
    test_indices = [index for index in range(len(rows)) if index % 2 == 1]
    train_rows, means, scales = normalize([rows[index] for index in train_indices])
    test_rows = [[(value - means[column]) / scales[column] for column, value in enumerate(rows[index])] for index in test_indices]
    model = train_logistic(train_rows, [labels[index] for index in train_indices])
    score = accuracy(test_rows, [labels[index] for index in test_indices], model)
    interval = wilson_interval(round(score * len(test_indices)), len(test_indices))
    half_width = (interval[1] - interval[0]) / 2
    return {"real_windows": len(real), "synthetic_windows": len(synthetic), "held_out_examples": len(test_indices), "held_out_accuracy": round(score, 4), "accuracy_ci95": [round(interval[0], 4), round(interval[1], 4)], "ci95_half_width": round(half_width, 4), "passes_precision_gate": abs(score - 0.5) <= 0.05 and half_width <= 0.05, "single_feature_accuracy": single_feature_attribution(rows, labels, train_indices, test_indices), "interpretation": "A score near 0.5 is only evidence of a small gap when its held-out confidence interval is narrow and groups were never split across train and test. Single-feature accuracies identify shortcuts, not causes."}


def calibration_report(real_paths: list[Path], recipes_path: Path, seed: int = 26147, window: int = 4096, recipe_id: str | None = None) -> dict[str, float | int | list[float] | bool]:
    return calibration_report_samples([read_complex64_window(path, 0, window) for path in real_paths], recipes_path, seed, window, recipe_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the current real-vs-synthetic gap.")
    parser.add_argument("--real-dir", type=Path, default=Path("data/real/powder-pcp-mini"))
    parser.add_argument("--recipes", type=Path, default=Path("data/recipes/mvp-recipes.json"))
    parser.add_argument("--recipe-id", default="stage-2-powder-ofdm", help="Generate candidates from one protocol-matched recipe family.")
    parser.add_argument("--stream-powder", type=int, metavar="COUNT", help="Stream COUNT provenance-distinct public POWDER windows into RAM for one evaluation; never writes them to disk.")
    parser.add_argument("--seed", type=int, default=26147)
    args = parser.parse_args()
    if args.stream_powder:
        print(calibration_report_samples(sampled_windows(args.stream_powder, seed=args.seed), args.recipes, seed=args.seed, recipe_id=args.recipe_id))
    else:
        paths = sorted(args.real_dir.glob("*.bin"))
        if len(paths) < 2:
            raise ValueError("At least two real complex64 files are required.")
        print(calibration_report(paths, args.recipes, recipe_id=args.recipe_id))


if __name__ == "__main__":
    main()
