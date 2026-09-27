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


def calibration_report(real_paths: list[Path], recipes_path: Path, seed: int = 26147, window: int = 4096) -> dict[str, float | int]:
    real = [read_complex64_window(path, 0, window) for path in real_paths]
    synthetic = [samples for samples, _ in generate_batch(load_recipes(recipes_path), len(real), seed)]
    rows, labels = [features(item) for item in real] + [features(item[:window]) for item in synthetic], [1] * len(real) + [0] * len(synthetic)
    # Deterministic stratified split: even-index examples train, odd-index examples evaluate.
    train_indices = [index for index in range(len(rows)) if index % 2 == 0]
    test_indices = [index for index in range(len(rows)) if index % 2 == 1]
    train_rows, means, scales = normalize([rows[index] for index in train_indices])
    test_rows = [[(value - means[column]) / scales[column] for column, value in enumerate(rows[index])] for index in test_indices]
    model = train_logistic(train_rows, [labels[index] for index in train_indices])
    return {"real_windows": len(real), "synthetic_windows": len(synthetic), "held_out_accuracy": round(accuracy(test_rows, [labels[index] for index in test_indices], model), 4), "interpretation": "High accuracy indicates an uncalibrated synthetic-to-real domain gap; do not deploy this classifier as a decoder."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the current real-vs-synthetic gap.")
    parser.add_argument("--real-dir", type=Path, default=Path("data/real/powder-pcp-mini"))
    parser.add_argument("--recipes", type=Path, default=Path("data/recipes/mvp-recipes.json"))
    args = parser.parse_args()
    paths = sorted(args.real_dir.glob("*.bin"))
    if len(paths) < 2:
        raise ValueError("At least two real complex64 files are required.")
    print(calibration_report(paths, args.recipes))


if __name__ == "__main__":
    main()
