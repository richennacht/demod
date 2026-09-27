#!/usr/bin/env python3
"""DSP-versus-learned parameter estimation with explicit disagreement.

This is a transparent MVP, not a replacement for carrier/timing recovery. The
learned branch is a small pure-Python MLP trained only from ephemeral recipes.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from generate_synthetic import generate_batch, load_recipes


PARAMETERS = ("dc_i", "dc_q", "carrier_offset_hz")


def dsp_features(samples: list[complex], sample_rate_hz: float) -> list[float]:
    """Fixed features shared by the DSP and learned branches."""
    count = len(samples)
    if count < 8:
        raise ValueError("At least eight complex samples are required.")
    mean_i, mean_q = sum(value.real for value in samples) / count, sum(value.imag for value in samples) / count
    variance_i = sum((value.real - mean_i) ** 2 for value in samples) / count
    variance_q = sum((value.imag - mean_q) ** 2 for value in samples) / count
    covariance = sum((value.real - mean_i) * (value.imag - mean_q) for value in samples) / count
    # Fourth power reduces modulation phase for BPSK/QPSK and gives a coarse,
    # intentionally fallible CFO estimate for the comparison branch.
    fourth = [value**4 for value in samples]
    phase_sum = sum((fourth[index] * fourth[index - 1].conjugate()) for index in range(1, count))
    cfo_hz = math.atan2(phase_sum.imag, phase_sum.real) * sample_rate_hz / (8 * math.pi)
    return [mean_i, mean_q, math.sqrt(variance_i + variance_q), variance_i, variance_q, covariance, cfo_hz]


def dsp_estimate(samples: list[complex], sample_rate_hz: float) -> dict[str, float]:
    values = dsp_features(samples, sample_rate_hz)
    return {"dc_i": values[0], "dc_q": values[1], "carrier_offset_hz": values[-1]}


def truth_from_audit(audit: dict[str, Any]) -> list[float]:
    dc_i, dc_q = audit["impairments"]["dc_offset"]
    return [float(dc_i), float(dc_q), float(audit["impairments"]["carrier_offset_hz"])]


def standardize(rows: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    means = [sum(row[column] for row in rows) / len(rows) for column in range(len(rows[0]))]
    scales = [math.sqrt(sum((row[column] - means[column]) ** 2 for row in rows) / len(rows)) or 1.0 for column in range(len(rows[0]))]
    return [[(value - means[index]) / scales[index] for index, value in enumerate(row)] for row in rows], means, scales


class TinyMLP:
    """A 7→12→3 tanh regressor; small enough to audit and run without packages."""

    def __init__(self, input_size: int, output_size: int, seed: int = 26147):
        rng = random.Random(seed)
        self.hidden = [[rng.uniform(-0.2, 0.2) for _ in range(input_size + 1)] for _ in range(12)]
        self.output = [[rng.uniform(-0.2, 0.2) for _ in range(13)] for _ in range(output_size)]

    def forward(self, row: list[float]) -> tuple[list[float], list[float]]:
        hidden = [math.tanh(weights[0] + sum(weight * value for weight, value in zip(weights[1:], row))) for weights in self.hidden]
        return hidden, [weights[0] + sum(weight * value for weight, value in zip(weights[1:], hidden)) for weights in self.output]

    def fit(self, rows: list[list[float]], targets: list[list[float]], epochs: int = 120, learning_rate: float = 0.015) -> None:
        for _ in range(epochs):
            for row, target in zip(rows, targets):
                hidden, prediction = self.forward(row)
                errors = [predicted - expected for predicted, expected in zip(prediction, target)]
                hidden_error = [sum(errors[out] * self.output[out][index + 1] for out in range(len(errors))) * (1 - hidden[index] ** 2) for index in range(len(hidden))]
                for out, error in enumerate(errors):
                    self.output[out][0] -= learning_rate * error
                    for index, value in enumerate(hidden, start=1):
                        self.output[out][index] -= learning_rate * error * value
                for index, error in enumerate(hidden_error):
                    self.hidden[index][0] -= learning_rate * error
                    for column, value in enumerate(row, start=1):
                        self.hidden[index][column] -= learning_rate * error * value


class DualParameterEstimator:
    def __init__(self, model: TinyMLP, input_mean: list[float], input_scale: list[float], output_mean: list[float], output_scale: list[float]):
        self.model, self.input_mean, self.input_scale = model, input_mean, input_scale
        self.output_mean, self.output_scale = output_mean, output_scale

    @classmethod
    def train_from_recipes(cls, recipes: list[dict[str, Any]], examples: int = 48, seed: int = 26147) -> "DualParameterEstimator":
        generated = generate_batch(recipes, examples, seed)
        rows = [dsp_features(samples, float(recipes[index % len(recipes)]["sample_rate_hz"])) for index, (samples, _) in enumerate(generated)]
        targets = [truth_from_audit(audit) for _, audit in generated]
        normalized_rows, input_mean, input_scale = standardize(rows)
        normalized_targets, output_mean, output_scale = standardize(targets)
        model = TinyMLP(len(rows[0]), len(PARAMETERS), seed)
        model.fit(normalized_rows, normalized_targets)
        return cls(model, input_mean, input_scale, output_mean, output_scale)

    def learned_estimate(self, samples: list[complex], sample_rate_hz: float) -> dict[str, float]:
        raw = dsp_features(samples, sample_rate_hz)
        row = [(value - self.input_mean[index]) / self.input_scale[index] for index, value in enumerate(raw)]
        _, normalized = self.model.forward(row)
        values = [normalized[index] * self.output_scale[index] + self.output_mean[index] for index in range(len(PARAMETERS))]
        return dict(zip(PARAMETERS, values))

    def compare(self, samples: list[complex], sample_rate_hz: float) -> dict[str, dict[str, float]]:
        dsp, learned = dsp_estimate(samples, sample_rate_hz), self.learned_estimate(samples, sample_rate_hz)
        tolerances = {"dc_i": 0.05, "dc_q": 0.05, "carrier_offset_hz": 250.0}
        return {key: {"dsp": round(dsp[key], 6), "learned": round(learned[key], 6), "absolute_disagreement": round(abs(dsp[key] - learned[key]), 6), "agreement": float(abs(dsp[key] - learned[key]) <= tolerances[key])} for key in PARAMETERS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DEmod dual parameter-estimation MVP in memory.")
    parser.add_argument("--recipes", type=Path, default=Path("data/recipes/mvp-recipes.json"))
    parser.add_argument("--examples", type=int, default=48)
    args = parser.parse_args()
    recipes = load_recipes(args.recipes)
    estimator = DualParameterEstimator.train_from_recipes(recipes, args.examples)
    samples, audit = generate_batch(recipes, 1, 9001)[0]
    print({"truth": dict(zip(PARAMETERS, truth_from_audit(audit))), "comparison": estimator.compare(samples, float(recipes[0]["sample_rate_hz"]))})


if __name__ == "__main__":
    main()
