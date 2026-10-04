"""Documents a known defect in src/generate_synthetic.py.

generate_example() builds linear signals with
    [points[rng.randrange(len(points))] for _ in range(symbols) for _ in range(sps)]
which draws a new random point for every SAMPLE, so `samples_per_symbol` has no effect and
the recipe signals are effectively one sample per symbol. A held-symbol signal would have a
lag-one autocorrelation near (sps-1)/sps.

This test is expected to fail until the generator holds each symbol for `sps` samples.
When that is fixed it will start passing: remove the decorator then, and retrain anything
that was fitted on recipe data (the shipped TinyMLP and centroid classifier).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_synthetic import generate_batch, load_recipes  # noqa: E402


class RecipeGeneratorTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_recipe_signals_hold_each_symbol_for_samples_per_symbol(self):
        recipe = load_recipes(ROOT / "data" / "recipes" / "mvp-recipes.json")[:1]  # clean stage, 4 samples/symbol
        x = np.asarray(generate_batch(recipe, 1, 91919)[0][0])
        lag1 = abs(np.mean(x[1:] * np.conj(x[:-1])))
        self.assertGreater(lag1, 0.5)


if __name__ == "__main__":
    unittest.main()
