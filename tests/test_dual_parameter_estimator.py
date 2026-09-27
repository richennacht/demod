import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
GENERATOR_SPEC = importlib.util.spec_from_file_location("generate_synthetic", ROOT / "src" / "generate_synthetic.py")
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
sys.modules["generate_synthetic"] = GENERATOR
GENERATOR_SPEC.loader.exec_module(GENERATOR)
SPEC = importlib.util.spec_from_file_location("dual_parameter_estimator", ROOT / "src" / "dual_parameter_estimator.py")
DUAL = importlib.util.module_from_spec(SPEC)
sys.modules["dual_parameter_estimator"] = DUAL
SPEC.loader.exec_module(DUAL)


class DualParameterEstimatorTests(unittest.TestCase):
    def test_dual_estimator_exposes_both_estimates_and_disagreement(self):
        recipes = GENERATOR.load_recipes(ROOT / "data" / "recipes" / "mvp-recipes.json")
        estimator = DUAL.DualParameterEstimator.train_from_recipes(recipes, examples=12, seed=77)
        samples, _ = GENERATOR.generate_example(recipes[0], 123)
        comparison = estimator.compare(samples, 1_000_000)
        self.assertEqual(set(comparison), {"dc_i", "dc_q", "carrier_offset_hz"})
        self.assertTrue(all("absolute_disagreement" in item for item in comparison.values()))


if __name__ == "__main__":
    unittest.main()
