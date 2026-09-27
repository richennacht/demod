import importlib.util
import sys
import unittest
from pathlib import Path


MODULE = Path(__file__).parents[1] / "src" / "generate_synthetic.py"
SPEC = importlib.util.spec_from_file_location("generate_synthetic", MODULE)
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules["generate_synthetic"] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


class SyntheticGeneratorTests(unittest.TestCase):
    def test_recipe_generation_is_deterministic_and_in_memory(self):
        recipes = GENERATOR.load_recipes(Path(__file__).parents[1] / "data" / "recipes" / "mvp-recipes.json")
        first, truth = GENERATOR.generate_example(recipes[0], seed=42)
        second, repeated_truth = GENERATOR.generate_example(recipes[0], seed=42)
        self.assertEqual(first, second)
        self.assertEqual(truth, repeated_truth)
        self.assertEqual(len(first), 4096)
        self.assertIn("snr_db", truth["impairments"])


if __name__ == "__main__":
    unittest.main()
