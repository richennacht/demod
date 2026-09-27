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

    def test_rf_impairment_recipe_exposes_five_new_controls(self):
        recipe = GENERATOR.load_recipes(Path(__file__).parents[1] / "data" / "recipes" / "mvp-recipes.json")[1]
        _, truth = GENERATOR.generate_example(recipe, seed=13)
        controls = truth["impairments"]
        for key in ("colored_noise_std", "tone_interferer_amplitude", "burst_probability", "cochannel_interferer_amplitude", "adc_bits"):
            self.assertIn(key, controls)
        self.assertEqual(controls["adc_bits"], 10)

    def test_powder_recipe_has_matched_rate_and_receiver_scale(self):
        recipe = next(item for item in GENERATOR.load_recipes(Path(__file__).parents[1] / "data" / "recipes" / "mvp-recipes.json") if item["recipe_id"] == "stage-2-powder-ofdm")
        samples, truth = GENERATOR.generate_example(recipe, seed=26147)
        rms = (sum(abs(sample) ** 2 for sample in samples) / len(samples)) ** 0.5
        self.assertEqual(truth["recipe_id"], "stage-2-powder-ofdm")
        self.assertEqual(recipe["sample_rate_hz"], 33333000)
        self.assertIn(round(rms, 3), {round(value, 3) for value in recipe["impairments"]["target_rms"]})


if __name__ == "__main__":
    unittest.main()
