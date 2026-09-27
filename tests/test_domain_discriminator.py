import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
GENERATOR_SPEC = importlib.util.spec_from_file_location("generate_synthetic", ROOT / "src" / "generate_synthetic.py")
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
sys.modules["generate_synthetic"] = GENERATOR
GENERATOR_SPEC.loader.exec_module(GENERATOR)
STREAM_SPEC = importlib.util.spec_from_file_location("powder_stream", ROOT / "src" / "powder_stream.py")
STREAM = importlib.util.module_from_spec(STREAM_SPEC)
sys.modules["powder_stream"] = STREAM
STREAM_SPEC.loader.exec_module(STREAM)
SPEC = importlib.util.spec_from_file_location("domain_discriminator", ROOT / "src" / "domain_discriminator.py")
DISCRIMINATOR = importlib.util.module_from_spec(SPEC)
sys.modules["domain_discriminator"] = DISCRIMINATOR
SPEC.loader.exec_module(DISCRIMINATOR)


class DomainDiscriminatorTests(unittest.TestCase):
    def test_feature_vector_is_finite_and_stable(self):
        samples, _ = GENERATOR.generate_example(GENERATOR.load_recipes(ROOT / "data" / "recipes" / "mvp-recipes.json")[0], 99)
        vector = DISCRIMINATOR.features(samples[:256])
        self.assertEqual(len(vector), 16)
        self.assertTrue(all(abs(value) < 1e9 for value in vector))

    def test_wilson_interval_is_bounded_and_contains_observed_accuracy(self):
        lower, upper = DISCRIMINATOR.wilson_interval(50, 100)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

    def test_calibration_acceptance_needs_precision(self):
        report = DISCRIMINATOR.calibration_report_samples([[complex(index, -index) for index in range(8)] for _ in range(6)], ROOT / "data" / "recipes" / "mvp-recipes.json", window=8, recipe_id="stage-0-clean-linear")
        self.assertFalse(report["passes_precision_gate"])


if __name__ == "__main__":
    unittest.main()
