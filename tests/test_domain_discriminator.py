import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
GENERATOR_SPEC = importlib.util.spec_from_file_location("generate_synthetic", ROOT / "src" / "generate_synthetic.py")
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
sys.modules["generate_synthetic"] = GENERATOR
GENERATOR_SPEC.loader.exec_module(GENERATOR)
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


if __name__ == "__main__":
    unittest.main()
