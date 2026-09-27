import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("denoise_iq", ROOT / "src" / "denoise_iq.py")
DENOISE = importlib.util.module_from_spec(SPEC)
sys.modules["denoise_iq"] = DENOISE
SPEC.loader.exec_module(DENOISE)


class DenoiseIQTests(unittest.TestCase):
    def test_default_pipeline_is_bit_for_bit_non_destructive(self):
        source = [complex(index, -index) for index in range(10)]
        derived, audit = DENOISE.apply_pipeline(source)
        self.assertEqual(derived, source)
        self.assertIsNot(derived, source)
        self.assertTrue(audit["raw_preserved"])
        self.assertTrue(all(not action["applied"] for action in audit["actions"]))

    def test_opt_in_dc_offset_removes_robust_location(self):
        source = [complex(3 + index % 2, -2) for index in range(12)]
        derived, audit = DENOISE.apply_pipeline(source, {"dc_offset": {"enabled": True}})
        self.assertAlmostEqual(sum(value.real for value in derived) / len(derived), 0.0, places=6)
        self.assertTrue(audit["actions"][0]["applied"])


if __name__ == "__main__":
    unittest.main()
