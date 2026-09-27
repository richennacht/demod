import importlib.util
import unittest
from pathlib import Path


MODULE = Path(__file__).parents[1] / "src" / "powder_stream.py"
SPEC = importlib.util.spec_from_file_location("powder_stream", MODULE)
STREAM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STREAM)


class PowderStreamTests(unittest.TestCase):
    def test_manifest_has_all_round_gain_state_combinations(self):
        urls = STREAM.capture_urls()
        self.assertEqual(len(urls), 768)
        self.assertEqual(len(set(urls)), 768)
        self.assertIn("Round3_Gain90/otax3102node_tx111111_id63.bin", urls[-1])


if __name__ == "__main__":
    unittest.main()
