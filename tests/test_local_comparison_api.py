import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_comparison_api import ComparisonService


class LocalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ComparisonService(ROOT / "data" / "recipes" / "mvp-recipes.json", examples=12)

    def test_comparison_keeps_raw_bytes_ephemeral_and_labels_scope(self):
        # Eight s16le interleaved I/Q samples: no filesystem recording is made.
        raw = bytes([0, 0, 0, 0, 0, 64, 0, 0, 0, 192, 0, 0, 0, 0, 0, 64] * 2)
        report = self.service.analyse_bytes(raw, "s16le", 1_000_000)
        self.assertFalse(report["provenance"]["raw_data_persisted"])
        self.assertIn("dc_i", report["automated_parameter_comparison"])
        self.assertIn("modulation", report["provenance"]["not_supported_by_model"])

    def test_invalid_format_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.analyse_bytes(b"\x00\x00\x00\x00", "unknown", 1_000_000)
