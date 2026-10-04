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
        self.assertIn("FEC", report["provenance"]["not_supported_by_model"])
        self.assertIn("modulation_classification", report)
        self.assertIn("raw_branch", report["analysis"])

    def test_invalid_format_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.analyse_bytes(b"\x00\x00\x00\x00", "unknown", 1_000_000)

    def test_manual_demodulation_returns_bits_and_gnu_radio_graph(self):
        # BPSK 0,1,0,1 at four samples per symbol in interleaved s16le.
        values = []
        for bit in (0, 1, 0, 1):
            for _ in range(4):
                values.extend(((-16384 if bit == 0 else 16384).to_bytes(2, "little", signed=True), (0).to_bytes(2, "little", signed=True)))
        report = self.service.demodulate_bytes(b"".join(values), "s16le", 1_000_000, "bpsk", 4)
        self.assertEqual(report["bits_preview"], "0101")
        self.assertIn("gnu_radio_graph", report)
