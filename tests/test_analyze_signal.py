import importlib.util
import sys
import unittest
from pathlib import Path

MODULE = Path(__file__).parents[1] / "src" / "analyze_signal.py"
SPEC = importlib.util.spec_from_file_location("analyze_signal", MODULE)
ANALYZER = importlib.util.module_from_spec(SPEC)
sys.modules["analyze_signal"] = ANALYZER
SPEC.loader.exec_module(ANALYZER)


class SignalAnalysisTests(unittest.TestCase):
    def test_summary_reports_duration_and_crossing_frequency(self):
        report = ANALYZER.summarise_real([-1.0, 1.0, -1.0, 1.0], 8)
        self.assertEqual(report["sample_rate_hz"], 8)
        self.assertEqual(report["duration_seconds"], 0.5)
        self.assertAlmostEqual(report["zero_crossing_frequency_hz"], 4.0)


if __name__ == "__main__":
    unittest.main()
