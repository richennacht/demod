import importlib.util
import os
import sys
import tempfile
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

    def test_raw_iq_report_marks_sample_rate_as_hypothesis(self):
        descriptor, name = tempfile.mkstemp(suffix=".iq")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(b"\x00\x00\x00\x00" * 16)
            report = ANALYZER.analyse(Path(name), 48_000, "s16le")
        finally:
            Path(name).unlink(missing_ok=True)
        self.assertEqual(report["input"]["sample_rate_source"], "user_hypothesis")
        self.assertIn("spectrum", report["measurements"])


if __name__ == "__main__":
    unittest.main()
