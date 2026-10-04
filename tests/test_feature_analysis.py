import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("feature_analysis", ROOT / "src" / "feature_analysis.py")
FEATURES = importlib.util.module_from_spec(SPEC)
sys.modules["feature_analysis"] = FEATURES
SPEC.loader.exec_module(FEATURES)


class FeatureAnalysisTests(unittest.TestCase):
    def test_tone_reports_peak_and_bandwidth(self):
        sample_rate = 1024
        signal = [complex(__import__("math").cos(2 * __import__("math").pi * 80 * n / sample_rate), __import__("math").sin(2 * __import__("math").pi * 80 * n / sample_rate)) for n in range(512)]
        report = FEATURES.analyse_iq(signal, sample_rate)
        self.assertAlmostEqual(report["spectrum"]["peak_frequency_hz"], 80, delta=4)
        self.assertLess(report["spectrum"]["occupied_bandwidth_99pct_hz"], 40)
        self.assertEqual(len(report["modulation_hypotheses"]), 4)


if __name__ == "__main__":
    unittest.main()


class FeatureAnalysisScalingTests(unittest.TestCase):
    def test_long_capture_features_stay_linear_time(self):
        # Regression guard: means/medians recomputed inside comprehensions made
        # a 30k-sample /analyse call take minutes. This bound is deliberately loose.
        import math
        import time
        from denoise_iq import apply_pipeline
        samples = [complex(math.cos(index * 0.3), math.sin(index * 0.3)) for index in range(20000)]
        started = time.perf_counter()
        FEATURES.analyse_iq(samples, 250_000)
        apply_pipeline(samples, {"impulse_blanking": {"enabled": True}})
        self.assertLess(time.perf_counter() - started, 15.0)
