import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    SPEC = importlib.util.spec_from_file_location("spectral_analysis", ROOT / "src" / "spectral_analysis.py")
    SPECTRAL = importlib.util.module_from_spec(SPEC)
    sys.modules["spectral_analysis"] = SPECTRAL
    SPEC.loader.exec_module(SPECTRAL)


@unittest.skipUnless(HAS_NUMPY, "NumPy-backed DSP tests require requirements-dsp.txt")
class SpectralAnalysisTests(unittest.TestCase):
    def test_fft_finds_complex_tone_and_emits_plots(self):
        fs, tone = 1024, 128
        signal = [complex(math.cos(2 * math.pi * tone * index / fs), math.sin(2 * math.pi * tone * index / fs)) for index in range(1024)]
        report = SPECTRAL.analyse_spectrum(signal, fs)
        self.assertAlmostEqual(report["spectrum"]["peak_frequency_hz"], tone, delta=2)
        self.assertTrue(report["waterfall"]["power_db_relative"])
        self.assertEqual(report["provenance"]["denoising_applied"], False)

    def test_energy_segment_finds_inserted_burst(self):
        signal = [0j] * 100 + [2 + 0j] * 200 + [0j] * 100
        report = SPECTRAL.segment_energy(SPECTRAL.np.asarray(signal), 1000, window=16, min_duration_ms=20)
        self.assertTrue(report["segments"])

    def test_welch_and_max_hold_see_a_burst_the_first_block_misses(self):
        fs, tone = 8000, 1000
        quiet = [0j] * 4096
        burst = [complex(math.cos(2 * math.pi * tone * n / fs), math.sin(2 * math.pi * tone * n / fs)) for n in range(4096)]
        report = SPECTRAL.analyse_spectrum(quiet + burst, fs)
        spectrum = report["spectrum"]
        peak = spectrum["frequency_hz"][spectrum["welch_power_db"].index(max(spectrum["welch_power_db"]))]
        self.assertAlmostEqual(peak, tone, delta=20)
        self.assertGreaterEqual(max(spectrum["max_hold_db"]), max(spectrum["welch_power_db"]))
        self.assertGreater(spectrum["welch_frames"], 1)

    def test_global_waterfall_keeps_quiet_frames_dark_and_time_preview_finds_burst(self):
        fs = 8000
        signal = [0.001 + 0j] * 4096 + [1 + 0j, -1 + 0j] * 2048
        report = SPECTRAL.analyse_spectrum(signal, fs)
        rows = report["waterfall"]["power_db_global"]
        self.assertLess(max(rows[0]), max(rows[-1]) - 20)
        preview = report["time_preview"]
        self.assertEqual(len(preview["i"]), preview["sample_count"])
        self.assertTrue(report["segmentation"]["envelope"]["power_db"])


if __name__ == "__main__":
    unittest.main()
