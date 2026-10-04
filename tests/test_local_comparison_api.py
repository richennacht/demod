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


class LocalUiServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import ThreadingHTTPServer
        from local_comparison_api import make_handler
        service = ComparisonService(ROOT / "data" / "recipes" / "mvp-recipes.json", examples=12)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path, method="GET"):
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + path, method=method)) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def test_health_lists_supported_inputs_for_the_ui(self):
        import json
        status, _, body = self._get("/health")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("s16le", payload["supported_formats"])
        self.assertEqual(payload["supported_demodulations"], ["bpsk", "qpsk", "2fsk"])

    def test_ui_is_served_locally_without_path_traversal(self):
        status, headers, body = self._get("/ui/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"app.js", body)
        self.assertEqual(self._get("/ui/../src/provenance.py")[0], 404)
        self.assertEqual(self._get("/ui/%2e%2e/README.md")[0], 404)

    def test_preflight_allows_private_network_access(self):
        status, headers, _ = self._get("/analyse", method="OPTIONS")
        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Private-Network"), "true")


MODELS_PRESENT = (ROOT / "data" / "models" / "speccfo.npz").exists() and (ROOT / "data" / "models" / "demod_amc.npz").exists()


def _qpsk_burst_bytes(cfo=0.006, sps=8, symbols=1200, snr_db=20, seed=3):
    import numpy as np
    rng = np.random.default_rng(seed)
    pts = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, symbols)))
    x = np.concatenate([np.zeros(3000, complex), np.repeat(pts, sps), np.zeros(3000, complex)])
    x = x * np.exp(2j * np.pi * cfo * np.arange(len(x)))
    x += (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) * 10 ** (-snr_db / 20) / 2 ** 0.5 * (np.arange(len(x)) >= 3000) * (np.arange(len(x)) < 3000 + symbols * sps)
    iq = np.empty(2 * len(x), dtype="<i2")
    iq[0::2], iq[1::2] = np.round(x.real * 8000), np.round(x.imag * 8000)
    return iq.tobytes()


class LearnedModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ComparisonService(ROOT / "data" / "recipes" / "mvp-recipes.json", examples=12)
        cls.raw = _qpsk_burst_bytes()

    @unittest.skipUnless(MODELS_PRESENT, "needs data/models/*.npz")
    def test_carrier_offset_is_scaled_to_the_stated_sample_rate_and_reports_confidence(self):
        for fs in (250_000.0, 1_000_000.0):
            report = self.service.analyse_bytes(self.raw, "s16le", fs)
            row = report["automated_parameter_comparison"]["carrier_offset_hz"]
            truth_hz = 0.006 * fs
            self.assertLess(abs(row["learned"] - truth_hz), 0.002 * fs / 100, fs)  # within 0.002 percent of fs
            self.assertTrue(0 <= row["learned_detail"]["confidence"] <= 1)
            self.assertLessEqual(abs(row["learned"]), fs / 2)
            self.assertEqual(report["learned_region"]["source"], "longest_energy_segment")

    @unittest.skipUnless(MODELS_PRESENT, "needs data/models/*.npz")
    def test_classifier_returns_probabilities_names_the_class_and_keeps_the_legacy_answer(self):
        report = self.service.analyse_bytes(self.raw, "s16le", 250_000.0)
        cls = report["modulation_classification"]
        self.assertEqual(cls["predicted_modulation"], "qpsk")
        self.assertAlmostEqual(sum(c["probability"] for c in cls["ranked_candidates"]), 1.0, delta=0.05)
        self.assertIn("legacy_centroid", cls)
        self.assertIn("calibrated", cls["model"])

    def test_falls_back_to_the_legacy_models_when_model_files_are_absent(self):
        import cfo_estimators
        import amc_models
        saved = (cfo_estimators._DEFAULT, amc_models._DEFAULT, cfo_estimators.MODEL_PATH, amc_models.MODEL_PATH)
        try:
            missing = ROOT / "data" / "models" / "does-not-exist.npz"
            cfo_estimators._DEFAULT, amc_models._DEFAULT = None, None
            cfo_estimators.MODEL_PATH = amc_models.MODEL_PATH = missing
            report = self.service.analyse_bytes(self.raw, "s16le", 250_000.0)
            self.assertNotIn("learned_detail", report["automated_parameter_comparison"]["carrier_offset_hz"])
            self.assertIn("distance", report["modulation_classification"]["ranked_candidates"][0])
        finally:
            cfo_estimators._DEFAULT, amc_models._DEFAULT, cfo_estimators.MODEL_PATH, amc_models.MODEL_PATH = saved
