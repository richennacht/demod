import io
import sys
import unittest
import wave
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rate_estimation import estimate, wav_metadata
from signal_sim import simulate, SimConfig


class RateTests(unittest.TestCase):
    def test_scale_identifiability_and_known_reference(self):
        x = simulate(np.random.default_rng(49), 1, SimConfig(length=4096, fixed={'sps': 8, 'modulation': 'qpsk', 'snr_db': 25}))['x'][0]
        r = estimate(x, known_symbol_rate_hz=31250)
        self.assertIsNone(r['absolute_sample_rate_hz'])
        self.assertEqual(r['learned']['samples_per_symbol'], 8)
        self.assertEqual(r['conditional_sample_rate_hz'], 250000)
        a, b = estimate(x, 250000), estimate(x, 500000)
        self.assertEqual(a['learned'], b['learned'])
        self.assertAlmostEqual(b['manual']['candidates'][0]['symbol_rate_hz'], 2*a['manual']['candidates'][0]['symbol_rate_hz'])

    def test_noise_and_silence_abstain(self):
        for x in (np.zeros(2048), np.random.default_rng(99).normal(size=2048) + 1j*np.random.default_rng(100).normal(size=2048)):
            self.assertTrue(estimate(x)['learned']['abstained'])

    def test_duration_uses_whole_capture_before_preview_truncation(self):
        r = estimate(np.random.default_rng(9).normal(size=12000), recording_duration_seconds=.05)
        self.assertEqual(r['absolute_sample_rate_hz'], 240000)
        self.assertEqual(r['sample_rate_source'], 'sample_count / analyst_recording_duration')

    def test_wav_header_rate_and_channels(self):
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000); w.writeframes(b'\0'*512)
        meta, samples = wav_metadata(buf.getvalue())
        self.assertEqual(meta['sample_rate_hz'], 48000)
        self.assertEqual(meta['source'], 'wav_header')
        self.assertEqual(samples.shape, (128, 2))
