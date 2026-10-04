import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from amc_models import NearestCentroid, centroid_features, cumulant_classify, cumulant_features, derotate  # noqa: E402
from cfo_estimators import SpecCFO, bins_to_cfo, cfo_to_bin, kay_wpa, lag1_autocorr, luise_reggiannini, power_periodogram, spec_features, tone_crb, wrap  # noqa: E402
from learned_analysis import analysis_region  # noqa: E402
from signal_sim import CONSTELLATIONS, MODULATIONS, SimConfig, rrc_taps, simulate  # noqa: E402

HAS_JAX = importlib.util.find_spec("jax") is not None
SPEC_MODEL = ROOT / "data" / "models" / "speccfo.npz"
AMC_MODEL = ROOT / "data" / "models" / "demod_amc.npz"


def batch(mod, snr, count=24, seed=0, cfo=(-0.1, 0.1), **fixed):
    return simulate(np.random.default_rng(seed), count, SimConfig(cfo=cfo, fixed={"modulation": mod, "snr_db": snr, **fixed}))


class SimulatorTests(unittest.TestCase):
    def test_labels_shapes_and_determinism(self):
        a, b = batch("qpsk", 15, seed=3), batch("qpsk", 15, seed=3)
        self.assertEqual(a["x"].shape, (24, 1024))
        np.testing.assert_array_equal(a["x"], b["x"])
        self.assertTrue(np.all(np.abs(a["cfo"]) <= 0.1))
        self.assertTrue(np.all(a["modulation"] == MODULATIONS.index("qpsk")))

    def test_every_modulation_generates_finite_samples(self):
        for mod in MODULATIONS:
            x = batch(mod, 10, count=3)["x"]
            self.assertTrue(np.all(np.isfinite(x)), mod)

    def test_constellations_have_unit_power_and_rrc_has_unit_energy(self):
        for name, pts in CONSTELLATIONS.items():
            self.assertAlmostEqual(float(np.mean(np.abs(pts) ** 2)), 1.0, places=6, msg=name)
        self.assertAlmostEqual(float(np.sum(rrc_taps(8, 0.35) ** 2)), 1.0, places=6)


class ClassicalEstimatorTests(unittest.TestCase):
    def test_wrap_maps_into_half_open_interval(self):
        np.testing.assert_allclose(wrap(np.array([0.6, -0.6, 0.5, 0.0])), [-0.4, 0.4, -0.5, 0.0], atol=1e-12)

    def test_kay_and_luise_reggiannini_recover_a_clean_tone_inside_their_range(self):
        n = np.arange(1024)
        tone = np.exp(2j * np.pi * 0.0731 * n + 1j)[None, :]
        self.assertAlmostEqual(float(kay_wpa(tone)[0]), 0.0731, places=6)
        self.assertAlmostEqual(float(luise_reggiannini(tone, lags=8)[0]), 0.0731, places=6)

    def test_luise_reggiannini_wraps_beyond_its_unambiguous_range(self):
        # Documented limit |f| < 1/(L+1): with 64 lags a 0.0731 tone aliases.
        n = np.arange(1024)
        tone = np.exp(2j * np.pi * 0.0731 * n)[None, :]
        self.assertGreater(abs(float(luise_reggiannini(tone, lags=64)[0]) - 0.0731), 0.01)

    def test_power_periodogram_is_exact_inside_its_range_and_ambiguous_by_one_over_m_outside(self):
        inside = batch("qpsk", 20, count=16, cfo=(-0.12, 0.12))
        est, strength = power_periodogram(inside["x"], 4)
        self.assertLess(np.max(np.abs(wrap(est - inside["cfo"]))), 1e-4)
        self.assertGreater(np.median(strength), 20)
        # Beyond +-1/(2M) the x^M line aliases, so the error is a multiple of 1/M (here 0.25).
        outside = batch("qpsk", 20, count=16, cfo=(0.14, 0.19))
        err = wrap(power_periodogram(outside["x"], 4)[0] - outside["cfo"])
        self.assertLess(np.max(np.abs((err + 0.125) % 0.25 - 0.125)), 1e-4)
        self.assertGreater(np.min(np.abs(err)), 0.2)
        self.assertGreater(np.max(np.abs(wrap(lag1_autocorr(outside["x"], 4) - outside["cfo"]))), 0.01)

    def test_crb_decreases_with_snr_and_length(self):
        self.assertLess(tone_crb(20, 1024), tone_crb(0, 1024))
        self.assertLess(tone_crb(10, 2048), tone_crb(10, 512))


class FeatureTests(unittest.TestCase):
    def test_spec_features_shape_range_and_bin_mapping(self):
        f = spec_features(batch("bpsk", 10, count=4)["x"])
        self.assertEqual(f.shape, (4, 4, 512))
        self.assertTrue(np.all(np.isfinite(f)) and f.min() >= -1.0 and f.max() <= 4.0)
        cfo = np.array([-0.2, -0.01, 0.0, 0.123])
        np.testing.assert_allclose(bins_to_cfo(cfo_to_bin(cfo)), cfo, atol=1 / 512)

    def test_spec_features_line_sits_at_the_true_offset_up_to_the_m_fold_alias(self):
        # The M=4 channel shows the line at f + k/4. The network resolves k using the M=1 and M=2 channels.
        b = batch("qpsk", 25, count=8, cfo=(-0.1, 0.1))
        f = spec_features(b["x"])
        peak = bins_to_cfo(np.argmax(f[:, 2], axis=1))
        err = wrap(peak - b["cfo"])
        self.assertLess(np.max(np.abs((err + 0.125) % 0.25 - 0.125)), 3 / 512)

    def test_vectorised_centroid_features_match_the_shipped_function(self):
        from modulation_classifier import features
        b = batch("16qam", 10, count=3)
        mine = centroid_features(b["x"])
        for row, x in zip(mine, b["x"]):
            np.testing.assert_allclose(row, features([complex(v) for v in x]), rtol=1e-6, atol=1e-9)

    def test_cumulants_classify_symbol_spaced_linear_modulations_without_carrier_offset(self):
        # Swami and Sadler assume synchronised, symbol-spaced samples. Pulse shaping and CFO break that.
        rng = np.random.default_rng(0)
        for name in ("bpsk", "qpsk", "8psk"):
            pts = CONSTELLATIONS[name]
            x = pts[rng.integers(0, len(pts), (40, 1024))] + 0.03 * (rng.standard_normal((40, 1024)) + 1j * rng.standard_normal((40, 1024)))
            self.assertGreater(np.mean(cumulant_classify(x) == MODULATIONS.index(name)), 0.95, name)
        self.assertEqual(cumulant_features(x).shape, (40, 2))

    def test_nearest_centroid_fits_and_predicts(self):
        rng = np.random.default_rng(1)
        feats = np.concatenate([rng.normal(0, 1, (50, 3)), rng.normal(6, 1, (50, 3))])
        labels = np.array([0] * 50 + [1] * 50)
        self.assertGreater(np.mean(NearestCentroid().fit(feats, labels).predict(feats) == labels), 0.95)

    def test_derotate_undoes_a_carrier_offset(self):
        x = np.ones((1, 256), dtype=complex) * np.exp(2j * np.pi * 0.05 * np.arange(256))
        np.testing.assert_allclose(derotate(x, np.array([0.05])), np.ones((1, 256)), atol=1e-9)


class RegionSelectionTests(unittest.TestCase):
    def test_longest_energy_segment_is_preferred(self):
        samples = np.arange(20000, dtype=complex)
        region, info = analysis_region(samples, [{"sample_start": 100, "sample_count": 1500}, {"sample_start": 5000, "sample_count": 8000}])
        self.assertEqual((info["source"], info["sample_start"], len(region)), ("longest_energy_segment", 5000, 8000))

    def test_falls_back_to_the_start_when_segments_are_short_or_missing(self):
        region, info = analysis_region(np.zeros(3000, dtype=complex), [{"sample_start": 10, "sample_count": 200}])
        self.assertEqual((info["source"], len(region)), ("capture_start", 3000))


@unittest.skipUnless(HAS_JAX and SPEC_MODEL.exists(), "needs JAX and data/models/speccfo.npz")
class SpecCfoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SpecCFO(SPEC_MODEL)

    def test_numpy_inference_matches_the_jax_network_it_was_trained_with(self):
        import jax.numpy as jnp
        import jax_models as jm
        feats = spec_features(batch("qpsk", 12, count=6, seed=5)["x"])
        params = {k: jnp.asarray(v) for k, v in self.model.params.items()}
        expected = np.asarray(jm.speccfo_apply(params, {}, jnp.asarray(feats), False)[0])
        np.testing.assert_allclose(self.model.logits(feats), expected, rtol=2e-3, atol=2e-3)

    def test_recovers_psk_offset_across_the_full_range_at_moderate_snr(self):
        b = batch("qpsk", 12, count=32, seed=8, cfo=(-0.2, 0.2))
        r = self.model.estimate(b["x"])
        self.assertLess(np.median(np.abs(wrap(r["cfo"] - b["cfo"]))), 1e-3)
        self.assertTrue(np.all((r["confidence"] >= 0) & (r["confidence"] <= 1.0001)))

    def test_estimate_is_in_cycles_per_sample_so_it_scales_with_any_sample_rate(self):
        b = batch("bpsk", 20, count=4, seed=2)
        cfo = self.model.estimate(b["x"])["cfo"]
        for fs in (48_000.0, 250_000.0, 20_000_000.0):
            self.assertTrue(np.all(np.abs(cfo * fs) <= 0.5 * fs))


@unittest.skipUnless(HAS_JAX and SPEC_MODEL.exists() and AMC_MODEL.exists(), "needs JAX and data/models/*.npz")
class DemodAmcModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from amc_models import DemodAMC
        cls.model = DemodAMC(AMC_MODEL)

    def test_numpy_inference_matches_the_jax_network_it_was_trained_with(self):
        import jax.numpy as jnp
        import jax_models as jm
        from amc_models import time_features
        x = derotate(batch("16qam", 15, count=4, seed=6)["x"], np.zeros(4))
        tf, sf = time_features(x), spec_features(x)
        raw = {k.removeprefix("params/"): v for k, v in self.model.p.items() if k.startswith("params/")}
        tree = {}
        for key, value in raw.items():
            layer, leaf = key.split("/")
            tree.setdefault(layer, {})[leaf] = jnp.asarray(value)
        expected = np.asarray(jm.demod_amc_apply(tree, {}, (jnp.asarray(tf), jnp.asarray(sf)), False)[0])
        np.testing.assert_allclose(self.model.logits(tf, sf), expected, rtol=2e-3, atol=2e-3)

    def test_probabilities_are_normalised_and_fixed_by_seed(self):
        x = batch("qpsk", 15, count=5, seed=4)["x"]
        p = self.model.probabilities(x)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(p, self.model.probabilities(x), atol=1e-6)


if __name__ == "__main__":
    unittest.main()


PINNED_REAL_SUM = -148.3654  # sum of x.real for rng(101), 8 QPSK 10 dB signals, recorded when augmentation was added


class AugmentationTests(unittest.TestCase):
    def test_every_pulse_has_unit_energy(self):
        from signal_sim import PULSES, pulse_taps
        for kind in PULSES:
            self.assertAlmostEqual(float(np.sum(pulse_taps(kind, 4, 0.35) ** 2)), 1.0, places=6, msg=kind)

    def test_augmentation_off_reproduces_the_paper_condition_stream_exactly(self):
        # Fair comparison depends on the v1 test sets not changing when augmentation code is present.
        fixed = {"modulation": "qpsk", "snr_db": 10}
        default = simulate(np.random.default_rng(101), 8, SimConfig(fixed=fixed))
        explicit = simulate(np.random.default_rng(101), 8, SimConfig(fixed=fixed, pulses=("rrc",), phase_noise_probability=0.0, interferer_probability=0.0, impulse_probability=0.0, dc_probability=0.0))
        np.testing.assert_array_equal(default["x"], explicit["x"])
        self.assertAlmostEqual(float(default["x"].real.sum()), PINNED_REAL_SUM, places=2)

    def test_augmented_signals_are_finite_with_unchanged_labels(self):
        cfg = SimConfig(pulses=("rrc", "rect", "rc", "halfsine", "gauss"), phase_noise_probability=1, interferer_probability=1, impulse_probability=1, dc_probability=1, fixed={"modulation": "qpsk", "snr_db": 20, "sps": 4})
        a = simulate(np.random.default_rng(5), 32, cfg)
        self.assertTrue(np.all(np.isfinite(a["x"])))
        self.assertTrue(np.all(a["modulation"] == MODULATIONS.index("qpsk")))
        self.assertTrue(np.all(np.abs(a["cfo"]) <= 0.2))

    def test_rectangular_pulse_qpsk_still_has_its_x4_line_at_the_true_offset(self):
        b = simulate(np.random.default_rng(2), 12, SimConfig(cfo=(-0.1, 0.1), pulses=("rect",), fixed={"modulation": "qpsk", "snr_db": 25, "sps": 4}))
        est, _ = power_periodogram(b["x"], 4)
        self.assertLess(np.max(np.abs(wrap(est - b["cfo"]))), 1e-3)



class SymbolSpacedTests(unittest.TestCase):
    def test_one_sample_per_symbol_gives_independent_samples_for_linear_modulations(self):
        b = simulate(np.random.default_rng(4), 16, SimConfig(cfo=(0, 0), pulses=("rrc", "rect"), fixed={"modulation": "qpsk", "snr_db": 30, "sps": 1}))
        lag1 = np.abs(np.mean(b["x"][:, 1:] * np.conj(b["x"][:, :-1]), axis=1))
        self.assertLess(float(np.max(lag1)), 0.15)

    def test_nonlinear_modulations_are_never_generated_at_one_sample_per_symbol(self):
        b = simulate(np.random.default_rng(4), 6, SimConfig(fixed={"modulation": "2fsk", "snr_db": 20, "sps": 1}))
        self.assertTrue(np.all(b["sps"] == 2))


@unittest.skipUnless(HAS_JAX, "needs JAX")
class PositionChannelTests(unittest.TestCase):
    def test_numpy_inference_matches_jax_for_a_model_with_position_channels(self):
        import json
        import tempfile
        import jax
        import jax.numpy as jnp
        import jax_models as jm
        params, _ = jm.speccfo_init(jax.random.PRNGKey(3), coords=True)
        self.assertEqual(params["in_w"].shape[1], 8)
        path = Path(tempfile.mkdtemp()) / "m.npz"
        np.savez(path, meta=json.dumps({}), **{f"params/{k}": np.asarray(v) for k, v in params.items()})
        model = SpecCFO(path)
        feats = spec_features(batch("qpsk", 10, count=3, seed=1)["x"])
        expected = np.asarray(jm.speccfo_apply(params, {}, jnp.asarray(feats), False)[0])
        np.testing.assert_allclose(model.logits(feats), expected, atol=1e-4)

    def test_models_without_position_channels_still_load_and_ignore_them(self):
        import jax
        import jax_models as jm
        params, _ = jm.speccfo_init(jax.random.PRNGKey(3), coords=False)
        self.assertEqual(params["in_w"].shape[1], 4)
