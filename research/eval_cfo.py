"""Evaluate every CFO estimator on identical, fixed-seed test sets.

  python research/eval_cfo.py [--quick]

Sets mirror the experiments in the recreated papers plus the repo's own metrics:

  awgn / rayleigh     Chen et al. 2023 Fig. 3 and 4: all modulations, SNR sweep, L=1024
  bpsk_oversampling   Chen et al. 2023 Fig. 5: BPSK at 4, 8, 16 samples/symbol
  bpsk_length         Chen et al. 2023 Fig. 6: BPSK at 512, 1024, 2048 samples
  oshea_channel       O'Shea et al. 2017 Figs. 4-7: QPSK, RRC 0.25, 4 sps, +-50 kHz at
                      400 kHz, SNR 0/5/10 dB, AWGN and Rayleigh sigma 0.5/1/2, block sweep
  repo_native         the repo's own recipe signals (1 MS/s, +-2.5 kHz, 4096 samples) with
                      its README tolerance of 250 Hz, scored against the shipped TinyMLP

Networks trained at 1024 samples (IQ-ResNet, O'Shea CNN) are only scored at 1024;
other lengths show n/a rather than an unfair extrapolation.

Writes research/results/cfo_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from cfo_estimators import SpecCFO, blind_multi_m, kay_wpa, lag1_autocorr, luise_reggiannini, power_periodogram, tone_crb, wrap  # noqa: E402
from signal_sim import MODULATIONS, SimConfig, simulate  # noqa: E402

OUT = ROOT / "research" / "results"
CKPT = ROOT / "research" / "checkpoints"
SNRS = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
MODS = [m for m in MODULATIONS if m != "noise"]
GENIE_M = {"bpsk": 2, "qpsk": 4, "8psk": 8, "16qam": 4, "64qam": 4}
GROSS = 0.005  # cycles/sample, 1.25 kHz at 250 kS/s
NET_LENGTH = 1024
SPEC_V2 = ROOT / "data" / "models" / "speccfo_v2.npz"
SPEC_V2_NOCOORDS = CKPT / "speccfo_v2_nocoords.npz"  # ablation: same data as v2, no absolute-position channels


def build(rng, mods, snrs, per_cell, length=1024, cfo=(-0.2, 0.2), rayleigh=0.0, **fixed):
    parts = []
    for mod in mods:
        for snr in snrs:
            cfg = SimConfig(length=length, cfo=cfo, rayleigh_probability=rayleigh, fixed={"modulation": mod, "snr_db": snr, **fixed})
            parts.append(simulate(rng, per_cell, cfg))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def by_genie(x, modulation, fn):
    """Apply fn(x_subset, M) grouped by the modulation's order so each call is vectorised."""
    out = np.empty(len(x))
    ms = np.array([GENIE_M.get(MODULATIONS[i], 1) for i in modulation])
    for m in np.unique(ms):
        sel = ms == m
        out[sel] = fn(x[sel], int(m))
    return out


def load_net(name):
    import jax
    from train_cfo import MODELS, iq_tensor
    data = np.load(CKPT / f"{name}.npz", allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    init, apply, _ = MODELS[name]
    params, state = init(jax.random.PRNGKey(0))
    leaves, treedef = jax.tree_util.tree_flatten_with_path({"params": params, "state": state})
    tree = jax.tree_util.tree_unflatten(treedef, [data["/".join(str(getattr(k, "key", k)) for k in p)] for p, _ in leaves])
    fn = jax.jit(lambda xb: apply(tree["params"], tree["state"], xb, False)[0])

    def run(x):
        return np.concatenate([np.asarray(fn(iq_tensor(x[i:i + 256]))) for i in range(0, len(x), 256)]) * meta["cfo_scale"]
    return run


def fast_spec(spec: SpecCFO) -> SpecCFO:
    import jax
    import jax.numpy as jnp
    import jax_models as jm
    sp = {k: jnp.asarray(v) for k, v in spec.params.items()}
    fn = jax.jit(lambda f: jm.speccfo_apply(sp, {}, f, False)[0])
    spec.logits_override = lambda f: np.asarray(fn(f))
    return spec


def method_table(spec):
    m = {
        "repo_dsp_lag1_x4": lambda b: lag1_autocorr(b["x"], 4),
        "kay_x1": lambda b: kay_wpa(b["x"], 1),
        "kay_genie_M": lambda b: by_genie(b["x"], b["modulation"], lambda x, k: kay_wpa(x, k)),
        "luise_reggiannini_genie_M": lambda b: by_genie(b["x"], b["modulation"], lambda x, k: luise_reggiannini(x, k, lags=min(8, x.shape[1] // 4))),
        "periodogram_genie_M": lambda b: by_genie(b["x"], b["modulation"], lambda x, k: power_periodogram(x, k)[0]),
        "blind_multi_M": lambda b: blind_multi_m(b["x"])[0],
    }
    for name in ("iq_resnet", "oshea_cfo"):
        if (CKPT / f"{name}.npz").exists():
            run = load_net(name)
            m[name] = (lambda r: (lambda b: r(b["x"]) if b["x"].shape[1] == NET_LENGTH else None))(run)
    if spec is not None:
        def batched(b, **kw):
            return np.concatenate([spec.estimate(b["x"][i:i + 256], **kw)["cfo"] for i in range(0, len(b["x"]), 256)])
        m["speccfo_coarse"] = lambda b: batched(b, refine=False)
        m["speccfo"] = lambda b: batched(b)
    if SPEC_V2.exists():
        spec2 = fast_spec(SpecCFO(SPEC_V2))
        m["speccfo_v2"] = lambda b: np.concatenate([spec2.estimate(b["x"][i:i + 256])["cfo"] for i in range(0, len(b["x"]), 256)])
    return m


def summarise(err, b, fs=None):
    e = np.abs(err)
    out = {"n": int(len(e)), "mse": float(np.mean(e**2)), "rmse": float(np.sqrt(np.mean(e**2))), "std": float(np.std(err)), "median_abs": float(np.median(e)), "gross_rate": float(np.mean(e > GROSS)), "by_snr": {}}
    out["by_mod"] = {}
    for mi in sorted(set(b["modulation"].tolist())):
        sel = (b["modulation"] == mi) & (b["snr_db"] >= 10)
        if sel.any():
            out["by_mod"][MODULATIONS[mi]] = {"rmse": float(np.sqrt(np.mean(e[sel] ** 2))), "median_abs": float(np.median(e[sel])), "gross_rate": float(np.mean(e[sel] > GROSS))}
    if fs:
        out["std_hz"], out["median_abs_hz"] = out["std"] * fs, out["median_abs"] * fs
    for s in sorted(set(b["snr_db"].tolist())):
        sel = b["snr_db"] == s
        out["by_snr"][str(int(s))] = {"mse": float(np.mean(e[sel] ** 2)), "rmse": float(np.sqrt(np.mean(e[sel] ** 2))), "std": float(np.std(err[sel])), "median_abs": float(np.median(e[sel])), "gross_rate": float(np.mean(e[sel] > GROSS))}
    return out


def score(methods, b, fs=None):
    res = {}
    for name, fn in methods.items():
        est = fn(b)
        if est is None:
            res[name] = None
            continue
        res[name] = summarise(wrap(est - b["cfo"]), b, fs)
    return res


def repo_native(spec, count, spec2=None, spec_abl=None):
    """The repo's own recipe signals, scored with the README's 250 Hz agreement tolerance."""
    from dual_parameter_estimator import DualParameterEstimator, dsp_estimate, truth_from_audit
    from generate_synthetic import generate_batch, load_recipes
    recipes = load_recipes(ROOT / "data" / "recipes" / "mvp-recipes.json")[:2]
    legacy = DualParameterEstimator.train_from_recipes(recipes + [], examples=96)
    batch = generate_batch(recipes, count, 91919)
    rows = {"legacy_dsp": [], "legacy_tinymlp": [], "speccfo": [], "speccfo_v2_nocoords": [], "speccfo_v2": [], "dc_dsp": [], "dc_tinymlp": [], "truth_cfo": [], "truth_dc": []}
    for samples, audit in batch:
        fs = float(audit["sample_rate_hz"])
        truth = truth_from_audit(audit)
        d, l = dsp_estimate(samples, fs), legacy.learned_estimate(samples, fs)
        rows["legacy_dsp"].append(d["carrier_offset_hz"])
        rows["legacy_tinymlp"].append(l["carrier_offset_hz"])
        if spec is not None:
            rows["speccfo"].append(float(spec.estimate(np.asarray(samples)[None, :])["cfo"][0]) * fs)
        if spec_abl is not None:
            rows["speccfo_v2_nocoords"].append(float(spec_abl.estimate(np.asarray(samples)[None, :])["cfo"][0]) * fs)
        if spec2 is not None:
            rows["speccfo_v2"].append(float(spec2.estimate(np.asarray(samples)[None, :])["cfo"][0]) * fs)
        rows["dc_dsp"].append(np.hypot(d["dc_i"] - truth[0], d["dc_q"] - truth[1]))
        rows["dc_tinymlp"].append(np.hypot(l["dc_i"] - truth[0], l["dc_q"] - truth[1]))
        rows["truth_cfo"].append(truth[2])
    truth = np.array(rows["truth_cfo"])
    out = {"n": len(truth), "tolerance_hz": 250.0, "sample_rate_hz": 1e6, "carrier_offset": {}, "dc": {}}
    for k in ("legacy_dsp", "legacy_tinymlp", "speccfo", "speccfo_v2_nocoords", "speccfo_v2"):
        if rows[k]:
            e = np.abs(np.array(rows[k]) - truth)
            out["carrier_offset"][k] = {"rmse_hz": float(np.sqrt(np.mean(e**2))), "median_abs_hz": float(np.median(e)), "within_250hz": float(np.mean(e <= 250.0)), "within_25hz": float(np.mean(e <= 25.0))}
    for k in ("dc_dsp", "dc_tinymlp"):
        e = np.array(rows[k])
        out["dc"][k.replace("dc_", "")] = {"rmse": float(np.sqrt(np.mean(e**2))), "within_0.05": float(np.mean(e <= 0.05)), "within_0.005": float(np.mean(e <= 0.005))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sets for a smoke test")
    args = ap.parse_args()
    q = 0.25 if args.quick else 1.0
    n = lambda v: max(2, int(v * q))
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = ROOT / "data" / "models" / "speccfo.npz"
    spec = fast_spec(SpecCFO(spec_path)) if spec_path.exists() else None
    methods = method_table(spec)
    results = {"protocol": {"length": 1024, "snrs": SNRS, "mods": MODS, "gross_threshold_cycles_per_sample": GROSS, "cfo_range": [-0.2, 0.2], "quick": args.quick, "net_length": NET_LENGTH}, "sets": {}}

    def record(name, desc, b, fs=None):
        results["sets"][name] = {"description": desc, "count": int(len(b["x"])), "methods": score(methods, b, fs)}
        print(name, {k: (round(v["rmse"], 5) if v else None) for k, v in results["sets"][name]["methods"].items()}, flush=True)

    record("awgn", "All 11 modulations, SNR sweep, CFO +-0.2, L=1024 (Chen 2023 Fig. 3)", build(np.random.default_rng(101), MODS, SNRS, n(40)))
    # Chen et al. write "normalised frequency offset -0.2 to 0.2" without a reference. Cycles per sample (above) is the
    # harder reading. If it means cycles per symbol, the range at 8 samples/symbol is +-0.025 cycles/sample:
    record("awgn_narrow", "All modulations at 8 samples/symbol, CFO +-0.025 cycles/sample (= +-0.2 cycles/symbol), the easier reading of Chen 2023", build(np.random.default_rng(111), MODS, SNRS, n(40), cfo=(-0.025, 0.025), sps=8))
    record("rayleigh", "Same with exponential-profile Rayleigh multipath, spread 0.5-2 samples", build(np.random.default_rng(202), MODS, SNRS, n(20), rayleigh=1.0))
    over = {}
    for sps in (4, 8, 16):
        b = build(np.random.default_rng(300 + sps), ["bpsk"], SNRS, n(60), sps=sps)
        over[sps] = {"description": f"BPSK, {sps} samples/symbol", "methods": score(methods, b)}
    results["sets"]["bpsk_oversampling"] = {"description": "BPSK at 4, 8, 16 samples/symbol, L=1024 (Chen 2023 Fig. 5)", "runs": over}
    lens = {}
    for L in (512, 1024, 2048):
        b = build(np.random.default_rng(400 + L), ["bpsk"], SNRS, n(60), length=L, sps=8)
        lens[L] = {"description": f"BPSK, L={L}", "methods": score(methods, b)}
    results["sets"]["bpsk_length"] = {"description": "BPSK at 512, 1024, 2048 samples, 8 samples/symbol (Chen 2023 Fig. 6)", "runs": lens}
    # O'Shea 2017: 400 kHz sampling, 100 kHz symbols, CFO +-50 kHz, SNR 0/5/10, AWGN + Rayleigh sigma 0.5/1/2.
    osh = {}
    for chan, spread in (("awgn", None), ("sigma0.5", 0.5), ("sigma1", 1.0), ("sigma2", 2.0)):
        for L in (64, 128, 256, 512, 1024):
            fixed = {"sps": 4, "rolloff": 0.25}
            if spread:
                fixed["delay_spread"] = spread
            b = build(np.random.default_rng(500 + L), ["qpsk"], [0, 5, 10], n(150), length=L, cfo=(-0.125, 0.125), rayleigh=1.0 if spread else 0.0, **fixed)
            osh[f"{chan}/L{L}"] = {"channel": chan, "length": L, "methods": score({k: v for k, v in methods.items() if k in ("periodogram_genie_M", "blind_multi_M", "kay_genie_M", "iq_resnet", "oshea_cfo", "speccfo_coarse", "speccfo", "speccfo_v2")}, b, fs=400_000.0)}
    results["sets"]["oshea_channel"] = {"description": "QPSK RRC 0.25, 4 sps, 400 kHz, +-50 kHz, SNR 0/5/10 dB (O'Shea 2017 Figs. 4-7). std_hz is error standard deviation at 400 kHz.", "runs": osh}
    spec2 = fast_spec(SpecCFO(SPEC_V2)) if SPEC_V2.exists() else None
    spec_abl = fast_spec(SpecCFO(SPEC_V2_NOCOORDS)) if SPEC_V2_NOCOORDS.exists() else None
    results["sets"]["repo_native"] = repo_native(spec, n(120) if not args.quick else 24, spec2, spec_abl)
    print("repo_native", json.dumps(results["sets"]["repo_native"]["carrier_offset"]), flush=True)
    results["tone_crb_std_cycles"] = {str(s): float(tone_crb(s, 1024)) for s in SNRS}
    (OUT / "cfo_results.json").write_text(json.dumps(results, indent=1))
    print("wrote", OUT / "cfo_results.json")


if __name__ == "__main__":
    main()
