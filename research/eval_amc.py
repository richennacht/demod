"""Evaluate every AMC model on identical fixed-seed test sets, and calibrate DemodAMC.

  python research/eval_amc.py

Decisions are per 1024-sample capture for every model. Window-based paper
models (128 samples) average log-probabilities over the capture's 8 windows,
so they get the same information as the full-capture model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from amc_models import NearestCentroid, ap_windows, centroid_features, cumulant_classify, derotate, iq_windows, time_features  # noqa: E402
from cfo_estimators import SpecCFO, spec_features  # noqa: E402
from signal_sim import LINEAR, MODULATIONS, SimConfig, simulate  # noqa: E402

OUT = ROOT / "research" / "results"
CKPT = ROOT / "research" / "checkpoints"
SNRS = [-10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30]
LIN = np.array([MODULATIONS.index(m) for m in LINEAR])
REPO4 = np.array([MODULATIONS.index(m) for m in ("bpsk", "qpsk", "8psk", "16qam")])


def build_set(per_cell, seed, cfo=(-0.05, 0.05), snrs=SNRS, mods=MODULATIONS):
    rng = np.random.default_rng(seed)
    parts = []
    for mod in mods:
        for snr in snrs:
            parts.append(simulate(rng, per_cell, SimConfig(length=1024, cfo=cfo, rayleigh_probability=0.3, hardware_probability=0.2, fixed={"modulation": mod, "snr_db": snr})))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load(name):
    import jax
    from train_amc import MODELS
    data = np.load(CKPT / f"{name}.npz", allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    init, apply, kind = MODELS[name.removesuffix("_v2")]
    params, state = init(jax.random.PRNGKey(0))
    leaves, treedef = jax.tree_util.tree_flatten_with_path({"params": params, "state": state})
    tree = jax.tree_util.tree_unflatten(treedef, [data["/".join(str(getattr(k, "key", k)) for k in p)] for p, _ in leaves])
    fn = jax.jit(lambda xb: apply(tree["params"], tree["state"], xb, False)[0])
    return fn, kind, meta


def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def model_logprobs(name, x, spec):
    fn, kind, _ = load(name)
    out = []
    for i in range(0, len(x), 128):
        xb = x[i:i + 128]
        if kind in ("iq", "ap"):
            w = iq_windows(xb) if kind == "iq" else ap_windows(xb)
            z = np.asarray(fn(w.reshape(-1, *w.shape[2:]))).reshape(w.shape[0], w.shape[1], -1)
            out.append(log_softmax(log_softmax(z).mean(axis=1)))
        else:
            if kind == "multi":
                xb = derotate(xb, spec.estimate(xb)["cfo"])
            out.append(log_softmax(np.asarray(fn((time_features(xb), spec_features(xb))))))
    return np.concatenate(out)


def ece(prob, y, bins=15):
    conf, pred = prob.max(axis=1), prob.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs(conf[sel].mean() - (pred[sel] == y[sel]).mean())
    return float(total)


def metrics(pred, y, snr, prob=None):
    out = {"accuracy": float(np.mean(pred == y)), "accuracy_snr_ge_0": float(np.mean((pred == y)[snr >= 0])), "by_snr": {str(int(s)): float(np.mean((pred == y)[snr == s])) for s in sorted(set(snr.tolist()))}}
    f1 = []
    for c in np.unique(y):
        tp = np.sum((pred == c) & (y == c))
        p_, r_ = tp / max(1, np.sum(pred == c)), tp / max(1, np.sum(y == c))
        f1.append(0 if tp == 0 else 2 * p_ * r_ / (p_ + r_))
    out["macro_f1"] = float(np.mean(f1))
    if prob is not None:
        out["ece"] = ece(prob, y)
    return out


def restrict(logp, allowed):
    z = np.full_like(logp, -np.inf)
    z[:, allowed] = logp[:, allowed]
    return z.argmax(axis=1)


def fit_temperature(logp, y):
    grid = np.exp(np.linspace(np.log(0.3), np.log(5), 80))
    nll = [-np.mean(log_softmax(logp / t)[np.arange(len(y)), y]) for t in grid]
    return float(grid[int(np.argmin(nll))])


def write_calibration(name, cal):
    """Store temperature and abstention threshold in the shipped model so inference uses them."""
    data = dict(np.load(CKPT / f"{name}.npz", allow_pickle=False))
    meta = json.loads(str(data["meta"]))
    meta.update(temperature=cal["temperature"], abstain_below=cal["abstain_below"], calibrated_on="validation set disjoint from the test sets, seed 3003")
    data["meta"] = json.dumps(meta)
    dst = ROOT / "data" / "models"
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / f"{name}.npz", **data)


def repo_native(models, spec_for, count):
    """The repo's own recipe signals (4 classes, 1 MS/s, 4096 samples) including the shipped centroid classifier."""
    from generate_synthetic import generate_batch, load_recipes
    from modulation_classifier import CentroidAMC
    recipes = load_recipes(ROOT / "data" / "recipes" / "mvp-recipes.json")[:2]
    shipped = CentroidAMC.train(recipes, examples=96)
    batch = generate_batch(recipes, count, 91919)
    x = np.stack([np.asarray(sig, dtype=np.complex128) for sig, _ in batch])
    y = np.array([MODULATIONS.index(a["modulation"]) for _, a in batch])
    snr = np.array([a["impairments"]["snr_db"] for _, a in batch])
    stage = np.array([a["recipe_id"] for _, a in batch])
    out = {"n": int(len(y)), "classes": ["bpsk", "qpsk", "8psk", "16qam"], "models": {}}
    preds = {"centroid_shipped_repo4": [], "accept": []}
    for sig, _ in batch:
        p = shipped.predict(list(sig))
        preds["centroid_shipped_repo4"].append(MODULATIONS.index(p["ranked_candidates"][0]["modulation"]))
        preds["accept"].append(not p["abstained"])
    pred, accept = np.array(preds["centroid_shipped_repo4"]), np.array(preds["accept"])
    def summary(p, acc=None):
        r = {"accuracy": float(np.mean(p == y)), "accuracy_clean_stage": float(np.mean((p == y)[stage == "stage-0-clean-linear"])), "accuracy_impaired_stage": float(np.mean((p == y)[stage == "stage-1-rf-impairments"]))}
        if acc is not None and acc.any():
            r["selective"] = {"coverage": float(acc.mean()), "accuracy_on_accepted": float(np.mean((p == y)[acc]))}
        return r
    out["models"]["centroid_shipped_repo4"] = summary(pred, accept)
    for name in models:
        logp = model_logprobs(name, x, spec_for(name))
        out["models"][name] = summary(restrict(logp, REPO4), None)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from eval_cfo import fast_spec
    spec = fast_spec(SpecCFO(ROOT / "data" / "models" / "speccfo.npz"))
    v2_path = ROOT / "data" / "models" / "speccfo_v2.npz"
    spec2 = fast_spec(SpecCFO(v2_path)) if v2_path.exists() else None
    spec_for = lambda n: spec2 if n.endswith("_v2") else spec
    test = build_set(30, 1001)
    shift = build_set(15, 2002, cfo=(-0.2, 0.2))
    val = build_set(12, 3003)
    train_c = simulate(np.random.default_rng(4004), 12000, SimConfig(length=1024, cfo=(-0.05, 0.05), rayleigh_probability=0.3, hardware_probability=0.2))
    results = {"protocol": {"capture_length": 1024, "snrs": SNRS, "classes": list(MODULATIONS), "test_per_cell": 30, "shift_per_cell": 15, "train_cfo": [-0.05, 0.05], "shift_cfo": [-0.2, 0.2]}, "sets": {"in_distribution": {}, "cfo_shift": {}}}
    nc = NearestCentroid().fit(centroid_features(train_c["x"]), train_c["modulation"])
    from generate_synthetic import load_recipes
    from modulation_classifier import CentroidAMC
    shipped = CentroidAMC.train(load_recipes(ROOT / "data" / "recipes" / "mvp-recipes.json"), examples=96)

    for set_name, b in (("in_distribution", test), ("cfo_shift", shift)):
        y, snr = b["modulation"], b["snr_db"]
        r = results["sets"][set_name]
        lin = np.isin(y, LIN)
        r["centroid_retrained"] = metrics(nc.predict(centroid_features(b["x"])), y, snr)
        cum = cumulant_classify(b["x"][lin])
        r["cumulants_linear5"] = metrics(cum, y[lin], snr[lin])
        # Cumulants assume a synchronised carrier. Give them the same SpecCFO derotation DemodAMC gets, so the baseline is not a strawman.
        xl = b["x"][lin]
        derot = np.concatenate([derotate(xl[i:i + 256], spec.estimate(xl[i:i + 256])["cfo"]) for i in range(0, len(xl), 256)])
        r["cumulants_derotated"] = metrics(cumulant_classify(derot), y[lin], snr[lin])
        sel4 = np.isin(y, REPO4)
        names4 = ("bpsk", "qpsk", "8psk", "16qam")
        shipped_pred, shipped_accept = [], []
        for row in b["x"][sel4]:
            pred = shipped.predict(list(row))
            shipped_pred.append(MODULATIONS.index(pred["ranked_candidates"][0]["modulation"]))
            shipped_accept.append(not pred["abstained"])
        shipped_pred, shipped_accept = np.array(shipped_pred), np.array(shipped_accept)
        r["centroid_shipped_repo4"] = metrics(shipped_pred, y[sel4], snr[sel4])
        r["centroid_shipped_repo4"]["selective"] = {"coverage": float(shipped_accept.mean()), "accuracy_on_accepted": float(np.mean((shipped_pred == y[sel4])[shipped_accept])) if shipped_accept.any() else None}
        for name in ("vtcnn2", "lstm_ap", "demod_amc_nocomp", "demod_amc", "demod_amc_v2"):
            if not (CKPT / f"{name}.npz").exists():
                continue
            logp = model_logprobs(name, b["x"], spec_for(name))
            m = metrics(logp.argmax(axis=1), y, snr, np.exp(logp))
            m["linear5"] = metrics(restrict(logp[lin], LIN), y[lin], snr[lin])
            m["repo4"] = metrics(restrict(logp[sel4], REPO4), y[sel4], snr[sel4])
            if name.startswith("demod_amc") and name != "demod_amc_nocomp" and set_name == "in_distribution":
                vlog = model_logprobs(name, val["x"], spec_for(name))
                t = fit_temperature(vlog, val["modulation"])
                vprob = np.exp(log_softmax(vlog / t))
                conf, ok = vprob.max(axis=1), vprob.argmax(axis=1) == val["modulation"]
                thr = 0.0
                for cand in np.linspace(0.3, 0.99, 70):
                    if ok[conf >= cand].mean() >= 0.95:
                        thr = float(cand)
                        break
                results.setdefault("calibration", {})[name] = {"temperature": t, "abstain_below": thr, "target_accepted_accuracy": 0.95}
                write_calibration(name, results["calibration"][name])
            if name in results.get("calibration", {}):
                cal = results["calibration"][name]
                prob = np.exp(log_softmax(logp / cal["temperature"]))
                m["ece_calibrated"] = ece(prob, y)
                accept = prob.max(axis=1) >= cal["abstain_below"]
                m["selective"] = {"coverage": float(accept.mean()), "accuracy_on_accepted": float(np.mean((prob.argmax(axis=1) == y)[accept]))}
                cm = np.zeros((len(MODULATIONS), len(MODULATIONS)), dtype=int)
                for a, p_ in zip(y, logp.argmax(axis=1)):
                    cm[a, p_] += 1
                m["confusion"] = cm.tolist()
                m["per_class_recall"] = {MODULATIONS[i]: float(cm[i, i] / max(1, cm[i].sum())) for i in range(len(MODULATIONS))}
            r[name] = m
            print(set_name, name, round(m["accuracy"], 4), round(m["accuracy_snr_ge_0"], 4), flush=True)
        print(set_name, "centroid_retrained", round(r["centroid_retrained"]["accuracy"], 4), "cumulants", round(r["cumulants_linear5"]["accuracy"], 4), "shipped", round(r["centroid_shipped_repo4"]["accuracy"], 4), flush=True)
    names = [n for n in ("vtcnn2", "lstm_ap", "demod_amc_nocomp", "demod_amc", "demod_amc_v2") if (CKPT / f"{n}.npz").exists()]
    results["sets"]["repo_native"] = repo_native(names, spec_for, 120)
    print("repo_native", {k: round(v["accuracy"], 3) for k, v in results["sets"]["repo_native"]["models"].items()}, flush=True)
    (OUT / "amc_results.json").write_text(json.dumps(results, indent=1))
    print("wrote", OUT / "amc_results.json")


if __name__ == "__main__":
    main()
