"""Train AMC models on freshly simulated captures.

  python research/train_amc.py --model demod_amc --steps 3000

Baselines take 128-sample windows as in their papers (windows_per_capture
random windows from each 1024-sample capture). The proposed model takes the
whole capture. All models see the same simulator distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

import jax_models as jm  # noqa: E402
from ckpt_util import load_resume, save_resume  # noqa: E402
from amc_models import ap_windows, derotate, iq_windows, time_features  # noqa: E402
from cfo_estimators import SpecCFO, spec_features  # noqa: E402
from signal_sim import MODULATIONS, SimConfig, simulate  # noqa: E402

NCLASS = len(MODULATIONS)
TRAIN_CFG = SimConfig(length=1024, modulations=MODULATIONS, snr_db=(-10, 30), cfo=(-0.05, 0.05), sps=(2, 16), rayleigh_probability=0.3, hardware_probability=0.2)
AUG_CFG = SimConfig(length=1024, modulations=MODULATIONS, snr_db=(-10, 30), cfo=(-0.05, 0.05), sps=(1, 16), rayleigh_probability=0.3, hardware_probability=0.2,
                    pulses=("rrc", "rrc", "rect", "rc", "halfsine", "gauss"), phase_noise_probability=0.3, interferer_probability=0.3, impulse_probability=0.15, dc_probability=0.3)

MODELS = {
    "vtcnn2": (lambda k: jm.vtcnn2_init(k, NCLASS), jm.vtcnn2_apply, "iq"),
    "lstm_ap": (lambda k: jm.lstm_ap_init(k, NCLASS), jm.lstm_ap_apply, "ap"),
    "demod_amc": (lambda k: jm.demod_amc_init(k, NCLASS), jm.demod_amc_apply, "multi"),
    "demod_amc_nocomp": (lambda k: jm.demod_amc_init(k, NCLASS), jm.demod_amc_apply, "multi_raw"),
}


def make_inputs(kind: str, x: np.ndarray, rng, cfo_model: SpecCFO | None, windows_per_capture: int):
    if kind in ("iq", "ap"):
        w = iq_windows(x) if kind == "iq" else ap_windows(x)
        pick = np.stack([rng.choice(w.shape[1], windows_per_capture, replace=False) for _ in range(w.shape[0])])
        return w[np.arange(w.shape[0])[:, None], pick].reshape(-1, *w.shape[2:])
    if kind == "multi":
        x = derotate(x, cfo_model.estimate(x)["cfo"])
    return time_features(x), spec_features(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--windows-per-capture", type=int, default=2)
    ap.add_argument("--seed", type=int, default=26147)
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "checkpoints")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--augment", action="store_true", help="v2: robustness augmentation as in train_cfo.py")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _v2")
    ap.add_argument("--cfo-model", type=Path, default=ROOT / "data" / "models" / "speccfo.npz")
    args = ap.parse_args()
    name = args.model + args.tag
    train_cfg = AUG_CFG if args.augment else TRAIN_CFG
    init, apply, kind = MODELS[args.model]
    cfo_model = SpecCFO(args.cfo_model) if kind == "multi" else None
    if cfo_model is not None:  # same network, JIT-compiled: the numpy path costs 0.5 s per batch
        sp = {k: jnp.asarray(v) for k, v in cfo_model.params.items()}
        fast = jax.jit(lambda f: jm.speccfo_apply(sp, {}, f, False)[0])
        cfo_model.logits_override = lambda f: np.asarray(fast(f))
    params, state = init(jax.random.PRNGKey(args.seed))
    lr = 1e-3 if kind in ("iq", "ap") else 2e-3  # O'Shea 2016 / Rajendran 2018 used Adam at about 1e-3
    sched = optax.warmup_cosine_decay_schedule(lr / 10, lr, args.steps // 20, args.steps, lr / 100)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = opt.init(params)

    def loss_fn(p, s, xb, yb, key):
        logits, s2 = apply(p, s, xb, True, key)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, yb)), s2

    @jax.jit
    def step(p, s, o, xb, yb, key):
        (loss, s2), g = jax.value_and_grad(loss_fn, has_aux=True)(p, s, xb, yb, key)
        upd, o = opt.update(g, o, p)
        return optax.apply_updates(p, upd), s2, o, loss

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed + 1)
    log, running, t0, first = [], [], time.time(), 1
    resume_path = args.out / f"{name}.resume.pkl"
    saved = load_resume(resume_path)
    if saved and saved["step"] < args.steps:
        params, state, opt_state = jax.tree_util.tree_map(jnp.asarray, saved["params"]), jax.tree_util.tree_map(jnp.asarray, saved["state"]), jax.tree_util.tree_map(jnp.asarray, saved["opt_state"])
        log, first = saved["log"], saved["step"] + 1
        rng.bit_generator.state = saved["rng_state"]
        print(f"resumed {args.model} at step {saved['step']}", flush=True)
    print("devices:", jax.devices(), flush=True)
    for i in range(first, args.steps + 1):
        b = simulate(rng, args.batch, train_cfg)
        xb = make_inputs(kind, b["x"], rng, cfo_model, args.windows_per_capture)
        yb = b["modulation"] if kind.startswith("multi") else np.repeat(b["modulation"], args.windows_per_capture)
        key, sub = jax.random.split(key)
        params, state, opt_state, loss = step(params, state, opt_state, xb, yb, sub)
        running.append(float(loss))
        if i % 100 == 0 or i == args.steps:
            entry = {"step": i, "loss": float(np.mean(running)), "seconds": round(time.time() - t0, 1)}
            log.append(entry)
            running = []
            print(json.dumps(entry), flush=True)
        if i % args.save_every == 0 and i < args.steps:
            save_resume(resume_path, i, params, state, opt_state, log, rng.bit_generator.state)
    resume_path.unlink(missing_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    flat = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path({"params": params, "state": state})[0]:
        flat["/".join(str(getattr(k, "key", k)) for k in path)] = np.asarray(leaf)
    meta = {"model": args.model, "tag": args.tag, "augment": args.augment, "cfo_model": str(args.cfo_model.name), "steps": args.steps, "batch": args.batch, "windows_per_capture": args.windows_per_capture, "seed": args.seed, "kind": kind, "classes": list(MODULATIONS), "train_config": train_cfg.__dict__, "log": log, "cfo_compensation": kind == "multi"}
    np.savez(args.out / f"{name}.npz", meta=json.dumps(meta, default=str), **flat)
    print("saved", args.out / f"{name}.npz")


if __name__ == "__main__":
    main()
