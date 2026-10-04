"""Train CFO estimators on freshly simulated batches (no stored dataset).

Every neural model sees the same simulator distribution and batch size. The
step budget is set per run and recorded in the checkpoint metadata.

  python research/train_cfo.py --model speccfo --steps 3000
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
from cfo_estimators import SPEC_BINS, spec_features, wrap  # noqa: E402
from signal_sim import MODULATIONS, SimConfig, simulate  # noqa: E402

CFO_SCALE = 0.2  # regression targets are cfo / 0.2, the range used by Chen et al. (2023)
TRAIN_CFG = SimConfig(length=1024, modulations=tuple(m for m in MODULATIONS if m != "noise"), snr_db=(-10, 30), cfo=(-0.2, 0.2), sps=(2, 16), rayleigh_probability=0.3, hardware_probability=0.2)

AUG_CFG = SimConfig(length=1024, modulations=TRAIN_CFG.modulations, snr_db=(-10, 30), cfo=(-0.2, 0.2), sps=(1, 16), rayleigh_probability=0.3, hardware_probability=0.2,
                    pulses=("rrc", "rrc", "rect", "rc", "halfsine", "gauss"), phase_noise_probability=0.3, interferer_probability=0.3, impulse_probability=0.15, dc_probability=0.3)
AUG_LENGTHS = (1024, 2048, 4096)  # the API estimates on whole bursts, so train on longer captures too

MODELS = {
    "iq_resnet": (jm.iq_resnet_init, jm.iq_resnet_apply, "regression"),
    "oshea_cfo": (jm.oshea_cfo_init, jm.oshea_cfo_apply, "regression"),
    "speccfo": (jm.speccfo_init, jm.speccfo_apply, "bins"),
}


def iq_tensor(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True)) + 1e-12)
    return np.stack([x.real, x.imag], axis=1).astype(np.float32)


def bin_targets(cfo: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    pos = wrap(cfo) * SPEC_BINS + SPEC_BINS // 2
    k = np.arange(SPEC_BINS)[None, :]
    d = (k - pos[:, None] + SPEC_BINS / 2) % SPEC_BINS - SPEC_BINS / 2
    t = np.exp(-0.5 * (d / sigma) ** 2)
    return (t / t.sum(axis=1, keepdims=True)).astype(np.float32)


def batch_for(kind: str, rng, n, augment: bool = False):
    if augment:
        import dataclasses
        b = simulate(rng, n, dataclasses.replace(AUG_CFG, length=int(AUG_LENGTHS[int(rng.integers(0, len(AUG_LENGTHS)))])))
    else:
        b = simulate(rng, n, TRAIN_CFG)
    if kind == "bins":
        return spec_features(b["x"]), bin_targets(b["cfo"])
    return iq_tensor(b["x"]), (b["cfo"] / CFO_SCALE).astype(np.float32)


def schedule_for(name: str, steps: int):
    if name == "iq_resnet":  # paper: lr 0.02, x0.1 at epochs 5 and 10 of 20
        return optax.piecewise_constant_schedule(0.02, {int(steps * 0.25): 0.1, int(steps * 0.5): 0.1})
    if name == "oshea_cfo":  # paper: 1e-3, halved on plateau. Halved every 20% here.
        return optax.piecewise_constant_schedule(1e-3, {int(steps * f): 0.5 for f in (0.2, 0.4, 0.6, 0.8)})
    return optax.warmup_cosine_decay_schedule(1e-4, 2e-3, steps // 20, steps, 2e-5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=26147)
    ap.add_argument("--lr-override", type=float, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "checkpoints")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--augment", action="store_true", help="v2: more pulse shapes, interferers, phase noise, impulses, longer captures (speccfo only)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _v2")
    ap.add_argument("--coords", action="store_true", help="speccfo only: add absolute-position input channels")
    args = ap.parse_args()
    if args.augment and args.model != "speccfo":
        raise SystemExit("--augment is only defined for speccfo (the other networks have a fixed 1024-sample input)")
    name = args.model + args.tag
    init, apply, kind = MODELS[args.model]
    params, state = jm.speccfo_init(jax.random.PRNGKey(args.seed), coords=True) if args.coords else init(jax.random.PRNGKey(args.seed))
    sched = schedule_for(args.model, args.steps) if args.lr_override is None else optax.constant_schedule(args.lr_override)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    opt_state = opt.init(params)

    def loss_fn(p, s, xb, yb):
        out, s2 = apply(p, s, xb, True)
        if kind == "bins":
            return -jnp.mean(jnp.sum(yb * jax.nn.log_softmax(out, axis=1), axis=1)), s2
        return jnp.mean((out - yb) ** 2), s2

    @jax.jit
    def step(p, s, o, xb, yb):
        (loss, s2), g = jax.value_and_grad(loss_fn, has_aux=True)(p, s, xb, yb)
        upd, o = opt.update(g, o, p)
        return optax.apply_updates(p, upd), s2, o, loss

    rng = np.random.default_rng(args.seed)
    log, t0, running, first = [], time.time(), [], 1
    resume_path = args.out / f"{name}.resume.pkl"
    saved = load_resume(resume_path)
    if saved and saved["step"] < args.steps:
        params, state, opt_state = jax.tree_util.tree_map(jnp.asarray, saved["params"]), jax.tree_util.tree_map(jnp.asarray, saved["state"]), jax.tree_util.tree_map(jnp.asarray, saved["opt_state"])
        log, first = saved["log"], saved["step"] + 1
        rng.bit_generator.state = saved["rng_state"]
        print(f"resumed {args.model} at step {saved['step']}", flush=True)
    print("devices:", jax.devices(), flush=True)
    for i in range(first, args.steps + 1):
        xb, yb = batch_for(kind, rng, args.batch, args.augment)
        params, state, opt_state, loss = step(params, state, opt_state, xb, yb)
        running.append(float(loss))
        if i % 100 == 0 or i == args.steps:
            entry = {"step": i, "loss": float(np.mean(running)), "seconds": round(time.time() - t0, 1)}
            log.append(entry)
            running = []
            print(json.dumps(entry), flush=True)
            if not np.isfinite(entry["loss"]):
                raise SystemExit("loss diverged")
        if i % args.save_every == 0 and i < args.steps:
            save_resume(resume_path, i, params, state, opt_state, log, rng.bit_generator.state)
    args.out.mkdir(parents=True, exist_ok=True)
    flat = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path({"params": params, "state": state})[0]:
        flat["/".join(str(getattr(k, "key", k)) for k in path)] = np.asarray(leaf)
    meta = {"model": args.model, "tag": args.tag, "augment": args.augment, "coords": args.coords, "steps": args.steps, "batch": args.batch, "seed": args.seed, "kind": kind, "cfo_scale": CFO_SCALE, "train_config": TRAIN_CFG.__dict__, "log": log, "lr_override": args.lr_override}
    np.savez(args.out / f"{name}.npz", meta=json.dumps(meta, default=str), **flat)
    resume_path.unlink(missing_ok=True)
    print("saved", args.out / f"{name}.npz")


if __name__ == "__main__":
    main()
