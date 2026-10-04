"""Resumable training checkpoints: params, state, optimiser state and step in one pickle."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import numpy as np


def _np(tree):
    return jax.tree_util.tree_map(lambda v: np.asarray(v), tree)


def save_resume(path: Path, step: int, params, state, opt_state, log, rng_state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"step": step, "params": _np(params), "state": _np(state), "opt_state": _np(opt_state), "log": log, "rng_state": rng_state}, fh)
    tmp.replace(path)


def load_resume(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)
