"""Pick SpecCFO's low-confidence fallback threshold on a validation set disjoint from every test set.

  python research/tune_cfo_fallback.py data/models/speccfo_v2.npz

Below the threshold the circular posterior mean replaces the peak. That is the
MSE-optimal answer when the network is unsure, and it removes large wrong-peak
errors at low SNR. The threshold minimising validation RMSE is written to the
model's metadata. Median error is reported next to it to show the high-SNR
precision is not traded away.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from cfo_estimators import SpecCFO, wrap  # noqa: E402
from eval_cfo import fast_spec  # noqa: E402
from signal_sim import MODULATIONS, SimConfig, simulate  # noqa: E402
from train_cfo import AUG_CFG, TRAIN_CFG  # noqa: E402

GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def main(path: Path) -> None:
    rng = np.random.default_rng(777)
    mods = [m for m in MODULATIONS if m != "noise"]
    parts = []
    for cfg in (TRAIN_CFG, dataclasses.replace(AUG_CFG, length=1024)):  # half paper-style, half augmented
        for mod in mods:
            for snr in (-10, -5, 0, 5, 10, 15, 20, 25, 30):
                parts.append(simulate(rng, 8, dataclasses.replace(cfg, fixed={"modulation": mod, "snr_db": snr})))
    x = np.concatenate([p["x"] for p in parts])
    cfo = np.concatenate([p["cfo"] for p in parts])
    spec = fast_spec(SpecCFO(path))
    rows = []
    for t in GRID:
        est = np.concatenate([spec.estimate(x[i:i + 256], fallback_below=t)["cfo"] for i in range(0, len(x), 256)])
        e = np.abs(wrap(est - cfo))
        rows.append({"fallback_below": t, "rmse": float(np.sqrt(np.mean(e**2))), "median_abs": float(np.median(e)), "gross_rate": float(np.mean(e > 0.005))})
        print(rows[-1], flush=True)
    best = min(rows, key=lambda r: r["rmse"])
    data = dict(np.load(path, allow_pickle=False))
    meta = json.loads(str(data["meta"]))
    meta.update(fallback_below=best["fallback_below"], fallback_tuned_on="validation set, seed 777, half paper-style half augmented, disjoint from the test sets", fallback_sweep=rows)
    data["meta"] = json.dumps(meta)
    np.savez(path, **data)
    print("chose", best)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
