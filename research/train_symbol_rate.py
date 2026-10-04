"""Train a small feature-based SPS baseline; persist recipes, weights and metrics only."""
import sys
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from signal_sim import simulate, SimConfig, LINEAR
from rate_estimation import features, CLASSES, MODEL_PATH


def corpus(seed, per_class):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for label, sps in enumerate(CLASSES):
        batch = simulate(rng, per_class, SimConfig(length=2048, modulations=LINEAR, snr_db=(5, 30),
                          pulses=('rrc', 'rect'), fixed={'sps': int(sps)}, hardware_probability=.3))
        X.extend(features(x) for x in batch['x']); Y.extend([label] * per_class)
    return np.asarray(X), np.asarray(Y)


def main():
    X, y = corpus(26147, 250)
    mean, scale = X.mean(0), X.std(0) + 1e-4
    X = (X - mean) / scale
    w = np.zeros((X.shape[1], len(CLASSES))); b = np.zeros(len(CLASSES))
    target = np.eye(len(CLASSES))[y]
    for _ in range(600):
        z = X @ w + b; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        d = (p - target) / len(y)
        w -= .08 * (X.T @ d + .001 * w); b -= .08 * d.sum(0)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    np.savez(MODEL_PATH, mean=mean, scale=scale, weights=w, bias=b)
    E, truth = corpus(826147, 100)
    z = ((E - mean)/scale) @ w + b; p = np.exp(z-z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    pred = p.argmax(1); accepted = p.max(1) >= .8
    confusion = np.zeros((4, 4), dtype=int); np.add.at(confusion, (truth, pred), 1)
    report = {'training_seed': 26147, 'evaluation_seed': 826147, 'training_examples': len(y), 'test_examples': len(truth),
              'classes_sps': CLASSES.tolist(), 'accuracy': float((pred == truth).mean()),
              'coverage': float(accepted.mean()), 'accepted_accuracy': float((pred[accepted] == truth[accepted]).mean()) if accepted.any() else None,
              'confusion': confusion.tolist(), 'scope': 'Independent synthetic seeds, same generator; PSK/QAM, RRC/rect, 5–30 dB; not real-data validation.',
              'recipe': {'length': 2048, 'modulations': list(LINEAR), 'pulses': ['rrc', 'rect'], 'snr_db': [5,30], 'sps': [2,4,8,16]}}
    (ROOT / 'research/results/symbol_rate_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
