"""Automatic modulation classification (AMC) for DEmod.

Feature extraction here is shared by training (research/train_amc.py) and
inference, so the deployed model sees exactly what it was trained on.

Baselines kept for comparison:
  centroid_features   vectorised copy of the repo's original CentroidAMC features
  cumulant_features   normalised fourth-order cumulants, Swami and Sadler (2000)

Proposed (DemodAMC):
  1. Remove the carrier offset first with SpecCFO, so the classifier never has
     to learn every possible rotation speed.
  2. Two views: a time branch on (I, Q, amplitude, instantaneous frequency)
     and a spectral branch on the x**M spectra, which expose the modulation
     order directly (BPSK has a line at M=2, QPSK at M=4, 8PSK at M=8).
  3. Global pooling, so any capture length works and longer captures help.
  4. Temperature-scaled probabilities and an abstention threshold chosen on
     held-out data, so "confidence" means something.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cfo_estimators import _gelu, default_model, spec_features
from signal_sim import MODULATIONS

_MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
MODEL_PATH = _MODELS / "demod_amc_v2.npz" if (_MODELS / "demod_amc_v2.npz").exists() else _MODELS / "demod_amc.npz"
CLASSES = MODULATIONS


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.complex128))
    x = x - x.mean(axis=1, keepdims=True)
    return x / (np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True)) + 1e-12)


def derotate(x: np.ndarray, cfo: np.ndarray) -> np.ndarray:
    n = np.arange(x.shape[1])
    return x * np.exp(-2j * np.pi * np.asarray(cfo)[:, None] * n[None, :])


def time_features(x: np.ndarray) -> np.ndarray:
    """(B, 4, L): I, Q, centred amplitude, instantaneous frequency in cycles/sample * 2."""
    x = _unit(x)
    amp = np.abs(x)
    dphi = np.angle(x[:, 1:] * np.conj(x[:, :-1])) / np.pi
    dphi = np.concatenate([dphi[:, :1], dphi], axis=1)
    return np.stack([x.real, x.imag, amp - amp.mean(axis=1, keepdims=True), dphi], axis=1).astype(np.float32)


def windows(x: np.ndarray, size: int = 128) -> np.ndarray:
    """Split (B, L) into (B, L//size, size) non-overlapping windows."""
    count = x.shape[1] // size
    return x[:, :count * size].reshape(x.shape[0], count, size)


def iq_windows(x: np.ndarray, size: int = 128) -> np.ndarray:
    w = windows(np.atleast_2d(x), size)
    w = w - w.mean(axis=2, keepdims=True)
    w = w / (np.sqrt(np.mean(np.abs(w) ** 2, axis=2, keepdims=True)) + 1e-12)
    return np.stack([w.real, w.imag], axis=2).astype(np.float32)  # (B, W, 2, size)


def ap_windows(x: np.ndarray, size: int = 128) -> np.ndarray:
    """Rajendran et al. (2018) input: L2-normalised amplitude and phase / pi."""
    w = windows(np.atleast_2d(x), size)
    amp = np.abs(w)
    amp = amp / (np.linalg.norm(amp, axis=2, keepdims=True) + 1e-12)
    return np.stack([amp, np.angle(w) / np.pi], axis=2).astype(np.float32)


# ------------------------------------------------------------------ baselines
def centroid_features(x: np.ndarray) -> np.ndarray:
    """Vectorised, numerically identical copy of modulation_classifier.features()."""
    v = np.atleast_2d(np.asarray(x, dtype=np.complex128))
    v = v - v.mean(axis=1, keepdims=True)
    mag = np.abs(v)
    mean_mag = mag.mean(axis=1)
    cv = np.sqrt(np.mean((mag - mean_mag[:, None]) ** 2, axis=1)) / np.maximum(mean_mag, 1e-12)
    ph = np.arctan2(v.imag, v.real)
    steps = (ph[:, 1:] - ph[:, :-1] + np.pi) % (2 * np.pi) - np.pi
    phase_std = np.sqrt(np.mean((steps - steps.mean(axis=1, keepdims=True)) ** 2, axis=1))
    return np.stack([cv, phase_std, np.abs(np.mean(v**2, axis=1)), np.abs(np.mean(v**4, axis=1)), np.mean(mag**2, axis=1)], axis=1)


class NearestCentroid:
    def fit(self, feats, labels):
        self.mean, self.scale = feats.mean(axis=0), feats.std(axis=0) + 1e-12
        z = (feats - self.mean) / self.scale
        self.classes = np.unique(labels)
        self.centroids = np.stack([z[labels == c].mean(axis=0) for c in self.classes])
        return self

    def distances(self, feats):
        z = (feats - self.mean) / self.scale
        return np.linalg.norm(z[:, None, :] - self.centroids[None], axis=2)

    def predict(self, feats):
        return self.classes[np.argmin(self.distances(feats), axis=1)]


THEORETICAL_CUMULANTS = {"bpsk": (2.0, -2.0), "qpsk": (1.0, -1.0), "8psk": (0.0, -1.0), "16qam": (0.68, -0.68), "64qam": (0.619, -0.619)}


def cumulant_features(x: np.ndarray) -> np.ndarray:
    """(|C40|, C42) normalised by C21**2 (Swami and Sadler 2000)."""
    v = np.atleast_2d(np.asarray(x, dtype=np.complex128))
    v = v - v.mean(axis=1, keepdims=True)
    m20, m21 = np.mean(v**2, axis=1), np.mean(np.abs(v) ** 2, axis=1)
    m40, m42 = np.mean(v**4, axis=1), np.mean(np.abs(v) ** 4, axis=1)
    c40 = m40 - 3 * m20**2
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21**2
    return np.stack([np.abs(c40) / m21**2, c42 / m21**2], axis=1)


def cumulant_classify(x: np.ndarray) -> np.ndarray:
    """Minimum distance to theoretical cumulants. Linear modulations only, as in the paper."""
    f = cumulant_features(x)
    names = list(THEORETICAL_CUMULANTS)
    ref = np.array([THEORETICAL_CUMULANTS[n] for n in names])
    best = np.argmin(np.linalg.norm(f[:, None, :] - ref[None], axis=2), axis=1)
    return np.array([CLASSES.index(names[i]) for i in best])


# ------------------------------------------------------------------ numpy inference of DemodAMC
def _conv_same(h: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int) -> np.ndarray:
    """Match jax.lax.conv_general_dilated 'SAME' padding for stride >= 1."""
    length, k = h.shape[2], w.shape[2]
    out_len = -(-length // stride)
    total = max((out_len - 1) * stride + k - length, 0)
    lo = total // 2
    hp = np.pad(h, ((0, 0), (0, 0), (lo, total - lo)))
    out = np.zeros((h.shape[0], w.shape[0], out_len), dtype=np.float32)
    for j in range(k):
        out += np.einsum("bcl,oc->bol", hp[:, :, j:j + stride * out_len:stride], w[:, :, j], optimize=True)
    return out + b[None, :, None]


class DemodAMC:
    def __init__(self, path: Path = MODEL_PATH):
        data = np.load(path, allow_pickle=False)
        self.p = {k: data[k] for k in data.files if k != "meta"}
        self.meta = json.loads(str(data["meta"]))
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.abstain_below = float(self.meta.get("abstain_below", 0.0))
        self.compensate = bool(self.meta.get("cfo_compensation", True))

    def inputs(self, x: np.ndarray, cfo: np.ndarray | None = None):
        x = np.atleast_2d(x)
        if self.compensate:
            if cfo is None:
                cfo = default_model().estimate(x)["cfo"]
            x = derotate(x, cfo)
        return time_features(x), spec_features(x)

    def logits(self, time_x, spec_x):
        p = self.p
        h = time_x
        for i, s in enumerate((1, 2, 2, 2)):
            h = _gelu(_conv_same(h, p[f"params/t{i}/w"], p[f"params/t{i}/b"], s))
        tf = np.concatenate([h.mean(axis=2), h.max(axis=2)], axis=1)
        s_ = spec_x
        for i, s in enumerate((2, 2, 2)):
            s_ = _gelu(_conv_same(s_, p[f"params/s{i}/w"], p[f"params/s{i}/b"], s))
        sf = np.concatenate([s_.mean(axis=2), s_.max(axis=2)], axis=1)
        z = _gelu(np.concatenate([tf, sf], axis=1) @ p["params/d1/w"] + p["params/d1/b"])
        return z @ p["params/d2/w"] + p["params/d2/b"]

    def probabilities(self, x: np.ndarray, cfo: np.ndarray | None = None) -> np.ndarray:
        z = self.logits(*self.inputs(x, cfo)) / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


_DEFAULT: DemodAMC | None = None


def default_model_amc() -> DemodAMC | None:
    global _DEFAULT
    if _DEFAULT is None and MODEL_PATH.exists():
        _DEFAULT = DemodAMC(MODEL_PATH)
    return _DEFAULT
