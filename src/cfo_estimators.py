"""Carrier frequency offset (CFO) estimators for DEmod.

All estimators work in normalised frequency (cycles per sample). Multiply by
the sample rate for Hz. The repository's first learned estimator regressed
absolute Hz from features that themselves depended on the sample rate, which
is why it produced impossible values (beyond +-fs/2) on captures recorded at a
different rate from its training recipes. Normalised units remove that failure.

Classical baselines (batch-vectorised, numpy):
  kay_wpa           Kay (1989) weighted phase averager, optionally on x**M
  lag1_autocorr     lag-one autocorrelation phase on x**M (the repo's old DSP branch)
  luise_reggiannini Luise and Reggiannini (1995), used as a fine stage
  power_periodogram M-th power periodogram peak with interpolation, the "expert
                    estimator" baseline in O'Shea, Karra and Clancy (2017)
  blind_multi_m     the periodogram with M chosen by spectral-line strength

Proposed estimator (SpecCFO):
  1. Physics-informed input: log power spectra of x**M for M in {1, 2, 4, 8},
     each re-indexed onto the same CFO axis (bin k of channel M holds the power
     at M*f_k), so every channel votes for the same location and M-fold
     ambiguities appear as aliases the network can see.
  2. A fully convolutional, circularly padded 1-D network over that axis.
     Shifting the CFO shifts the input, so the network is translation
     equivariant by construction and outputs a probability over CFO bins
     rather than a regressed number (no regression-to-the-mean, built-in
     confidence).
  3. A classical refinement around the network's coarse bin: zero-padded
     periodogram of x**M for each M, parabolic interpolation, and the M with
     the strongest line wins. This recovers near maximum-likelihood precision
     at high SNR, where purely learned regressors plateau.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPEC_BINS = 512
SPEC_ORDERS = (1, 2, 4, 8)
SPEC_FFT = 2048
_MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
MODEL_PATH = _MODELS / "speccfo_v2.npz" if (_MODELS / "speccfo_v2.npz").exists() else _MODELS / "speccfo.npz"


def _prep(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.complex128))
    x = x - x.mean(axis=1, keepdims=True)
    return x / (np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True)) + 1e-12)


def wrap(f: np.ndarray) -> np.ndarray:
    return (np.asarray(f) + 0.5) % 1.0 - 0.5


# ---------------------------------------------------------------- classical
def kay_wpa(x: np.ndarray, order: int = 1) -> np.ndarray:
    """Kay's weighted phase averager on x**order (Kay 1989), divided by order."""
    y = _prep(x) ** order
    n = y.shape[1]
    t = np.arange(n - 1)
    w = 1.5 * n / (n**2 - 1) * (1 - ((t - (n / 2 - 1)) / (n / 2)) ** 2)
    dphi = np.angle(y[:, 1:] * np.conj(y[:, :-1]))
    return wrap((dphi * w).sum(axis=1) / (2 * np.pi)) / order


def lag1_autocorr(x: np.ndarray, order: int = 4) -> np.ndarray:
    """Phase of the lag-one autocorrelation of x**order. The repo's previous DSP CFO."""
    y = _prep(x) ** order
    return np.angle(np.sum(y[:, 1:] * np.conj(y[:, :-1]), axis=1)) / (2 * np.pi * order)


def luise_reggiannini(x: np.ndarray, order: int = 1, lags: int | None = None) -> np.ndarray:
    """Luise and Reggiannini (1995) estimator on x**order. Unambiguous only for |f| < 1/(order*(L+1))."""
    y = _prep(x) ** order
    n = y.shape[1]
    lags = lags or n // 2
    acc = np.zeros(y.shape[0], dtype=complex)
    for m in range(1, lags + 1):
        acc += np.mean(y[:, m:] * np.conj(y[:, :-m]), axis=1)
    return np.angle(acc) / (np.pi * (lags + 1) * order)


def _parabolic(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Sub-bin offset from a parabola through log-power at k-1, k, k+1."""
    n = p.shape[1]
    rows = np.arange(p.shape[0])
    a, b, c = (np.log(p[rows, (k + d) % n] + 1e-30) for d in (-1, 0, 1))
    den = a - 2 * b + c
    return np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)


def power_periodogram(x: np.ndarray, order: int = 4, pad: int = 8, search: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """M-th power periodogram peak (O'Shea 2017 expert baseline, Rife-Boorstyn style).

    Returns (cfo, line_strength) where line_strength is peak over median power.
    `search` optionally limits the peak search to [lo, hi] in CFO units per row.
    """
    y = _prep(x) ** order
    nfft = y.shape[1] * pad
    p = np.abs(np.fft.fft(y, nfft, axis=1)) ** 2
    if search is None:
        k = np.argmax(p, axis=1)
    else:
        lo, hi = search
        k = np.empty(p.shape[0], dtype=int)
        for r in range(p.shape[0]):
            g = np.arange(int(np.floor(lo[r] * order * nfft)), int(np.ceil(hi[r] * order * nfft)) + 1)
            k[r] = g[np.argmax(p[r, g % nfft])] % nfft
    delta = _parabolic(p, k)
    g = wrap((k + delta) / nfft)
    # Line prominence against the local continuous spectrum, not the global
    # median: a modulated signal's own band would otherwise look like a line.
    half = max(8, nfft // 64)
    window = (k[:, None] + np.arange(-half, half + 1)[None, :]) % nfft
    local = np.median(np.take_along_axis(p, window, axis=1), axis=1)
    strength = p[np.arange(p.shape[0]), k] / (local + 1e-30)
    return g / order, strength


def blind_multi_m(x: np.ndarray, orders=SPEC_ORDERS) -> tuple[np.ndarray, np.ndarray]:
    """Classical blind baseline: periodogram for each M, keep the strongest relative line."""
    results = [power_periodogram(x, m) for m in orders]
    strengths = np.stack([s for _, s in results])
    best = np.argmax(strengths, axis=0)
    cfo = np.stack([f for f, _ in results])[best, np.arange(strengths.shape[1])]
    return cfo, np.array(orders)[best]


def tone_crb(snr_db: np.ndarray, n: int) -> np.ndarray:
    """Cramer-Rao bound std for an unmodulated tone (cycles/sample). Blind estimators on modulated signals cannot beat it."""
    snr = 10 ** (np.asarray(snr_db) / 10)
    return np.sqrt(6 / ((2 * np.pi) ** 2 * snr * n * (n**2 - 1)))


# ---------------------------------------------------------------- SpecCFO features
def _cell_index(order: int, bins: int = SPEC_BINS, nfft: int = SPEC_FFT) -> np.ndarray:
    f = (np.arange(bins) - bins // 2) / bins
    width = max(1, order * nfft // bins)
    start = np.round((order * (f - 0.5 / bins)) * nfft).astype(int)
    return (start[:, None] + np.arange(width)[None, :]) % nfft


_CELLS = {m: _cell_index(m) for m in SPEC_ORDERS}


def spec_features(x: np.ndarray, frame: int = 1024) -> np.ndarray:
    """(batch, len(SPEC_ORDERS), SPEC_BINS) float32 log spectra on a shared CFO axis.

    Longer inputs are split into frames whose power spectra are averaged
    (Bartlett), so a long capture sharpens the lines instead of being truncated.
    """
    x = _prep(x)
    usable = max(frame, (x.shape[1] // frame) * frame) if x.shape[1] >= frame else x.shape[1]
    frames = x[:, :usable].reshape(x.shape[0], -1, min(frame, usable))
    out = np.empty((x.shape[0], len(SPEC_ORDERS), SPEC_BINS), dtype=np.float32)
    for c, m in enumerate(SPEC_ORDERS):
        y = frames**m
        y = y / (np.sqrt(np.mean(np.abs(y) ** 2, axis=2, keepdims=True)) + 1e-12)
        p = np.mean(np.abs(np.fft.fft(y, SPEC_FFT, axis=2)) ** 2, axis=1)
        cell = p[:, _CELLS[m]].max(axis=2)
        db = 10 * np.log10(cell / (np.median(cell, axis=1, keepdims=True) + 1e-30) + 1e-12)
        out[:, c] = np.clip(db, -10, 40) / 10
    return out


def bins_to_cfo(index: np.ndarray) -> np.ndarray:
    return (np.asarray(index) - SPEC_BINS // 2) / SPEC_BINS


def cfo_to_bin(cfo: np.ndarray) -> np.ndarray:
    return (np.round(wrap(cfo) * SPEC_BINS).astype(int) + SPEC_BINS // 2) % SPEC_BINS


# ---------------------------------------------------------------- numpy inference
def _conv1d_circular(h: np.ndarray, w: np.ndarray, b: np.ndarray, dilation: int) -> np.ndarray:
    """h (B,Cin,L), w (Cout,Cin,K): circular 'same' convolution matching the JAX trainer."""
    k = w.shape[2]
    pad = dilation * (k - 1) // 2
    hp = np.concatenate([h[:, :, -pad:], h, h[:, :, :pad]], axis=2) if pad else h
    length = h.shape[2]
    out = np.zeros((h.shape[0], w.shape[0], length), dtype=np.float32)
    for j in range(k):
        out += np.einsum("bcl,oc->bol", hp[:, :, j * dilation:j * dilation + length], w[:, :, j], optimize=True)
    return out + b[None, :, None]


def _gelu(v):
    return 0.5 * v * (1 + np.tanh(0.7978845608 * (v + 0.044715 * v**3)))


class SpecCFO:
    """Trained SpecCFO network plus classical refinement. Pure numpy at inference."""

    def __init__(self, path: Path = MODEL_PATH):
        data = np.load(path, allow_pickle=False)
        self.params = {k.removeprefix("params/"): data[k] for k in data.files if k != "meta" and not k.startswith("state/")}
        self.meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
        self.dilations = self.meta.get("dilations", [1, 2, 4, 8, 16, 32, 1])

    logits_override = None  # set to a faster callable (e.g. JIT-compiled JAX) by the trainers

    def logits(self, feats: np.ndarray) -> np.ndarray:
        if self.logits_override is not None:
            return np.asarray(self.logits_override(feats))
        p = self.params
        if p["in_w"].shape[1] == feats.shape[1] + 4:  # model trained with absolute-position channels
            f = (np.arange(feats.shape[2]) - feats.shape[2] // 2) / feats.shape[2]
            coords = np.stack([np.sin(2 * np.pi * f), np.cos(2 * np.pi * f), np.sin(4 * np.pi * f), np.cos(4 * np.pi * f)]).astype(np.float32)
            feats = np.concatenate([feats, np.broadcast_to(coords[None], (feats.shape[0], 4, feats.shape[2]))], axis=1)
        h = _conv1d_circular(feats, p["in_w"], p["in_b"], 1)
        for i, d in enumerate(self.dilations):
            r = _gelu(_conv1d_circular(h, p[f"b{i}_w"], p[f"b{i}_b"], d))
            h = h + r
        h = _gelu(h)
        return _conv1d_circular(h, p["out_w"], p["out_b"], 1)[:, 0, :]

    def coarse(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = self.logits(spec_features(x))
        z = z - z.max(axis=1, keepdims=True)
        prob = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        k = prob.argmax(axis=1)
        delta = _parabolic(prob + 1e-12, k)
        rows = np.arange(len(k))
        conf = sum(prob[rows, (k + d) % SPEC_BINS] for d in (-2, -1, 0, 1, 2))
        return wrap(bins_to_cfo(k + delta)), conf, prob

    def estimate(self, x: np.ndarray, refine: bool = True, min_line: float = 12.0, fallback_below: float | None = None) -> dict[str, np.ndarray]:
        """Coarse network estimate, then periodogram refinement on the strongest x**M line near it.

        `min_line` (line-to-local-median power) and `fallback_below` were tuned on a
        validation set that is disjoint from the reported test sets. Below
        `fallback_below` confidence the circular posterior mean replaces the
        peak, which lowers squared error when the network is unsure.
        """
        if fallback_below is None:
            fallback_below = float(self.meta.get("fallback_below", 0.2))
        f0, conf, prob = self.coarse(x)
        grid = bins_to_cfo(np.arange(SPEC_BINS))
        posterior_mean = np.angle((prob * np.exp(2j * np.pi * grid)).sum(axis=1)) / (2 * np.pi)
        f0 = np.where(conf >= fallback_below, f0, posterior_mean)
        if not refine:
            return {"cfo": f0, "confidence": conf, "order": np.zeros_like(f0, dtype=int), "refined": np.zeros_like(f0, dtype=bool)}
        best_f, best_s, best_m = f0.copy(), np.zeros_like(f0), np.zeros(len(f0), dtype=int)
        cell = 1.5 / SPEC_BINS
        for m in SPEC_ORDERS:
            f, s = power_periodogram(x, m, search=(f0 - cell, f0 + cell))
            f = f0 + wrap(m * (f - f0)) / m
            take = s > best_s
            best_f[take], best_s[take], best_m[take] = f[take], s[take], m
        refined = (best_s >= min_line) & (conf >= fallback_below)
        return {"cfo": np.where(refined, best_f, f0), "confidence": conf, "order": np.where(refined, best_m, 0), "refined": refined, "line_strength": best_s}


_DEFAULT: SpecCFO | None = None


def default_model() -> SpecCFO | None:
    global _DEFAULT
    if _DEFAULT is None and MODEL_PATH.exists():
        _DEFAULT = SpecCFO(MODEL_PATH)
    return _DEFAULT
