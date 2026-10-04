"""JAX definitions of the recreated paper models and DEmod's proposed models.

Recreations follow the architectures as published. Where a paper leaves a
hyperparameter unstated, the choice made here is written next to it and in
research/README.md.

CFO
  iq_resnet     Chen, Zheng, Zhu, Xuan, Yang (2023), "Deep Learning-Based Frequency
                Offset Estimation", arXiv:2311.16155. Conv16 -> Res(16) -> Res(32,/2)
                -> Res(64,/2) -> FC(1), BN+ReLU, MSE. Kernel size not stated: 3.
  oshea_cfo     O'Shea, Karra, Clancy (2017), "Learning Approximate Neural Estimators
                for Wireless Channel State Information", arXiv:1707.06260, Table II.
                Conv(32)+ReLU, AvgPool, Conv(128)+ReLU, Conv(256)+ReLU, Linear(1), MSE.
                Kernel sizes and strides not stated: k=8, pool 4, strides 4.
  speccfo       Proposed. Circular dilated residual CNN over the shared-CFO-axis
                spectra from cfo_estimators.spec_features, softmax over CFO bins.

AMC
  vtcnn2        O'Shea, Corgan, Clancy (2016), "Convolutional Radio Modulation
                Recognition Networks", EANN. Conv(256,1x3) -> Conv(80,2x3) ->
                Dense(256) -> Dense(classes), ReLU, dropout 0.5, input 2x128.
  lstm_ap       Rajendran et al. (2018), "Deep Learning Models for Wireless Signal
                Classification With Distributed Low-Cost Spectrum Sensors", IEEE TCCN.
                Two LSTM layers of 128 cells over (amplitude, phase), input 128 steps.
  demod_amc     Proposed. CFO-compensated multi-view network: a time branch on
                (I, Q, amplitude, instantaneous frequency) and a spectral branch on
                the x**M spectra, fused, with logits averaged over windows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

DN = ("NCH", "OIH", "NCH")


def _he(key, shape, fan_in):
    return jax.random.normal(key, shape) * jnp.sqrt(2.0 / fan_in)


def conv(x, w, b=None, stride=1, dilation=1, padding="SAME"):
    y = jax.lax.conv_general_dilated(x, w, (stride,), padding, rhs_dilation=(dilation,), dimension_numbers=DN)
    return y if b is None else y + b[None, :, None]


def conv_circular(x, w, b, dilation=1):
    k = w.shape[2]
    pad = dilation * (k - 1) // 2
    xp = jnp.concatenate([x[:, :, -pad:], x, x[:, :, :pad]], axis=2) if pad else x
    return conv(xp, w, b, dilation=dilation, padding="VALID")


def batchnorm(x, p, s, train, momentum=0.9):
    if train:
        mean, var = x.mean(axis=(0, 2)), x.var(axis=(0, 2))
        s = {"mean": momentum * s["mean"] + (1 - momentum) * mean, "var": momentum * s["var"] + (1 - momentum) * var}
    else:
        mean, var = s["mean"], s["var"]
    y = (x - mean[None, :, None]) / jnp.sqrt(var[None, :, None] + 1e-5)
    return y * p["g"][None, :, None] + p["b"][None, :, None], s


def _bn_init(c):
    return {"g": jnp.ones(c), "b": jnp.zeros(c)}, {"mean": jnp.zeros(c), "var": jnp.ones(c)}


# ------------------------------------------------------------------ IQ-ResNet (Chen 2023)
def iq_resnet_init(key, length=1024, k=3):
    keys = iter(jax.random.split(key, 32))
    params, state = {}, {}
    params["c0"] = {"w": _he(next(keys), (16, 2, k), 2 * k), "b": jnp.zeros(16)}
    params["bn0"], state["bn0"] = _bn_init(16)
    cin = 16
    for name, cout, stride in (("r1", 16, 1), ("r2", 32, 2), ("r3", 64, 2)):
        blk, st = {}, {}
        blk["w1"] = _he(next(keys), (cout, cin, k), cin * k)
        blk["bn1"], st["bn1"] = _bn_init(cout)
        blk["w2"] = _he(next(keys), (cout, cout, k), cout * k)
        blk["bn2"], st["bn2"] = _bn_init(cout)
        if stride != 1 or cin != cout:
            blk["proj"] = _he(next(keys), (cout, cin, 1), cin)
        params[name], state[name], cin = blk, st, cout
    flat = 64 * (length // 4)
    params["fc"] = {"w": jax.random.normal(next(keys), (flat, 1)) * jnp.sqrt(1.0 / flat), "b": jnp.zeros(1)}
    return params, state


def iq_resnet_apply(params, state, x, train):
    new = {}
    h = conv(x, params["c0"]["w"], params["c0"]["b"])
    h, new["bn0"] = batchnorm(h, params["bn0"], state["bn0"], train)
    h = jax.nn.relu(h)
    for name, stride in (("r1", 1), ("r2", 2), ("r3", 2)):
        p, s = params[name], state[name]
        y = conv(h, p["w1"], stride=stride)
        y, s1 = batchnorm(y, p["bn1"], s["bn1"], train)
        y = conv(jax.nn.relu(y), p["w2"])
        y, s2 = batchnorm(y, p["bn2"], s["bn2"], train)
        skip = conv(h, p["proj"], stride=stride) if "proj" in p else h
        h = jax.nn.relu(y + skip)
        new[name] = {"bn1": s1, "bn2": s2}
    out = h.reshape(h.shape[0], -1) @ params["fc"]["w"] + params["fc"]["b"]
    return out[:, 0], new


# ------------------------------------------------------------------ O'Shea 2017 CFO CNN
def oshea_cfo_init(key, length=1024, k=8):
    keys = jax.random.split(key, 4)
    after = length // 4 // 4 // 4
    return {
        "c1": {"w": _he(keys[0], (32, 2, k), 2 * k), "b": jnp.zeros(32)},
        "c2": {"w": _he(keys[1], (128, 32, k), 32 * k), "b": jnp.zeros(128)},
        "c3": {"w": _he(keys[2], (256, 128, k), 128 * k), "b": jnp.zeros(256)},
        "fc": {"w": jax.random.normal(keys[3], (256 * after, 1)) * jnp.sqrt(1.0 / (256 * after)), "b": jnp.zeros(1)},
    }, {}


def oshea_cfo_apply(params, state, x, train):
    h = jax.nn.relu(conv(x, params["c1"]["w"], params["c1"]["b"]))
    h = h.reshape(h.shape[0], h.shape[1], -1, 4).mean(axis=3)
    h = jax.nn.relu(conv(h, params["c2"]["w"], params["c2"]["b"], stride=4))
    h = jax.nn.relu(conv(h, params["c3"]["w"], params["c3"]["b"], stride=4))
    out = h.reshape(h.shape[0], -1) @ params["fc"]["w"] + params["fc"]["b"]
    return out[:, 0], state


# ------------------------------------------------------------------ SpecCFO (proposed)
SPEC_DILATIONS = (1, 2, 4, 8, 16, 32, 1)


COORD_CHANNELS = 4


def coord_channels(length):
    """sin and cos of 2*pi*f and 4*pi*f over the shared CFO axis. They give the network its absolute position,
    which a translation-equivariant network otherwise lacks, so it can learn that offsets beyond +-0.2 never occur
    and pick the one in-range alias of an x^M line."""
    f = (jnp.arange(length) - length // 2) / length
    return jnp.stack([jnp.sin(2 * jnp.pi * f), jnp.cos(2 * jnp.pi * f), jnp.sin(4 * jnp.pi * f), jnp.cos(4 * jnp.pi * f)])


def speccfo_init(key, channels=24, k=5, inputs=4, coords=False):
    inputs = inputs + (COORD_CHANNELS if coords else 0)
    keys = iter(jax.random.split(key, 16))
    p = {"in_w": _he(next(keys), (channels, inputs, k), inputs * k), "in_b": jnp.zeros(channels)}
    for i, _ in enumerate(SPEC_DILATIONS):
        p[f"b{i}_w"] = _he(next(keys), (channels, channels, k), channels * k) * 0.5
        p[f"b{i}_b"] = jnp.zeros(channels)
    p["out_w"] = _he(next(keys), (1, channels, 1), channels) * 0.1
    p["out_b"] = jnp.zeros(1)
    return p, {}


def speccfo_apply(params, state, feats, train):
    if params["in_w"].shape[1] == feats.shape[1] + COORD_CHANNELS:
        feats = jnp.concatenate([feats, jnp.broadcast_to(coord_channels(feats.shape[2])[None], (feats.shape[0], COORD_CHANNELS, feats.shape[2]))], axis=1)
    h = conv_circular(feats, params["in_w"], params["in_b"])
    for i, d in enumerate(SPEC_DILATIONS):
        h = h + jax.nn.gelu(conv_circular(h, params[f"b{i}_w"], params[f"b{i}_b"], d))
    h = jax.nn.gelu(h)
    return conv_circular(h, params["out_w"], params["out_b"])[:, 0, :], state


# ------------------------------------------------------------------ VT-CNN2 (O'Shea 2016)
def vtcnn2_init(key, classes, length=128):
    keys = jax.random.split(key, 4)
    # Conv2D(256, 1x3) on a 2xL "image" is a per-row Conv1D shared over I and Q.
    return {
        "c1": {"w": _he(keys[0], (256, 1, 3), 3), "b": jnp.zeros(256)},
        "c2": {"w": _he(keys[1], (80, 256 * 2, 3), 256 * 2 * 3), "b": jnp.zeros(80)},
        "d1": {"w": _he(keys[2], (80 * length, 256), 80 * length), "b": jnp.zeros(256)},
        "d2": {"w": jax.random.normal(keys[3], (256, classes)) * jnp.sqrt(1.0 / 256), "b": jnp.zeros(classes)},
    }, {}


def _dropout(key, h, rate, train):
    if not train or key is None:
        return h
    keep = jax.random.bernoulli(key, 1 - rate, h.shape)
    return jnp.where(keep, h / (1 - rate), 0.0)


def vtcnn2_apply(params, state, x, train, key=None):
    b, _, length = x.shape
    keys = jax.random.split(key, 3) if key is not None else (None, None, None)
    rows = x.reshape(b * 2, 1, length)  # zero-padded 1x3 conv on each of the I and Q rows
    h = jax.nn.relu(conv(rows, params["c1"]["w"], params["c1"]["b"]))
    h = h.reshape(b, 2, 256, length).transpose(0, 2, 1, 3).reshape(b, 512, length)
    h = _dropout(keys[0], h, 0.5, train)
    h = jax.nn.relu(conv(h, params["c2"]["w"], params["c2"]["b"]))  # 2x3 kernel spans both rows
    h = _dropout(keys[1], h, 0.5, train)
    h = jax.nn.relu(h.reshape(b, -1) @ params["d1"]["w"] + params["d1"]["b"])
    h = _dropout(keys[2], h, 0.5, train)
    return h @ params["d2"]["w"] + params["d2"]["b"], state


# ------------------------------------------------------------------ LSTM amplitude/phase (Rajendran 2018)
def _lstm_init(key, n_in, n_hid):
    k1, k2 = jax.random.split(key)
    return {"wx": jax.random.normal(k1, (n_in, 4 * n_hid)) * jnp.sqrt(1.0 / n_in), "wh": jax.random.normal(k2, (n_hid, 4 * n_hid)) * jnp.sqrt(1.0 / n_hid), "b": jnp.zeros(4 * n_hid).at[n_hid:2 * n_hid].set(1.0)}


def _lstm_run(p, seq):
    n_hid = p["wh"].shape[0]

    def step(carry, xt):
        h, c = carry
        z = xt @ p["wx"] + h @ p["wh"] + p["b"]
        i, f, g, o = (jax.nn.sigmoid(z[:, :n_hid]), jax.nn.sigmoid(z[:, n_hid:2 * n_hid]), jnp.tanh(z[:, 2 * n_hid:3 * n_hid]), jax.nn.sigmoid(z[:, 3 * n_hid:]))
        c = f * c + i * g
        h = o * jnp.tanh(c)
        return (h, c), h

    b = seq.shape[1]
    (h, _), hs = jax.lax.scan(step, (jnp.zeros((b, n_hid)), jnp.zeros((b, n_hid))), seq)
    return hs, h


def lstm_ap_init(key, classes, hidden=128):
    k1, k2, k3 = jax.random.split(key, 3)
    return {"l1": _lstm_init(k1, 2, hidden), "l2": _lstm_init(k2, hidden, hidden), "d": {"w": jax.random.normal(k3, (hidden, classes)) * jnp.sqrt(1.0 / hidden), "b": jnp.zeros(classes)}}, {}


def lstm_ap_apply(params, state, ap, train, key=None):
    seq = ap.transpose(2, 0, 1)  # (L, B, 2)
    hs, _ = _lstm_run(params["l1"], seq)
    _, h = _lstm_run(params["l2"], hs)
    return h @ params["d"]["w"] + params["d"]["b"], state


# ------------------------------------------------------------------ DEmod AMC (proposed)
def demod_amc_init(key, classes, ch=48, spec_in=4, time_in=4):
    keys = iter(jax.random.split(key, 24))
    p = {}
    cin = time_in
    for i, (cout, k) in enumerate(((ch, 7), (ch, 5), (2 * ch, 5), (2 * ch, 3))):
        p[f"t{i}"] = {"w": _he(next(keys), (cout, cin, k), cin * k), "b": jnp.zeros(cout)}
        cin = cout
    cin = spec_in
    for i, (cout, k) in enumerate(((ch, 7), (ch, 5), (2 * ch, 5))):
        p[f"s{i}"] = {"w": _he(next(keys), (cout, cin, k), cin * k), "b": jnp.zeros(cout)}
        cin = cout
    fused = 2 * ch * 2 + 2 * ch * 2  # mean+max pooling of both branches
    p["d1"] = {"w": _he(next(keys), (fused, 128), fused), "b": jnp.zeros(128)}
    p["d2"] = {"w": jax.random.normal(next(keys), (128, classes)) * jnp.sqrt(1.0 / 128), "b": jnp.zeros(classes)}
    return p, {}


def demod_amc_apply(params, state, inputs, train, key=None):
    time_x, spec_x = inputs
    h = time_x
    for i, stride in enumerate((1, 2, 2, 2)):
        h = jax.nn.gelu(conv(h, params[f"t{i}"]["w"], params[f"t{i}"]["b"], stride=stride))
    t_feat = jnp.concatenate([h.mean(axis=2), h.max(axis=2)], axis=1)
    s = spec_x
    for i, stride in enumerate((2, 2, 2)):
        s = jax.nn.gelu(conv(s, params[f"s{i}"]["w"], params[f"s{i}"]["b"], stride=stride))
    s_feat = jnp.concatenate([s.mean(axis=2), s.max(axis=2)], axis=1)
    z = jnp.concatenate([t_feat, s_feat], axis=1)
    z = _dropout(key, jax.nn.gelu(z @ params["d1"]["w"] + params["d1"]["b"]), 0.3, train)
    return z @ params["d2"]["w"] + params["d2"]["b"], state


def to_numpy(tree):
    return jax.tree_util.tree_map(lambda v: np.asarray(v), tree)
