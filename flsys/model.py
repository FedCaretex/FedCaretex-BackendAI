"""MLP numpy 15 -> 32 -> 16 -> 1 (sigmoid), backprop manual.

Dipilih numpy murni (bukan TF/PyTorch) supaya: (1) arsitektur identik dapat
diimplementasikan 1:1 di Dart untuk inferensi/SHAP/training on-device,
(2) serialisasi bobot deterministik untuk hashing, (3) ringan (~1.000 parameter).
"""

from __future__ import annotations

import numpy as np

LAYERS = (15, 32, 16, 1)


def param_shapes(layers=LAYERS):
    shapes = []
    for a, b in zip(layers[:-1], layers[1:]):
        shapes.append((a, b))  # W
        shapes.append((b,))  # b
    return shapes


def n_params(layers=LAYERS) -> int:
    return int(sum(int(np.prod(s)) for s in param_shapes(layers)))


def init_params(seed: int = 0, layers=LAYERS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for a, b in zip(layers[:-1], layers[1:]):
        W = rng.normal(0, np.sqrt(2.0 / a), size=(a, b))  # He init
        chunks.append(W.ravel())
        chunks.append(np.zeros(b))
    return np.concatenate(chunks).astype(np.float32)


def unpack(flat: np.ndarray, layers=LAYERS):
    out, i = [], 0
    for a, b in zip(layers[:-1], layers[1:]):
        W = flat[i : i + a * b].reshape(a, b)
        i += a * b
        bias = flat[i : i + b]
        i += b
        out.append((W, bias))
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def predict_proba(flat: np.ndarray, X: np.ndarray, layers=LAYERS) -> np.ndarray:
    h = X.astype(np.float32)
    ps = unpack(flat, layers)
    for W, b in ps[:-1]:
        h = np.maximum(h @ W + b, 0.0)
    W, b = ps[-1]
    return _sigmoid(h @ W + b).ravel()


def _loss_grad(flat, X, y, pos_weight, layers=LAYERS):
    ps = unpack(flat, layers)
    acts, pre = [X], []
    h = X
    for W, b in ps[:-1]:
        z = h @ W + b
        pre.append(z)
        h = np.maximum(z, 0.0)
        acts.append(h)
    W, b = ps[-1]
    p = _sigmoid(h @ W + b).ravel()
    w = np.where(y > 0.5, pos_weight, 1.0)
    eps = 1e-7
    loss = -np.mean(w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    dz = (w * (p - y) / len(y))[:, None]  # dLoss/dlogit
    grads = [None] * len(ps)
    grads[-1] = (acts[-1].T @ dz, dz.sum(0))
    dh = dz @ ps[-1][0].T
    for li in range(len(ps) - 2, -1, -1):
        dzl = dh * (pre[li] > 0)
        grads[li] = (acts[li].T @ dzl, dzl.sum(0))
        if li > 0:
            dh = dzl @ ps[li][0].T
    g = np.concatenate([np.concatenate([gw.ravel(), gb]) for gw, gb in grads])
    return float(loss), g.astype(np.float32)


def train_local(
    flat0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    prox_mu: float = 0.0,
    w_global: np.ndarray | None = None,
    layers=LAYERS,
) -> np.ndarray:
    """SGD minibatch dengan class weight lokal. prox_mu>0 -> FedProx:
    gradien ditambah mu * (w - w_global) (turunan dari (mu/2)||w - w_global||^2)."""
    rng = np.random.default_rng(seed)
    w = flat0.copy()
    npos = float((y > 0.5).sum())
    nneg = float(len(y) - npos)
    pos_weight = (nneg / npos) if npos > 0 else 1.0
    pos_weight = float(np.clip(pos_weight, 1.0, 20.0))
    ref = w_global if w_global is not None else flat0
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch_size):
            b = order[s : s + batch_size]
            _, g = _loss_grad(w, X[b], y[b], pos_weight, layers)
            if prox_mu > 0:
                g = g + prox_mu * (w - ref)
            w = w - lr * g
    return w
