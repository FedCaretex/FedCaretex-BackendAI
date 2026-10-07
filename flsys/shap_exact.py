"""SHAP eksak atas 8 fitur asli (one-hot dikelompokkan kembali).

Dengan 8 fitur, Shapley eksak = 2^8 = 256 evaluasi model: murah, deterministik,
dan dapat diimplementasikan di perangkat (Dart) tanpa aproksimasi.
Nilai koalisi v(S): fitur di S memakai nilai x, fitur di luar S memakai BASELINE
(rata-rata fitur data validasi, dalam ruang encoded; dikirim bersama model).
Properti efisiensi: sum(phi) = f(x) - f(baseline).
"""

from __future__ import annotations

from itertools import product
from math import factorial

import numpy as np

from .data import FEATURE_GROUPS, RAW_FEATURES
from .model import predict_proba

N_G = len(RAW_FEATURES)  # 8


def _coalition_masks() -> np.ndarray:
    return np.array(list(product([0, 1], repeat=N_G)), dtype=np.int8)  # (256, 8)


_MASKS = _coalition_masks()


def _mix(x: np.ndarray, baseline: np.ndarray, mask_row: np.ndarray) -> np.ndarray:
    out = baseline.copy()
    for g, name in enumerate(RAW_FEATURES):
        if mask_row[g]:
            cols = FEATURE_GROUPS[name]
            out[cols] = x[cols]
    return out


def coalition_values(params: np.ndarray, x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    batch = np.stack([_mix(x, baseline, m) for m in _MASKS]).astype(np.float32)
    return predict_proba(params, batch).astype(np.float64)  # (256,)


def exact_shap(params: np.ndarray, x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Return phi (8,) dalam satuan probabilitas, urutan = RAW_FEATURES."""
    v = coalition_values(params, x, baseline)
    index = {tuple(m): i for i, m in enumerate(_MASKS)}
    phi = np.zeros(N_G)
    nf = factorial(N_G)
    for g in range(N_G):
        for i, m in enumerate(_MASKS):
            if m[g]:
                continue
            s = int(m.sum())
            w = factorial(s) * factorial(N_G - s - 1) / nf
            m2 = m.copy()
            m2[g] = 1
            phi[g] += w * (v[index[tuple(m2)]] - v[i])
    return phi


def baseline_from(X_val: np.ndarray) -> np.ndarray:
    return X_val.mean(axis=0).astype(np.float32)
